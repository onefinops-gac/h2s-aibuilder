import os
import re
from io import BytesIO
from datetime import date, datetime, timedelta, timezone

from flask import Flask, jsonify, render_template, request, session, make_response
import firebase_admin
from firebase_admin import auth as firebase_auth
from google.cloud import bigquery


# ============================================================
# Configuration
# ============================================================

def required_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )
    return value


# Deployment configuration is supplied by Cloud Run environment variables.
PROJECT_ID = required_env("MGMT_PROJECT_ID")
DATASET = required_env("BQ_DATASET")
KS_DATASET = required_env("KS_DATASET")
FIREBASE_PROJECT_ID = required_env("FIREBASE_PROJECT_ID")

BQ_DATA_MODE = required_env("BQ_DATA_MODE").lower()
if BQ_DATA_MODE not in {"real", "demo", "combined"}:
    raise RuntimeError(
        "BQ_DATA_MODE must be one of: real, demo, combined"
    )

_TABLE_ENV_BY_MODE = {
    "real": "BQ_TABLE_NAME",
    "demo": "BQ_DEMO_TABLE_NAME",
    "combined": "BQ_COMBINED_TABLE_NAME",
}
BQ_SELECTED_TABLE = required_env(_TABLE_ENV_BY_MODE[BQ_DATA_MODE])
TABLE = f"{PROJECT_ID}.{DATASET}.{BQ_SELECTED_TABLE}"

DATA_START_DATE = required_env("DATA_START_DATE")
try:
    date.fromisoformat(DATA_START_DATE)
except ValueError as exc:
    raise RuntimeError(
        "DATA_START_DATE must be YYYY-MM-DD"
    ) from exc
DATA_START_SQL = f"DATE '{DATA_START_DATE}'"

BQ_COST_SOURCE = required_env("BQ_COST_SOURCE").lower()
if BQ_COST_SOURCE not in {"stored", "computed"}:
    raise RuntimeError(
        "BQ_COST_SOURCE must be one of: stored, computed"
    )

if BQ_COST_SOURCE == "stored":
    EVENT_COUNT_EXPR = "SUM(COALESCE(request_count, 1))"
else:
    EVENT_COUNT_EXPR = "COUNT(*)"

if BQ_DATA_MODE == "combined":
    COST_AGGREGATES = """
          SUM(IF(data_source = 'REAL', COALESCE(prompt_tokens, 0), 0)) AS real_prompt_tokens,
          SUM(IF(data_source = 'REAL', COALESCE(completion_tokens, 0), 0)) AS real_completion_tokens,
          SUM(IF(data_source = 'REAL', COALESCE(thoughts_tokens, 0), 0)) AS real_thoughts_tokens,
          SUM(IF(data_source = 'REAL', COALESCE(cached_content_tokens, 0), 0)) AS real_cached_content_tokens,
          SUM(IF(data_source = 'DEMO', COALESCE(estimated_cost_usd, 0), 0)) AS stored_demo_cost_usd,
    """
elif BQ_DATA_MODE == "demo":
    COST_AGGREGATES = """
          0 AS real_prompt_tokens,
          0 AS real_completion_tokens,
          0 AS real_thoughts_tokens,
          0 AS real_cached_content_tokens,
          SUM(COALESCE(estimated_cost_usd, 0)) AS stored_demo_cost_usd,
    """
elif BQ_COST_SOURCE == "stored":
    COST_AGGREGATES = """
          0 AS real_prompt_tokens,
          0 AS real_completion_tokens,
          0 AS real_thoughts_tokens,
          0 AS real_cached_content_tokens,
          SUM(COALESCE(estimated_cost_usd, 0)) AS stored_demo_cost_usd,
    """
else:
    COST_AGGREGATES = """
          SUM(COALESCE(prompt_tokens, 0)) AS real_prompt_tokens,
          SUM(COALESCE(completion_tokens, 0)) AS real_completion_tokens,
          SUM(COALESCE(thoughts_tokens, 0)) AS real_thoughts_tokens,
          SUM(COALESCE(cached_content_tokens, 0)) AS real_cached_content_tokens,
          0 AS stored_demo_cost_usd,
    """

if BQ_DATA_MODE == "combined":
    COST_DETAIL_FIELDS = "estimated_cost_usd, data_source"
elif BQ_DATA_MODE == "demo":
    COST_DETAIL_FIELDS = "estimated_cost_usd, 'DEMO' AS data_source"
elif BQ_COST_SOURCE == "stored":
    COST_DETAIL_FIELDS = "estimated_cost_usd, data_source"
else:
    COST_DETAIL_FIELDS = "0 AS estimated_cost_usd, 'REAL' AS data_source"


app = Flask(__name__)

@app.context_processor
def inject_firebase_config():
    return {
        "firebase_config": {
            "apiKey": os.getenv("FIREBASE_WEB_API_KEY", ""),
            "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
            "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
            "appId": os.getenv("FIREBASE_WEB_APP_ID", ""),
            "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
        }
    }


bq = bigquery.Client(project=PROJECT_ID)


# Firebase Authentication is configured above via environment.\n
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(options={"projectId": FIREBASE_PROJECT_ID})


@app.before_request
def require_firebase_auth_for_api():
    path = request.path

    if not path.startswith("/api/") or path == "/api/health":
        return None

    if path == "/api/admin-access/login":
        return None

    is_admin_access = path.startswith("/api/admin-access/")
    is_bind_pending = path == "/api/admin-access/bind-pending"

    if is_admin_access and session.get("local_bootstrap_admin") and not is_bind_pending:
        return None

    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        if is_admin_access and not is_bind_pending:
            return None
        return jsonify({
            "error": "authentication_required",
            "message": "Sign in to access the dashboard API."
        }), 401

    token = header[len("Bearer "):].strip()
    if not token:
        return jsonify({"error": "authentication_required"}), 401

    try:
        decoded = firebase_auth.verify_id_token(
            token, app=firebase_admin.get_app()
        )
        request.firebase_user = {
            "uid": decoded.get("uid"),
            "email": decoded.get("email"),
            "email_verified": decoded.get("email_verified", False),
        }
    except Exception:
        app.logger.warning("Rejected invalid Firebase ID token")
        return jsonify({
            "error": "invalid_authentication",
            "message": "Your session is invalid or expired. Please sign in again."
        }), 401

    if is_bind_pending:
        return None

    if is_admin_access:
        return None

    from admin_access import current_identity
    identity = current_identity()
    if not identity or identity.get("role") not in ("global_admin", "tenant_admin"):
        return jsonify({
            "error": "dashboard_role_required",
            "message": "Your verified account has no active dashboard administrator assignment."
        }), 403

    request.dashboard_identity = identity
    return None


# ============================================================
# Vertex AI token pricing - USD per 1M tokens
# ============================================================

VERTEX_TOKEN_PRICING = {
    "gemini-3.8-flash": {"input": 0.825, "cached_input": 0.0825, "output": 4.125},
    "gemini-3.8-flash-cyber": {"input": 1.65, "cached_input": 0.165, "output": 8.25},
    "gemini-3.7-flash": {"input": 0.825, "cached_input": 0.0825, "output": 4.125},
    "gemini-3.6-flash": {"input": 0.825, "cached_input": 0.0825, "output": 4.125},
    "gemini-3.5-flash": {"input": 0.825, "cached_input": 0.0825, "output": 4.95},
    "gemini-3.5-flash-lite": {"input": 0.33, "cached_input": 0.033, "output": 2.75},
    "gemini-3.1-flash-lite": {"input": 0.275, "cached_input": 0.0275, "output": 1.65},
    "gemini-3-flash-preview": {"input": 0.50, "cached_input": 0.05, "output": 1.50},
    "gemini-3.1-pro-preview": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gemini-2.5-pro": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "cached_input": 0.03, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "cached_input": 0.01, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.15, "cached_input": 0.015, "output": 0.60},
    "gemini-2.0-flash-lite": {"input": 0.075, "cached_input": 0.0075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "cached_input": 0.3125, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "cached_input": 0.01875, "output": 0.30},
}


def model_price(model):
    m = str(model or "").strip().lower()
    for name in sorted(VERTEX_TOKEN_PRICING, key=len, reverse=True):
        if name in m:
            return VERTEX_TOKEN_PRICING[name]
    return None


def estimate_cost_usd(row):
    pricing = model_price(row.get("model"))
    if not pricing:
        return 0.0

    prompt_tokens = int(row.get("prompt_tokens") or 0)
    cached_tokens = int(row.get("cached_content_tokens") or 0)
    completion_tokens = int(row.get("completion_tokens") or 0)
    thoughts_tokens = int(row.get("thoughts_tokens") or 0)

    cached_tokens = max(0, min(cached_tokens, prompt_tokens))
    normal_input_tokens = prompt_tokens - cached_tokens
    output_tokens = completion_tokens + thoughts_tokens

    cost = (
        normal_input_tokens * pricing["input"] / 1_000_000
        + cached_tokens * pricing["cached_input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )
    return round(cost, 8)


def aggregated_cost_usd(row):
    if "stored_demo_cost_usd" in row:
        real_usage = {
            "model": row.get("model"),
            "prompt_tokens": row.get("real_prompt_tokens") or 0,
            "completion_tokens": row.get("real_completion_tokens") or 0,
            "thoughts_tokens": row.get("real_thoughts_tokens") or 0,
            "cached_content_tokens": row.get("real_cached_content_tokens") or 0,
        }
        demo_cost = float(row.get("stored_demo_cost_usd") or 0)
        return round(demo_cost + estimate_cost_usd(real_usage), 8)

    source = str(row.get("data_source") or "").strip().upper()
    if (
        (source == "DEMO" or BQ_COST_SOURCE == "stored")
        and row.get("estimated_cost_usd") is not None
    ):
        return round(float(row.get("estimated_cost_usd") or 0), 8)

    return estimate_cost_usd(row)


def query(sql, params=None):
    identity = getattr(request, "dashboard_identity", None)
    query_params = list(params or [])

    if identity and identity.get("role") == "tenant_admin":
        raw_domain = (
            identity.get("tenant_domain")
            or identity.get("tenant_id")
            or ""
        )
        domain = str(raw_domain).strip().lower().lstrip("@").rstrip(".")
        if "@" in domain:
            domain = domain.rsplit("@", 1)[1]

        if not domain or "." not in domain or any(c.isspace() for c in domain):
            raise RuntimeError(
                "Tenant isolation failed closed: no valid assigned email domain"
            )

        project_ids = sorted(set(
            str(v).strip()
            for v in (identity.get("project_ids") or [])
            if str(v).strip()
        ))

        if not project_ids:
            raise RuntimeError(
                "Tenant isolation failed closed: no projects assigned"
            )

        table_name = f"`{TABLE}`"
        table_pattern = re.compile(
            r"\b(FROM|JOIN)\s+" + re.escape(table_name),
            re.IGNORECASE
        )

        project_predicate = (
            " AND project_id IN UNNEST(@__tenant_project_ids)"
            if project_ids else ""
        )

        tenant_predicate = (
            "COALESCE("
            "NULLIF(LOWER(TRIM(CAST(tenant_domain AS STRING))), ''), "
            "LOWER(REGEXP_EXTRACT("
            "CAST(user_principal AS STRING), r'@([^@]+)$'))"
            ") = @__tenant_domain"
        )

        scoped_source = (
            f"(SELECT * FROM {table_name} "
            "WHERE " + tenant_predicate
            + project_predicate
            + ")"
        )

        sql, replaced_count = table_pattern.subn(
            lambda match: f"{match.group(1)} {scoped_source}", sql
        )
        if replaced_count == 0:
            raise RuntimeError(
                "Tenant isolation failed closed: query does not reference "
                "the configured usage table"
            )

        query_params.append(
            bigquery.ScalarQueryParameter("__tenant_domain", "STRING", domain)
        )
        if project_ids:
            query_params.append(
                bigquery.ArrayQueryParameter(
                    "__tenant_project_ids", "STRING", project_ids
                )
            )

    config = bigquery.QueryJobConfig(query_parameters=query_params)
    return list(bq.query(sql, job_config=config).result())


def clean(row):
    result = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        elif hasattr(v, "isoformat") and not isinstance(v, (str, int, float, bool)):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result


SG_TZ = timezone(timedelta(hours=8))
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def today_sg():
    return datetime.now(SG_TZ).date()


def _norm_period(period):
    period = (period or "monthly").strip().lower()
    return {
        "today": "daily", "day": "daily", "date": "daily",
        "month": "monthly", "current-month": "monthly",
        "year": "yearly", "current-year": "yearly",
    }.get(period, period)


def _month_end(d):
    nxt = date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    return nxt - timedelta(days=1)


def selection(period=None, args=None):
    args = args if args is not None else request.args
    period = _norm_period(period if period is not None else args.get("period"))
    today = today_sg()
    floor = date.fromisoformat(DATA_START_DATE)

    def clamp(d):
        return min(max(d, floor), today)

    if period == "daily":
        raw = (args.get("date") or "").strip()
        try:
            d = clamp(date.fromisoformat(raw)) if raw else today
        except ValueError:
            d = today
        return {"period": "daily", "start": d, "end": d, "anchor": d,
                "custom": d != today}

    if period == "monthly":
        raw = (args.get("month") or "").strip()
        try:
            y, m = (int(x) for x in raw.split("-")[:2])
            d = clamp(date(y, m, 1))
        except (ValueError, TypeError):
            d = today
        first = date(d.year, d.month, 1)
        return {"period": "monthly", "start": first, "end": _month_end(first),
                "anchor": first,
                "custom": (first.year, first.month) != (today.year, today.month)}

    if period == "yearly":
        raw = (args.get("year") or "").strip()
        try:
            y = clamp(date(int(raw), 1, 1)).year
        except (ValueError, TypeError):
            y = today.year
        return {"period": "yearly", "start": date(y, 1, 1), "end": date(y, 12, 31),
                "anchor": date(y, 1, 1), "custom": y != today.year}

    return {"period": period, "start": floor, "end": today, "anchor": today,
            "custom": False}


def period_condition(period, column="event_timestamp", args=None):
    sel = selection(period, args=args)

    if sel["period"] in ("all", "all-time", "all_time"):
        return f"DATE({column}, 'Asia/Singapore') >= {DATA_START_SQL}"

    if sel["period"] in ("daily", "monthly", "yearly"):
        return (
            f"DATE({column}, 'Asia/Singapore') "
            f"BETWEEN DATE '{sel['start'].isoformat()}' "
            f"AND DATE '{sel['end'].isoformat()}'"
        )

    return f"""
        {column} >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
    """


def trend_window(period):
    sel = selection(period)
    today = today_sg()
    floor = date.fromisoformat(DATA_START_DATE)

    if sel["period"] == "monthly":
        y = sel["anchor"].year
        start = max(date(y, 1, 1), date(floor.year, floor.month, 1))
        last_month = min(date(y, 12, 1), date(today.year, today.month, 1))
        end = min(_month_end(last_month), today)
        bucket = "MONTH"
    elif sel["period"] == "yearly":
        start = date(floor.year, 1, 1)
        end = today
        bucket = "YEAR"
    else:
        end = sel["anchor"]
        start = max(floor, end - timedelta(days=29))
        bucket = "DAY"

    return f"DATE '{start.isoformat()}'", f"DATE '{end.isoformat()}'", bucket


@app.route("/")
def index():
    return render_template("dashboard.html", project=PROJECT_ID, dataset=DATASET)


# ============================================================
# Executive summary
# ============================================================

@app.route("/api/summary")
def summary():
    period = request.args.get("period", "monthly").strip().lower()
    condition = period_condition(period)

    rows = query(f"""
        SELECT
          COALESCE(NULLIF(TRIM(user_principal), ''), 'Unknown User') AS user_principal,
          COALESCE(NULLIF(TRIM(project_id), ''), 'Unknown Project') AS project_id,
          COALESCE(NULLIF(TRIM(model), ''), 'unknown') AS model,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
          COALESCE(SUM(thoughts_tokens), 0) AS thoughts_tokens,
          COALESCE(SUM(tool_use_prompt_tokens), 0) AS tool_use_prompt_tokens,
          COALESCE(SUM(cached_content_tokens), 0) AS cached_content_tokens,
          {COST_AGGREGATES}
          COALESCE(SUM(total_tokens), 0) AS total_tokens,
          {EVENT_COUNT_EXPR} AS telemetry_events,
          COUNTIF(REGEXP_CONTAINS(COALESCE(request_id, ''), r':input:')) AS input_events
        FROM `{TABLE}`
        WHERE {condition}
        GROUP BY user_principal, project_id, model
    """)

    result = {
        "period": period,
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thoughts_tokens": 0,
        "tool_use_prompt_tokens": 0,
        "cached_content_tokens": 0,
        "total_tokens": 0,
        "active_users": 0,
        "active_projects": 0,
        "active_models": 0,
        "estimated_cost": 0.0,
    }

    users = set()
    projects = set()
    models = set()

    for row in rows:
        item = clean(row)
        result["request_count"] += int(item.get("input_events") or 0) or int(item.get("telemetry_events") or 0)
        result["prompt_tokens"] += int(item.get("prompt_tokens") or 0)
        result["completion_tokens"] += int(item.get("completion_tokens") or 0)
        result["thoughts_tokens"] += int(item.get("thoughts_tokens") or 0)
        result["tool_use_prompt_tokens"] += int(item.get("tool_use_prompt_tokens") or 0)
        result["cached_content_tokens"] += int(item.get("cached_content_tokens") or 0)
        result["total_tokens"] += int(item.get("total_tokens") or 0)
        result["estimated_cost"] += aggregated_cost_usd(item)

        user = str(item.get("user_principal") or "").strip()
        project = str(item.get("project_id") or "").strip()
        model = str(item.get("model") or "").strip()

        if user and user != "Unknown User":
            users.add(user)
        if project and project != "Unknown Project":
            projects.add(project)
        if model and model.lower() != "unknown":
            models.add(model)

    result["active_users"] = len(users)
    result["active_projects"] = len(projects)
    result["active_models"] = len(models)
    result["estimated_cost"] = round(result["estimated_cost"], 4)

    return jsonify(result)


# ============================================================
# Projects / Models / Users
# ============================================================


@app.route("/api/reports")
def reports():
    period = request.args.get("period", "monthly").strip().lower()
    dimension = request.args.get("dimension", "user").strip().lower()

    if period not in {"daily", "monthly", "yearly", "custom"}:
        return jsonify({"error": "Unsupported reporting period"}), 400
    if dimension not in {"user", "project", "model", "token"}:
        return jsonify({"error": "unsupported report dimension"}), 400

    report_start = None
    report_end = None
    if period == "custom":
        try:
            report_start = date.fromisoformat(request.args.get("start_date", ""))
            report_end = date.fromisoformat(request.args.get("end_date", ""))
        except ValueError:
            return jsonify({"error": "Choose valid start and end dates."}), 400

        floor = date.fromisoformat(DATA_START_DATE)
        today = today_sg()
        if report_start > report_end:
            return jsonify({"error": "Start date must be on or before end date."}), 400
        if report_start < floor or report_end > today:
            return jsonify({
                "error": f"Dates must be between {floor.isoformat()} and {today.isoformat()}."
            }), 400

        condition = (
            "DATE(event_timestamp, 'Asia/Singapore') "
            f"BETWEEN DATE '{report_start.isoformat()}' "
            f"AND DATE '{report_end.isoformat()}'"
        )
    else:
        selected = selection(period, args=request.args)
        report_start, report_end = selected["start"], selected["end"]
        condition = period_condition(period, args=request.args)

    # Keep model-level rows so each model's estimated cost uses its own pricing.
    rows = query(f"""
        SELECT
          COALESCE(NULLIF(TRIM(user_principal), ''), 'Unknown User') AS user_principal,
          COALESCE(NULLIF(TRIM(project_id), ''), 'Unknown Project') AS project_id,
          COALESCE(NULLIF(TRIM(model), ''), 'unknown') AS model,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
          COALESCE(SUM(thoughts_tokens), 0) AS thoughts_tokens,
          COALESCE(SUM(tool_use_prompt_tokens), 0) AS tool_use_prompt_tokens,
          COALESCE(SUM(cached_content_tokens), 0) AS cached_content_tokens,
          {COST_AGGREGATES}
          COALESCE(SUM(total_tokens), 0) AS total_tokens,
          {EVENT_COUNT_EXPR} AS telemetry_events,
          COUNTIF(REGEXP_CONTAINS(COALESCE(request_id, ''), r':input:')) AS input_events
        FROM `{TABLE}`
        WHERE {condition}
        GROUP BY user_principal, project_id, model
    """)

    summary = {
        "estimated_cost_usd": 0.0, "total_tokens": 0,
        "prompt_tokens": 0, "cached_content_tokens": 0,
        "completion_tokens": 0, "thoughts_tokens": 0,
        "tool_use_prompt_tokens": 0, "requests": 0,
        "active_users": set(), "active_projects": set(), "active_models": set()
    }
    grouped = {}

    for raw in rows:
        item = clean(raw)
        user = str(item.get("user_principal") or "Unknown User")
        project = str(item.get("project_id") or "Unknown Project")
        model = str(item.get("model") or "unknown")
        cost = aggregated_cost_usd(item)
        requests_count = int(item.get("input_events") or 0) or int(item.get("telemetry_events") or 0)

        for key, value in (
            ("estimated_cost_usd", cost),
            ("total_tokens", int(item.get("total_tokens") or 0)),
            ("prompt_tokens", int(item.get("prompt_tokens") or 0)),
            ("cached_content_tokens", int(item.get("cached_content_tokens") or 0)),
            ("completion_tokens", int(item.get("completion_tokens") or 0)),
            ("thoughts_tokens", int(item.get("thoughts_tokens") or 0)),
            ("tool_use_prompt_tokens", int(item.get("tool_use_prompt_tokens") or 0)),
            ("requests", requests_count),
        ):
            summary[key] += value

        if user and not user.lower().startswith("unknown"):
            summary["active_users"].add(user)
        if project and not project.lower().startswith("unknown"):
            summary["active_projects"].add(project)
        if model and not model.lower().startswith("unknown"):
            summary["active_models"].add(model)

        if dimension == "token":
            categories = (
                ("Input tokens", int(item.get("prompt_tokens") or 0)),
                ("Cached input tokens", int(item.get("cached_content_tokens") or 0)),
                ("Output tokens", int(item.get("completion_tokens") or 0)),
                ("Thought tokens", int(item.get("thoughts_tokens") or 0)),
            )
            for name, count in categories:
                target = grouped.setdefault(name, {"name": name, "tokens": 0})
                target["tokens"] += count
            continue

        name = {"user": user, "project": project, "model": model}[dimension]
        target = grouped.setdefault(name, {
            "name": name, "estimated_cost_usd": 0.0, "tokens": 0,
            "requests": 0, "input_tokens": 0, "cached_tokens": 0,
            "output_tokens": 0, "thoughts_tokens": 0
        })
        target["estimated_cost_usd"] += cost
        target["tokens"] += int(item.get("total_tokens") or 0)
        target["requests"] += requests_count
        target["input_tokens"] += int(item.get("prompt_tokens") or 0)
        target["cached_tokens"] += int(item.get("cached_content_tokens") or 0)
        target["output_tokens"] += int(item.get("completion_tokens") or 0)
        target["thoughts_tokens"] += int(item.get("thoughts_tokens") or 0)

    summary["active_users"] = len(summary["active_users"])
    summary["active_projects"] = len(summary["active_projects"])
    summary["active_models"] = len(summary["active_models"])

    result_rows = list(grouped.values())
    result_rows.sort(
        key=lambda row: row.get("estimated_cost_usd", row.get("tokens", 0)),
        reverse=True
    )

    return jsonify({
        "period": period,
        "dimension": dimension,
        "start_date": report_start.isoformat() if report_start else None,
        "end_date": report_end.isoformat() if report_end else None,
        "period_label": (
            f"{report_start.isoformat()} to {report_end.isoformat()}"
            if report_start and report_end else period.title()
        ),
        "summary": summary,
        "rows": result_rows
    })


@app.route("/api/users")
def users():
    period = request.args.get("period", "daily").lower()
    condition = period_condition(period)

    rows = query(f"""
        SELECT
          COALESCE(NULLIF(TRIM(user_principal), ''), 'Unknown User') AS user_principal,
          COALESCE(NULLIF(TRIM(project_id), ''), 'Unknown Project') AS project_id,
          COALESCE(NULLIF(TRIM(model), ''), 'unknown') AS model,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
          COALESCE(SUM(thoughts_tokens), 0) AS thoughts_tokens,
          COALESCE(SUM(tool_use_prompt_tokens), 0) AS tool_use_prompt_tokens,
          COALESCE(SUM(cached_content_tokens), 0) AS cached_content_tokens,
          {COST_AGGREGATES}
          COALESCE(SUM(total_tokens), 0) AS total_tokens,
          {EVENT_COUNT_EXPR} AS telemetry_events,
          COUNTIF(REGEXP_CONTAINS(COALESCE(request_id, ''), r':input:')) AS input_events
        FROM `{TABLE}`
        WHERE {condition}
        GROUP BY user_principal, project_id, model
        ORDER BY total_tokens DESC
    """)

    output = []
    for row in rows:
        item = clean(row)
        item["request_count"] = int(item.get("input_events") or 0) or int(item.get("telemetry_events") or 0)
        item["estimated_cost_usd"] = aggregated_cost_usd(item)
        output.append(item)
    return jsonify(output)


@app.route("/api/projects")
def projects():
    period = request.args.get("period", "daily").lower()
    condition = period_condition(period)

    rows = query(f"""
        SELECT
          COALESCE(NULLIF(TRIM(project_id), ''), 'Unknown Project') AS project_id,
          COALESCE(NULLIF(TRIM(model), ''), 'unknown') AS model,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
          COALESCE(SUM(thoughts_tokens), 0) AS thoughts_tokens,
          COALESCE(SUM(cached_content_tokens), 0) AS cached_content_tokens,
          {COST_AGGREGATES}
          COALESCE(SUM(total_tokens), 0) AS total_tokens,
          {EVENT_COUNT_EXPR} AS telemetry_events,
          COUNTIF(
            REGEXP_CONTAINS(COALESCE(request_id, ''), r':input:')
          ) AS input_events,
          COUNT(DISTINCT NULLIF(TRIM(user_principal), '')) AS users
        FROM `{TABLE}`
        WHERE {condition}
        GROUP BY project_id, model
        ORDER BY total_tokens DESC
    """)

    grouped = {}
    for row in rows:
        item = clean(row)
        project = str(item.get("project_id") or "Unknown Project")
        if project not in grouped:
            grouped[project] = {
                "project_id": project,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "thoughts_tokens": 0,
                "cached_content_tokens": 0,
                "total_tokens": 0,
                "telemetry_events": 0,
                "input_events": 0,
                "users": set(),
                "models": set(),
                "estimated_cost": 0.0,
            }

        g = grouped[project]
        g["prompt_tokens"] += int(item.get("prompt_tokens") or 0)
        g["completion_tokens"] += int(item.get("completion_tokens") or 0)
        g["thoughts_tokens"] += int(item.get("thoughts_tokens") or 0)
        g["cached_content_tokens"] += int(item.get("cached_content_tokens") or 0)
        g["total_tokens"] += int(item.get("total_tokens") or 0)
        g["telemetry_events"] += int(item.get("telemetry_events") or 0)
        g["input_events"] += int(item.get("input_events") or 0)
        g["models"].add(str(item.get("model") or "unknown"))
        g["estimated_cost"] += aggregated_cost_usd(item)

    user_rows = query(f"""
        SELECT
          COALESCE(NULLIF(TRIM(project_id), ''), 'Unknown Project') AS project_id,
          COALESCE(NULLIF(TRIM(user_principal), ''), 'Unknown User') AS user_principal
        FROM `{TABLE}`
        WHERE {condition}
        GROUP BY project_id, user_principal
    """)
    for row in user_rows:
        item = clean(row)
        project = str(item.get("project_id") or "Unknown Project")
        if project in grouped:
            grouped[project]["users"].add(str(item.get("user_principal") or "Unknown User"))

    output = []
    for g in grouped.values():
        request_count = g["input_events"] or g["telemetry_events"]
        output.append({
            "project_id": g["project_id"],
            "prompt_tokens": g["prompt_tokens"],
            "completion_tokens": g["completion_tokens"],
            "thoughts_tokens": g["thoughts_tokens"],
            "cached_content_tokens": g["cached_content_tokens"],
            "total_tokens": g["total_tokens"],
            "request_count": request_count,
            "users": len(g["users"]),
            "models": len(g["models"]),
            "estimated_cost": round(g["estimated_cost"], 8),
        })

    output.sort(key=lambda x: x["total_tokens"], reverse=True)
    return jsonify(output)


@app.route("/api/models")
def models():
    period = request.args.get("period", "monthly").strip().lower()
    condition = period_condition(period)

    rows = query(f"""
        SELECT
          COALESCE(NULLIF(TRIM(model), ''), 'unknown') AS model,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
          COALESCE(SUM(thoughts_tokens), 0) AS thoughts_tokens,
          COALESCE(SUM(cached_content_tokens), 0) AS cached_content_tokens,
          {COST_AGGREGATES}
          COALESCE(SUM(total_tokens), 0) AS total_tokens,
          {EVENT_COUNT_EXPR} AS telemetry_events,
          COUNTIF(REGEXP_CONTAINS(COALESCE(request_id, ''), r':input:')) AS input_events,
          COUNT(DISTINCT NULLIF(TRIM(project_id), '')) AS projects,
          COUNT(DISTINCT NULLIF(TRIM(user_principal), '')) AS users
        FROM `{TABLE}`
        WHERE {condition}
        GROUP BY model
        ORDER BY total_tokens DESC
    """)

    output = []
    for row in rows:
        item = clean(row)
        item["request_count"] = int(item.get("input_events") or 0) or int(item.get("telemetry_events") or 0)
        item["estimated_cost"] = aggregated_cost_usd(item)
        output.append(item)
    return jsonify(output)


# ============================================================
# Cost trend
# ============================================================

@app.route("/api/cost-trend")
def cost_trend():
    period = request.args.get("period", "monthly").lower()
    if period == "yearly":
        start = f"DATE_TRUNC({DATA_START_SQL}, YEAR)"
        bucket = "DATE_TRUNC(DATE(event_timestamp, 'Asia/Singapore'), YEAR)"
    else:
        start = f"DATE_TRUNC({DATA_START_SQL}, MONTH)"
        bucket = "DATE_TRUNC(DATE(event_timestamp, 'Asia/Singapore'), MONTH)"

    rows = query(f"""
        SELECT
          {bucket} AS period_date,
          COALESCE(NULLIF(TRIM(model), ''), 'unknown') AS model,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
          COALESCE(SUM(thoughts_tokens), 0) AS thoughts_tokens,
          COALESCE(SUM(cached_content_tokens), 0) AS cached_content_tokens,
          {COST_AGGREGATES}
          COALESCE(SUM(total_tokens), 0) AS total_tokens,
          {EVENT_COUNT_EXPR} AS request_count
        FROM `{TABLE}`
        WHERE DATE(event_timestamp, 'Asia/Singapore') >= {start}
        GROUP BY period_date, model
        ORDER BY period_date
    """)

    grouped = {}
    for row in rows:
        item = clean(row)
        key = str(item["period_date"])
        if key not in grouped:
            grouped[key] = {
                "period_date": key,
                "total_tokens": 0,
                "request_count": 0,
                "estimated_cost": 0.0,
            }
        grouped[key]["total_tokens"] += int(item.get("total_tokens") or 0)
        grouped[key]["request_count"] += int(item.get("request_count") or 0)
        grouped[key]["estimated_cost"] += aggregated_cost_usd(item)

    return jsonify(list(grouped.values()))


# ============================================================
# Activity correlation
# ============================================================

def aggregate_activity_rows(events, limit=None):
    def event_direction(row):
        request_id = str(row.get("request_id") or "")
        match = re.search(r":(input|output):", request_id, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        prompt = int(row.get("prompt_tokens") or 0)
        completion = int(row.get("completion_tokens") or 0)
        if prompt > 0 and completion == 0:
            return "input"
        if completion > 0 and prompt == 0:
            return "output"
        return "unknown"

    def event_time(row):
        value = row.get("event_timestamp")
        if isinstance(value, datetime):
            return value
        if value is None:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    def grouping_key(row):
        return (
            str(row.get("user_principal") or "UNKNOWN"),
            str(row.get("project_id") or ""),
            str(row.get("model") or "UNKNOWN"),
            str(row.get("source") or "UNKNOWN"),
            str(row.get("authentication_type") or "UNKNOWN"),
        )

    input_events = []
    output_events = []
    standalone_events = []

    for index, row in enumerate(events):
        direction = event_direction(row)
        if direction == "input":
            input_events.append((index, row))
        elif direction == "output":
            output_events.append((index, row))
        else:
            standalone_events.append((index, row))

    candidates = []
    for input_index, input_row in input_events:
        input_time = event_time(input_row)
        if input_time is None:
            continue
        input_key = grouping_key(input_row)

        for output_index, output_row in output_events:
            output_time = event_time(output_row)
            if output_time is None or input_key != grouping_key(output_row):
                continue
            distance = abs((output_time - input_time).total_seconds())
            if distance <= 120:
                candidates.append((distance, input_index, output_index))

    candidates.sort(key=lambda item: item[0])
    matched_inputs = set()
    matched_outputs = set()
    pairs = []

    for distance, input_index, output_index in candidates:
        if input_index in matched_inputs or output_index in matched_outputs:
            continue
        matched_inputs.add(input_index)
        matched_outputs.add(output_index)
        pairs.append((input_index, output_index, distance))

    combined = []

    for input_index, output_index, distance in pairs:
        input_row = dict(events[input_index])
        output_row = dict(events[output_index])
        input_tokens = int(input_row.get("prompt_tokens") or 0)
        output_tokens = int(output_row.get("completion_tokens") or 0)
        input_time = event_time(input_row)
        output_time = event_time(output_row)
        activity_time = max(input_time, output_time) if input_time and output_time else (input_time or output_time)

        combined_row = dict(input_row)
        combined_row["event_timestamp"] = activity_time
        combined_row["prompt_tokens"] = input_tokens
        combined_row["completion_tokens"] = output_tokens
        combined_row["total_tokens"] = input_tokens + output_tokens
        if (
            BQ_COST_SOURCE == "stored"
            or str(combined_row.get("data_source") or "").upper() == "DEMO"
        ):
            combined_row["estimated_cost_usd"] = (
                float(input_row.get("estimated_cost_usd") or 0)
                + float(output_row.get("estimated_cost_usd") or 0)
            )

        for field in (
            "user_principal", "user_id", "project_id", "authentication_type",
            "location", "publisher", "model", "model_version", "source", "status"
        ):
            if not combined_row.get(field):
                combined_row[field] = output_row.get(field)

        combined_row["_activity_aggregation"] = "INPUT_OUTPUT_PAIRED"
        combined_row["_pair_distance_seconds"] = round(distance, 3)
        combined.append(combined_row)

    for index, row in input_events:
        if index not in matched_inputs:
            standalone = dict(row)
            standalone["_activity_aggregation"] = "INPUT_ONLY"
            combined.append(standalone)

    for index, row in output_events:
        if index not in matched_outputs:
            standalone = dict(row)
            standalone["_activity_aggregation"] = "OUTPUT_ONLY"
            combined.append(standalone)

    for _, row in standalone_events:
        standalone = dict(row)
        standalone["_activity_aggregation"] = "STANDALONE"
        combined.append(standalone)

    combined.sort(
        key=lambda row: event_time(row) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    for row in combined:
        use_stored = (
            BQ_COST_SOURCE == "stored"
            or str(row.get("data_source") or "").upper() == "DEMO"
        )
        if use_stored:
            row["estimated_cost_usd"] = float(row.get("estimated_cost_usd") or 0)
        else:
            row["estimated_cost_usd"] = estimate_cost_usd(row)

    return combined[:limit] if limit is not None else combined


@app.route("/api/requests")
def requests():
    activity_table = TABLE

    try:
        rows = query(f"""
            SELECT
              CASE
                WHEN STARTS_WITH(COALESCE(request_id, ''), 'ai-token:')
                  AND REGEXP_CONTAINS(
                    request_id,
                    r':[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z$'
                  )
                THEN TIMESTAMP(REGEXP_EXTRACT(
                  request_id,
                  r':([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z)$'
                ))
                ELSE event_timestamp
              END AS event_timestamp,
              request_id, user_principal, user_id, project_id,
              authentication_type, NULL AS publisher, model,
              NULL AS model_version,
              prompt_tokens, completion_tokens, thoughts_tokens,
              tool_use_prompt_tokens, cached_content_tokens, total_tokens,
              source, status,
              estimated_cost_usd, data_source
            FROM `{activity_table}`
            WHERE event_timestamp >= TIMESTAMP_SUB(
              CURRENT_TIMESTAMP(), INTERVAL 30 DAY
            )
            ORDER BY event_timestamp DESC, total_tokens DESC
            LIMIT 2000
        """)

        combined = aggregate_activity_rows(
            [dict(row) for row in rows], limit=50
        )
        return jsonify([clean(row) for row in combined])

    except Exception:
        app.logger.exception(
            "AI Activity query failed: mode=%s source=%s",
            BQ_DATA_MODE, activity_table
        )
        return jsonify({"error": "Unable to load AI Activity"}), 500


@app.route("/api/user-detail")
def user_detail():
    user = request.args.get("user", "").strip()
    period = request.args.get("period", "daily").strip().lower()
    start_date = request.args.get("start", "").strip()
    end_date = request.args.get("end", "").strip()
    project_filter = request.args.get("project", "").strip()

    if not user:
        return jsonify([])

    user_param = bigquery.ScalarQueryParameter("user", "STRING", user)
    params = [user_param]

    if project_filter and project_filter != "__ALL__":
        project_condition = """
          AND COALESCE(NULLIF(TRIM(project_id), ''), 'Unknown Project') = @project
        """
        params.append(bigquery.ScalarQueryParameter("project", "STRING", project_filter))
    else:
        project_condition = ""

    if period == "custom":
        try:
            start_obj = date.fromisoformat(start_date)
            end_obj = date.fromisoformat(end_date)
        except ValueError:
            return jsonify({"error": "Invalid custom date. Use YYYY-MM-DD."}), 400
        if end_obj < start_obj:
            return jsonify({"error": "End date cannot be before start date."}), 400
        params.extend([
            bigquery.ScalarQueryParameter("start_date", "DATE", start_obj),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_obj),
        ])
        date_condition = """
          AND DATE(event_timestamp, 'Asia/Singapore') BETWEEN @start_date AND @end_date
        """
    else:
        date_condition = "AND " + period_condition(period, "event_timestamp")

    rows = query(f"""
        SELECT
          CASE
            WHEN STARTS_WITH(request_id, 'ai-token:')
              AND REGEXP_CONTAINS(
                request_id,
                r':[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z$'
              )
            THEN TIMESTAMP(REGEXP_EXTRACT(
              request_id,
              r':([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z)$'
            ))
            ELSE event_timestamp
          END AS event_timestamp,
          request_id, user_principal, user_id, project_id,
          authentication_type, NULL AS location, NULL AS publisher, model, NULL AS model_version,
          prompt_tokens, completion_tokens, thoughts_tokens,
          tool_use_prompt_tokens, cached_content_tokens, total_tokens,
          source, status,
          {COST_DETAIL_FIELDS}
        FROM `{TABLE}`
        WHERE COALESCE(NULLIF(TRIM(user_principal), ''), '') = @user
          {date_condition}
          {project_condition}
        ORDER BY event_timestamp DESC
        LIMIT 5000
    """, params)

    combined = aggregate_activity_rows([dict(row) for row in rows], limit=5000)
    return jsonify([clean(row) for row in combined])


@app.route("/api/trend")
def trend():
    period = request.args.get("period", "daily").lower()
    start, end, bucket = trend_window(period)

    if bucket == "MONTH":
        bucket_expr = "DATE_TRUNC(DATE(event_timestamp, 'Asia/Singapore'), MONTH)"
        range_expr = f"GENERATE_DATE_ARRAY(DATE_TRUNC({start}, MONTH), DATE_TRUNC({end}, MONTH), INTERVAL 1 MONTH)"
        label_format = "%b"
    elif bucket == "YEAR":
        bucket_expr = "DATE_TRUNC(DATE(event_timestamp, 'Asia/Singapore'), YEAR)"
        range_expr = f"GENERATE_DATE_ARRAY(DATE_TRUNC({start}, YEAR), DATE_TRUNC({end}, YEAR), INTERVAL 1 YEAR)"
        label_format = "%Y"
    else:
        bucket_expr = "DATE(event_timestamp, 'Asia/Singapore')"
        range_expr = f"GENERATE_DATE_ARRAY({start}, {end}, INTERVAL 1 DAY)"
        label_format = "%d %b"

    rows = query(f"""
        WITH period_range AS (
          SELECT period_start
          FROM UNNEST({range_expr}) AS period_start
        ),
        usage AS (
          SELECT
            {bucket_expr} AS period_start,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            {EVENT_COUNT_EXPR} AS telemetry_events,
            COUNTIF(REGEXP_CONTAINS(COALESCE(request_id, ''), r':input:')) AS input_events
          FROM `{TABLE}`
          WHERE DATE(event_timestamp, 'Asia/Singapore') BETWEEN {start} AND {end}
          GROUP BY period_start
        )
        SELECT
          pr.period_start,
          COALESCE(u.total_tokens, 0) AS total_tokens,
          COALESCE(u.input_events, 0) AS input_events,
          COALESCE(u.telemetry_events, 0) AS telemetry_events
        FROM period_range pr
        LEFT JOIN usage u ON pr.period_start = u.period_start
        ORDER BY pr.period_start
    """)

    cost_rows = query(f"""
        SELECT
          {bucket_expr} AS period_start,
          COALESCE(NULLIF(TRIM(model), ''), 'unknown') AS model,
          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
          COALESCE(SUM(thoughts_tokens), 0) AS thoughts_tokens,
          COALESCE(SUM(cached_content_tokens), 0) AS cached_content_tokens,
          {COST_AGGREGATES}
          COALESCE(SUM(total_tokens), 0) AS total_tokens
        FROM `{TABLE}`
        WHERE DATE(event_timestamp, 'Asia/Singapore') BETWEEN {start} AND {end}
        GROUP BY period_start, model
    """)

    cost_by_period = {}
    for row in cost_rows:
        item = clean(row)
        key = str(item["period_start"])
        cost_by_period[key] = cost_by_period.get(key, 0.0) + aggregated_cost_usd(item)

    output = []
    for row in rows:
        item = clean(row)
        key = str(item["period_start"])
        total_tokens = int(item.get("total_tokens") or 0)
        request_count = int(item.get("input_events") or 0) or int(item.get("telemetry_events") or 0)
        item["date"] = key
        item["label"] = datetime.strptime(key, "%Y-%m-%d").strftime(label_format)
        item["full_label"] = datetime.strptime(key, "%Y-%m-%d").strftime(
            {"MONTH": "%B %Y", "YEAR": "%Y"}.get(bucket, "%d %b %Y")
        )
        item["request_count"] = request_count
        item["avg_tokens_per_request"] = round(total_tokens / request_count, 2) if request_count else 0
        item["estimated_cost"] = round(cost_by_period.get(key, 0.0), 8)
        output.append(item)

    return jsonify(output)


@app.errorhandler(Exception)
def handle_api_exception(exc):
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return exc
    if not request.path.startswith("/api/"):
        raise exc
    app.logger.exception("API error on %s", request.path)
    message = str(getattr(exc, "message", None) or exc)
    return jsonify({
        "error": "server_error",
        "message": message[:400],
    }), 500


@app.route("/api/diagnostics")
def diagnostics():
    from admin_access import current_identity
    identity = getattr(request, "dashboard_identity", None) or current_identity()
    if not identity or identity.get("role") != "global_admin":
        return jsonify({"error": "global_admin_required"}), 403

    required = [
        "event_timestamp", "request_id", "user_principal", "user_id",
        "tenant_domain", "project_id", "model", "authentication_type",
        "source", "status", "prompt_tokens", "completion_tokens",
        "thoughts_tokens", "tool_use_prompt_tokens", "cached_content_tokens",
        "total_tokens", "request_count", "estimated_cost_usd", "data_source",
    ]
    table = bq.get_table(TABLE)
    present = {f.name: f.field_type for f in table.schema}
    stats = list(bq.query(f"""
        SELECT
          MIN(DATE(event_timestamp, 'Asia/Singapore')) AS first_day,
          MAX(DATE(event_timestamp, 'Asia/Singapore')) AS last_day,
          COUNT(DISTINCT DATE(event_timestamp, 'Asia/Singapore')) AS days_with_data,
          COUNT(*) AS rows_total,
          SUM(estimated_cost_usd) AS total_cost_usd
        FROM `{TABLE}`
    """).result())
    return jsonify({
        "table": TABLE,
        "mode": BQ_DATA_MODE,
        "cost_source": BQ_COST_SOURCE,
        "data_start_date": DATA_START_DATE,
        "missing_columns": [c for c in required if c not in present],
        "event_timestamp_type": present.get("event_timestamp"),
        "coverage": clean(dict(stats[0])) if stats else {},
    })


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "project": PROJECT_ID,
        "dataset": DATASET,
        "table": TABLE,
        "mode": BQ_DATA_MODE,
        "cost_source": BQ_COST_SOURCE,
        "data_start_date": DATA_START_DATE,
        "timezone": "Asia/Singapore",
        "polling_seconds": 60,
    })


from admin_access import register_admin_access
register_admin_access(app)

from killswitch import register_killswitch
register_killswitch(
    app, bq,
    project_id=PROJECT_ID,
    finops_dataset=DATASET,
    ks_dataset=KS_DATASET,
    usage_table=TABLE,
)

from ai_advisor import register_advisor
register_advisor(
    app, query, period_condition,
    usage_table=TABLE,
    project_id=PROJECT_ID,
)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)


