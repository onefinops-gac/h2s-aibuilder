"""Kill-switch dashboard APIs.

Reads the killswitch_mdata dataset (projects, thresholds, budget_events,
killswitch_actions, notifications, audit_history) and combines it with LIVE
month-to-date spend from the usage table, so budgets are compared with the
same numbers the FinOps dashboard shows.

All endpoints are under /api/killswitch/*, so they are protected by the same
Firebase + role check as every other /api/* route (see app.py before_request).
Tenant admins only ever see their own tenant domain and assigned projects.
"""
from datetime import datetime, timezone
import time

from flask import Blueprint, jsonify, request
from google.cloud import bigquery

bp = Blueprint("killswitch_api", __name__)

_cfg = {}

# Short-lived cache for expensive project spend aggregation.
_PROJECT_ROWS_CACHE_SECONDS = 15
_project_rows_cache = {}

def _invalidate_project_rows_cache():
    _project_rows_cache.clear()

def _project_rows_cache_key(identity):
    role = identity.get("role")
    if role == "global_admin":
        return ("global_admin",)
    if role == "tenant_admin":
        domain = str(identity.get("tenant_domain") or identity.get("tenant_id") or "").strip().lower()
        projects = tuple(sorted(str(v).strip() for v in (identity.get("project_ids") or []) if str(v).strip()))
        return ("tenant_admin", domain, projects)
    return None


STATUS_ORDER = ["NORMAL", "WARNING", "HIGH_USAGE", "CRITICAL", "BUDGET_EXCEEDED"]


def register_killswitch(app, bq, project_id, finops_dataset, ks_dataset, usage_table):
    _cfg.update(
        bq=bq,
        project=project_id,
        finops=f"{project_id}.{finops_dataset}",
        ks=f"{project_id}.{ks_dataset}",
        usage=usage_table,
    )
    app.register_blueprint(bp)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _ks(table):
    return f"`{_cfg['ks']}.{table}`"


def _scope(alias=""):
    """Tenant isolation predicate + params. Fails closed for tenant admins."""
    identity = getattr(request, "dashboard_identity", None)
    if not identity:
        raise PermissionError("No dashboard identity")
    if identity.get("role") == "global_admin":
        return "TRUE", []

    if identity.get("role") != "tenant_admin":
        raise PermissionError("Role not allowed")

    raw = identity.get("tenant_domain") or identity.get("tenant_id") or ""
    domain = str(raw).strip().lower().lstrip("@").rstrip(".")
    if "@" in domain:
        domain = domain.rsplit("@", 1)[1]
    if not domain or "." not in domain or any(c.isspace() for c in domain):
        raise PermissionError("Tenant isolation failed closed: no tenant domain")

    projects = sorted({
        str(v).strip() for v in (identity.get("project_ids") or []) if str(v).strip()
    })
    if not projects:
        raise PermissionError("Tenant isolation failed closed: no projects assigned")

    a = f"{alias}." if alias else ""
    predicate = (
        f"LOWER({a}tenant_domain) = @ks_domain "
        f"AND {a}project_id IN UNNEST(@ks_projects)"
    )
    params = [
        bigquery.ScalarQueryParameter("ks_domain", "STRING", domain),
        bigquery.ArrayQueryParameter("ks_projects", "STRING", projects),
    ]
    return predicate, params


def _run(sql, params=None):
    cfg = bigquery.QueryJobConfig(query_parameters=list(params or []))
    return list(_cfg["bq"].query(sql, job_config=cfg).result())


def _clean(row):
    out = {}
    for k, v in dict(row).items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif v.__class__.__name__ == "Decimal":
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _limit():
    try:
        return max(1, min(int(request.args.get("limit", 200)), 1000))
    except ValueError:
        return 200


def _forbidden(exc):
    return jsonify({"error": "forbidden", "message": str(exc)}), 403


def _status(utilization_pct):
    if utilization_pct is None:
        return "NORMAL"
    if utilization_pct >= 100:
        return "BUDGET_EXCEEDED"
    if utilization_pct >= 90:
        return "CRITICAL"
    if utilization_pct >= 70:
        return "HIGH_USAGE"
    if utilization_pct >= 50:
        return "WARNING"
    return "NORMAL"


def _project_rows():
    """One row per project: budget, live MTD spend, projected spend, status."""
    # Cache by identity scope so tenant data can never cross tenant boundaries.
    identity = getattr(request, "dashboard_identity", None) or {}
    cache_key = _project_rows_cache_key(identity)
    now_mono = time.monotonic()
    cached = _project_rows_cache.get(cache_key) if cache_key is not None else None
    if cached and now_mono < cached["expires"]:
        return [dict(row) for row in cached["rows"]]

    pred, params = _scope("p")
    rows = _run(f"""
        WITH mtd AS (
          SELECT
            project_id,
            SUM(COALESCE(estimated_cost_usd, 0)) AS mtd_spend_usd,
            SUM(COALESCE(total_tokens, 0)) AS mtd_tokens,
            COUNT(DISTINCT user_id) AS active_users
          FROM `{_cfg['project']}.{_cfg['finops'].split('.', 1)[1]}.usage_daily`
          WHERE usage_date >= DATE_TRUNC(
                  CURRENT_DATE('Asia/Singapore'), MONTH
                )
            AND usage_date < DATE_ADD(
                  CURRENT_DATE('Asia/Singapore'), INTERVAL 1 DAY
                )
          GROUP BY project_id
        ),
        last_action AS (
          SELECT project_id, action_type, action_status, action_timestamp
          FROM {_ks('killswitch_actions')}
          WHERE TRUE
          QUALIFY ROW_NUMBER() OVER (
            PARTITION BY project_id
            ORDER BY action_timestamp DESC,
                     IF(action_type = 'RECOVER', 1, 0) DESC
          ) = 1
        ),
        thr AS (
          SELECT project_id,
                 COUNTIF(enabled) AS thresholds_enabled
          FROM {_ks('thresholds')}
          GROUP BY project_id
        ),
        proj AS (
          SELECT project_id, projected_monthly_spend_usd
          FROM `{_cfg['finops']}.project_monthly_summary`
          QUALIFY ROW_NUMBER() OVER (
            PARTITION BY project_id
            ORDER BY created_at DESC
          ) = 1
        )
        SELECT
          p.project_id,
          p.department_name,
          p.tenant_domain,
          p.region,
          CAST(p.monthly_budget_usd AS FLOAT64) AS monthly_budget_usd,
          COALESCE(m.mtd_spend_usd, 0) AS mtd_spend_usd,
          COALESCE(m.mtd_tokens, 0) AS mtd_tokens,
          COALESCE(m.active_users, 0) AS active_users,
          CAST(pr.projected_monthly_spend_usd AS FLOAT64) AS projected_spend_usd,
          COALESCE(t.thresholds_enabled, 0) AS thresholds_enabled,
          a.action_type AS last_action_type,
          a.action_status AS last_action_status,
          a.action_timestamp AS last_action_at
        FROM {_ks('projects')} p
        LEFT JOIN mtd m USING (project_id)
        LEFT JOIN proj pr USING (project_id)
        LEFT JOIN thr t USING (project_id)
        LEFT JOIN last_action a USING (project_id)
        WHERE {pred}
        ORDER BY p.project_id
    """, params)

    now = datetime.now(timezone.utc)
    out = []
    for r in rows:
        item = _clean(r)
        budget = float(item.get("monthly_budget_usd") or 0)
        mtd = float(item.get("mtd_spend_usd") or 0)
        projected = item.get("projected_spend_usd")
        projected = float(projected) if projected is not None else None
        item["mtd_spend_usd"] = round(mtd, 2)
        item["actual_utilization_pct"] = round(mtd / budget * 100, 2) if budget else 0
        item["projected_utilization_pct"] = (
            round(projected / budget * 100, 2) if (projected is not None and budget) else None
        )
        # Budget status follows actual month-to-date spend.
        # Keep projected utilization available for display, but do not
        # let an inflated projection alone mark a project as tripped.
        item["budget_status"] = _status(item["actual_utilization_pct"])
        item["actual_status"] = item["budget_status"]
        item["as_of"] = now.isoformat()
        out.append(item)

    if cache_key is not None:
        _project_rows_cache[cache_key] = {
            "rows": [dict(row) for row in out],
            "expires": time.monotonic() + _PROJECT_ROWS_CACHE_SECONDS,
        }
    return out


# ------------------------------------------------------------------
# endpoints
# ------------------------------------------------------------------
@bp.route("/api/killswitch/projects")
def ks_projects():
    try:
        return jsonify(_project_rows())
    except PermissionError as exc:
        return _forbidden(exc)


@bp.route("/api/killswitch/summary")
def ks_summary():
    try:
        projects = _project_rows()
        pred, params = _scope()

        def grouped(table, col, ts_col):
            rows = _run(f"""
                SELECT {col} AS k, COUNT(*) AS n, MAX({ts_col}) AS latest
                FROM {_ks(table)}
                WHERE {pred}
                GROUP BY k
            """, params)
            return {str(r["k"]): int(r["n"]) for r in rows}

        actions = grouped("killswitch_actions", "action_type", "action_timestamp")
        notifications = grouped("notifications", "severity", "notification_timestamp")
        events = grouped("budget_events", "event_type", "event_timestamp")
        audit_rows = _run(
            f"SELECT COUNT(*) AS n FROM {_ks('audit_history')} WHERE {pred}", params
        )
        audit_total = audit_rows[0]["n"] if audit_rows else 0

        status_counts = {s: 0 for s in STATUS_ORDER}
        for p in projects:
            status_counts[p["budget_status"]] = status_counts.get(p["budget_status"], 0) + 1

        total_budget = sum(p["monthly_budget_usd"] for p in projects)
        mtd = sum(p["mtd_spend_usd"] for p in projects)
        projected = sum(
            (p["projected_spend_usd"] if p["projected_spend_usd"] is not None else p["mtd_spend_usd"])
            for p in projects
        )

        return jsonify({
            "month": datetime.now(timezone.utc).strftime("%Y-%m"),
            "projects": len(projects),
            "total_budget_usd": round(total_budget, 2),
            "mtd_spend_usd": round(mtd, 2),
            "projected_spend_usd": round(projected, 2),
            "mtd_utilization_pct": round(mtd / total_budget * 100, 2) if total_budget else 0,
            "projected_utilization_pct": round(projected / total_budget * 100, 2) if total_budget else 0,
            "status_counts": status_counts,
            "events": sum(events.values()),
            "event_types": events,
            "actions": actions,
            "actions_total": sum(actions.values()),
            "simulated_blocks": actions.get("SIMULATE_AI_ACCESS_BLOCK", 0),
            "notifications": notifications,
            "notifications_total": sum(notifications.values()),
            "audit_records": int(audit_total),
        })
    except PermissionError as exc:
        return _forbidden(exc)


def _list_endpoint(table, ts_col):
    try:
        pred, params = _scope()
        rows = _run(f"""
            SELECT * FROM {_ks(table)}
            WHERE {pred}
            ORDER BY {ts_col} DESC
            LIMIT {_limit()}
        """, params)
        return jsonify([_clean(r) for r in rows])
    except PermissionError as exc:
        return _forbidden(exc)


@bp.route("/api/killswitch/events")
def ks_events():
    return _list_endpoint("budget_events", "event_timestamp")


@bp.route("/api/killswitch/actions")
def ks_actions():
    return _list_endpoint("killswitch_actions", "action_timestamp")


@bp.route("/api/killswitch/notifications")
def ks_notifications():
    return _list_endpoint("notifications", "notification_timestamp")


@bp.route("/api/killswitch/audit")
def ks_audit():
    return _list_endpoint("audit_history", "audit_timestamp")


@bp.route("/api/killswitch/monthly")
def ks_monthly():
    """Actual spend per month (from the usage table) vs the combined budget."""
    try:
        pred, params = _scope("p")
        rows = _run(f"""
          WITH scoped AS (
            SELECT p.project_id, CAST(p.monthly_budget_usd AS FLOAT64) AS budget
            FROM {_ks('projects')} p
            WHERE {pred}
          ),
          spend AS (
            SELECT
              DATE_TRUNC(DATE(u.event_timestamp, 'Asia/Singapore'), MONTH) AS month,
              SUM(COALESCE(u.estimated_cost_usd, 0)) AS spend_usd,
              COUNT(DISTINCT u.project_id) AS active_projects
            FROM `{_cfg['usage']}` u
            JOIN scoped s USING (project_id)
            GROUP BY month
          )
          SELECT
            month,
            spend_usd,
            active_projects,
            (SELECT SUM(budget) FROM scoped) AS budget_usd
          FROM spend
          ORDER BY month
        """, params)
        out = []
        for r in rows:
            item = _clean(r)
            budget = float(item.get("budget_usd") or 0)
            item["spend_usd"] = round(float(item.get("spend_usd") or 0), 2)
            item["utilization_pct"] = round(item["spend_usd"] / budget * 100, 2) if budget else 0
            out.append(item)
        return jsonify(out)
    except PermissionError as exc:
        return _forbidden(exc)


# ------------------------------------------------------------------
# /api/killswitch/status  +  simulate / recover  (used by the dashboard UI)
#
# SIMULATION ONLY: these endpoints read and write the killswitch_mdata
# tables. They never disable an API key, service or billing account.
# ------------------------------------------------------------------
def _is_global_admin():
    identity = getattr(request, "dashboard_identity", None) or {}
    return identity.get("role") == "global_admin"


def _actor():
    user = getattr(request, "firebase_user", None) or {}
    return user.get("email") or "unknown-admin"


def _threshold():
    pred, params = _scope()
    rows = _run(f"""
        SELECT MAX(threshold_pct) AS pct
        FROM {_ks('thresholds')}
        WHERE enabled AND {pred}
    """, params)
    pct = rows[0]["pct"] if rows and rows[0]["pct"] is not None else 90
    return float(pct) / 100.0


def _states():
    """project_id -> 'tripped' | 'armed' from the latest BLOCK / RECOVER action."""
    pred, params = _scope()
    rows = _run(f"""
        SELECT project_id, action_type
        FROM {_ks('killswitch_actions')}
        WHERE action_type IN ('SIMULATE_AI_ACCESS_BLOCK', 'RECOVER')
          AND {pred}
          AND NOT STARTS_WITH(
            COALESCE(action_reason, ''),
            'Synthetic demo action;'
          )
        QUALIFY ROW_NUMBER() OVER (
          PARTITION BY project_id
          ORDER BY action_timestamp DESC,
                   IF(action_type = 'RECOVER', 1, 0) DESC
        ) = 1
    """, params)
    return {
        r["project_id"]: ("tripped" if r["action_type"] == "SIMULATE_AI_ACCESS_BLOCK" else "armed")
        for r in rows
    }


def _events(limit=200):
    pred, params = _scope()
    rows = _run(f"""
        SELECT
          action_timestamp AS ts,
          project_id,
          CASE action_type
            WHEN 'SEND_WARNING' THEN 'warning'
            WHEN 'NOTIFY_ADMIN' THEN 'notified'
            WHEN 'SIMULATE_AI_ACCESS_BLOCK' THEN 'tripped'
            WHEN 'RECOVER' THEN 'recovered'
            WHEN 'DUPLICATE_IGNORED' THEN 'ignored'
            ELSE LOWER(action_type)
          END AS type,
          CASE
            WHEN STARTS_WITH(COALESCE(action_reason, ''), 'Manual') THEN action_reason
            WHEN action_type = 'SEND_WARNING'
              THEN CONCAT('Crossed ', CAST(threshold_pct AS STRING), '% of budget')
            WHEN action_type = 'NOTIFY_ADMIN'
              THEN CONCAT('Budget owners notified (', CAST(threshold_pct AS STRING), '% threshold)')
            WHEN action_type = 'SIMULATE_AI_ACCESS_BLOCK'
              THEN CONCAT('Reached ', CAST(threshold_pct AS STRING), '% threshold. AI access block simulated')
            ELSE COALESCE(action_reason, action_type)
          END AS detail,
          IF(STARTS_WITH(COALESCE(action_reason, ''), 'Manual'),
             COALESCE(REGEXP_EXTRACT(action_reason, r' by (.+)$'), 'admin'),
             'system') AS actor,
          action_type IN ('SIMULATE_AI_ACCESS_BLOCK', 'RECOVER') AS simulated
        FROM {_ks('killswitch_actions')}
        WHERE {pred}
        ORDER BY ts DESC
        LIMIT {int(limit)}
    """, params)
    return [_clean(r) for r in rows]


@bp.route("/api/killswitch/status")
def ks_status():
    try:
        threshold = _threshold()
        states = _states()
        projects = []

        for p in _project_rows():
            mtd = float(p.get("mtd_spend_usd") or 0)
            projected = p.get("projected_spend_usd")
            projected = float(projected) if projected is not None else None

            projects.append({
                "project_id": p["project_id"],
                "department_name": p.get("department_name"),
                "budget_usd": p["monthly_budget_usd"],
                "spend_usd": round(mtd, 2),
                "mtd_spend_usd": round(mtd, 2),
                "projected_spend_usd": (
                    round(projected, 2) if projected is not None else None
                ),
                "actual_utilization_pct": p.get("actual_utilization_pct"),
                "projected_utilization_pct": p.get("projected_utilization_pct"),
                "budget_status": p.get("budget_status", "NORMAL"),
                "state": states.get(p["project_id"], "armed"),
            })

        return jsonify({
            "mode": "simulation",
            "threshold": threshold,
            "basis": "actual_mtd",
            "can_admin": _is_global_admin(),
            "projects": projects,
            "events": _events(),
        })
    except PermissionError as exc:
        return _forbidden(exc)


def _project_in_scope(project_id):
    pred, params = _scope()
    rows = _run(f"""
        SELECT tenant_domain FROM {_ks('projects')}
        WHERE project_id = @pid AND {pred}
        LIMIT 1
    """, params + [bigquery.ScalarQueryParameter("pid", "STRING", project_id)])
    return rows[0]["tenant_domain"] if rows else None


def _record(project_id, tenant, action_type, status, reason, pct, level):
    _run(f"""
        INSERT INTO {_ks('killswitch_actions')}
          (action_id, event_id, project_id, tenant_domain, action_timestamp,
           threshold_pct, threshold_level, action_type, action_status,
           action_reason, data_source)
        VALUES (GENERATE_UUID(), NULL, @pid, @tenant, CURRENT_TIMESTAMP(),
                @pct, @level, @atype, @astatus, @reason, 'DEMO')
    """, [
        bigquery.ScalarQueryParameter("pid", "STRING", project_id),
        bigquery.ScalarQueryParameter("tenant", "STRING", tenant),
        bigquery.ScalarQueryParameter("pct", "INT64", pct),
        bigquery.ScalarQueryParameter("level", "STRING", level),
        bigquery.ScalarQueryParameter("atype", "STRING", action_type),
        bigquery.ScalarQueryParameter("astatus", "STRING", status),
        bigquery.ScalarQueryParameter("reason", "STRING", reason),
    ])
    _run(f"""
        INSERT INTO {_ks('audit_history')}
          (audit_id, project_id, tenant_domain, audit_timestamp,
           audit_event, audit_details, actor, data_source)
        VALUES (GENERATE_UUID(), @pid, @tenant, CURRENT_TIMESTAMP(),
                @atype, @reason, @actor, 'DEMO')
    """, [
        bigquery.ScalarQueryParameter("pid", "STRING", project_id),
        bigquery.ScalarQueryParameter("tenant", "STRING", tenant),
        bigquery.ScalarQueryParameter("atype", "STRING", action_type),
        bigquery.ScalarQueryParameter("reason", "STRING", reason),
        bigquery.ScalarQueryParameter("actor", "STRING", _actor()),
    ])


def _action_endpoint(action):
    try:
        body = request.get_json(silent=True) or {}
        project_id = str(body.get("project_id") or "").strip()
        if not project_id:
            return jsonify({"error": "project_id is required"}), 400

        if action == "recover" and not _is_global_admin():
            return jsonify({"error": "Only a Global Admin can recover a project"}), 403

        tenant = _project_in_scope(project_id)
        if tenant is None:
            return jsonify({"error": "Unknown project or not in your scope"}), 404

        tripped = _states().get(project_id) == "tripped"
        pct = int(round(_threshold() * 100))
        who = _actor()

        if action == "simulate":
            if tripped:
                _record(project_id, tenant, "DUPLICATE_IGNORED", "IGNORED",
                        f"Manual duplicate trigger ignored (idempotent) by {who}", pct, "CRITICAL")
                return jsonify({"message": "Already tripped. Duplicate trigger ignored."})
            _record(project_id, tenant, "SIMULATE_AI_ACCESS_BLOCK", "SIMULATED_BLOCK_ONLY",
                    f"Manual simulated trip by {who}. Real billing untouched.", pct, "CRITICAL")
            return jsonify({"message": f"Simulated trip recorded for {project_id}. Real billing untouched."})

        if not tripped:
            return jsonify({"message": "Project is already armed. Nothing to recover."})
        _record(project_id, tenant, "RECOVER", "SIMULATED_RECOVERY",
                f"Manual recovery by {who} (simulated)", pct, "CRITICAL")
        return jsonify({"message": f"{project_id} recovered and re-armed (simulated)."})
    except PermissionError as exc:
        return _forbidden(exc)


@bp.route("/api/killswitch/simulate", methods=["POST"])
def ks_simulate():
    _invalidate_project_rows_cache()
    return _action_endpoint("simulate")


@bp.route("/api/killswitch/recover", methods=["POST"])
def ks_recover():
    _invalidate_project_rows_cache()
    return _action_endpoint("recover")

