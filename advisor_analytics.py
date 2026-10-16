"""Deterministic analytics behind the AI FinOps Advisor.

Everything numeric the advisor says (forecast tokens, risk scores, savings) is computed HERE, in
plain Python, from BigQuery results. Gemini only chooses what to highlight and explains it; it never
produces a number. That keeps forecasts auditable and stops the model inventing figures.

Pure functions (fit_forecast, score_risk, build_tips, lifecycle_for) have no I/O and are unit-tested in
tests/test_advisor.py. The build_* functions run SQL through the dashboard's tenant-isolated query()
and therefore always reference the usage table as  FROM `{T}`  (query() rewrites exactly that form).
"""
import calendar
import math
from datetime import date, timedelta

# ----------------------------------------------------------------------------
# Tunable assumptions (all shown to the admin, none hidden inside prompts)
# ----------------------------------------------------------------------------
ASSUMPTIONS = {
    "downshift_share": 0.40,        # share of routine Pro traffic assumed movable to Flash
    "thinking_cut": 0.30,           # saving from capping reasoning/"thinking" tokens
    "cache_saving": 0.08,           # saving from raising context-cache hit rate
    "verbose_output_saving": 0.10,  # saving from constraining output length
    "big_context_saving": 0.10,     # saving from trimming context / retrieval
    "growth_floor_monthly": -0.10,  # forecast growth is clamped to [-10%, +15%] per month ...
    "growth_cap_monthly": 0.15,
    "growth_damping": 0.97,         # ... and fades 3% per month so a hot trend is not extrapolated forever
    "band_z": 1.28,                 # 80% prediction interval
}

# Model lifecycle: (id prefix, retirement date, firm?)  Longest matching prefix wins.
# Source: Google Cloud "Vertex AI model versions" page, checked 2026-10-14. firm=False means Google
# publishes "not before <date>". UPDATE THIS TABLE when Google announces changes.
MODEL_LIFECYCLE_SOURCE = "Google Cloud Vertex AI model-versions page, checked 2026-10-14"
MODEL_LIFECYCLE = [
    ("gemini-1.0", "2025-04-21", True),
    ("gemini-1.5", "2025-09-24", True),
    ("gemini-2.0-flash-lite", "2026-06-01", True),
    ("gemini-2.0-flash", "2026-06-01", True),
    ("gemini-2.5-flash-image", "2026-10-02", True),
    ("gemini-2.5-flash-lite", "2026-10-16", False),
    ("gemini-2.5-flash", "2026-10-16", False),
    ("gemini-2.5-pro", "2026-10-16", False),
]


def lifecycle_for(model, today):
    """Return {"status","date","note"} for a model id, or None when it has no known retirement."""
    m = str(model or "").lower()
    best = None
    for prefix, iso, firm in MODEL_LIFECYCLE:
        if m.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, iso, firm)
    if not best:
        return None
    _, iso, firm = best
    d = date.fromisoformat(iso)
    if d <= today and firm:
        return {"status": "retired", "date": iso, "note": f"retired on {iso}; requests typically fail with 404"}
    if d <= today:
        return {"status": "retirement_window_open", "date": iso,
                "note": f"Google lists retirement as not before {iso}; it can be switched off at any time now"}
    days = (d - today).days
    return {"status": "retiring_soon" if days <= 120 else "scheduled", "date": iso,
            "note": f"{'retires' if firm else 'retirement not before'} {iso}, {days} days away"}


# ----------------------------------------------------------------------------
# Forecast (pure)
# ----------------------------------------------------------------------------
def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return my - b * mx, b


def _next_month(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    m += 1
    if m > 12:
        y, m = y + 1, 1
    return f"{y:04d}-{m:02d}"


def fit_forecast(months, values, horizon=12):
    """Forecast monthly totals.

    months: ascending 'YYYY-MM' strings of COMPLETE months; values: totals (e.g. tokens).
    Method: log-linear trend (exponential growth) with damped growth, optional seasonality when
    >= 24 months exist (two observations per calendar month, shrunk 50% toward zero), and an 80%
    band that widens with the horizon. Returns a dict with available=False when history is too short.
    """
    months, values = list(months), [float(v or 0) for v in values]
    while values and values[0] <= 0:                       # drop leading empty months
        months, values = months[1:], values[1:]
    n = len(values)
    if n < 4:
        return {"available": False, "history_months": n,
                "reason": f"Need at least 4 complete months of history to forecast (have {n})."}
    ys = [math.log(max(v, 1.0)) for v in values]
    xs = list(range(n))
    a, b = _ols(xs, ys)
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]

    seasonal, use_season = {}, False
    if n >= 24:
        by_m = {}
        for mo, r in zip(months, resid):
            by_m.setdefault(int(mo[5:7]), []).append(r)
        if len(by_m) == 12 and all(len(v) >= 2 for v in by_m.values()):
            use_season = True
            seasonal = {m: max(-0.35, min(0.35, 0.5 * sum(v) / len(v))) for m, v in by_m.items()}
    adj = [r - seasonal.get(int(mo[5:7]), 0.0) for mo, r in zip(months, resid)]
    sigma = max(math.sqrt(sum(r * r for r in adj) / max(n - 2, 1)), 0.05)

    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in resid) / ss_tot if ss_tot else 0.0

    g_raw = math.exp(b) - 1
    g = max(ASSUMPTIONS["growth_floor_monthly"], min(ASSUMPTIONS["growth_cap_monthly"], g_raw))
    capped = abs(g - g_raw) > 1e-9
    bc = math.log(1 + g)

    mu, ym = a + b * (n - 1), months[-1]
    z, damp, out = ASSUMPTIONS["band_z"], ASSUMPTIONS["growth_damping"], []
    for h in range(1, horizon + 1):
        mu += bc * (damp ** h)
        ym = _next_month(ym)
        s = seasonal.get(int(ym[5:7]), 0.0)
        sh = sigma * (1 + 0.06 * h)
        out.append({"month": ym, "tokens": int(round(math.exp(mu + s))),
                    "low": int(round(math.exp(mu + s - z * sh))), "high": int(round(math.exp(mu + s + z * sh)))})

    total = sum(m["tokens"] for m in out)
    last12 = sum(values[-12:]) if n >= 12 else None
    confidence = "high" if (n >= 18 and r2 >= 0.7) else "medium" if (n >= 9 and r2 >= 0.4) else "low"
    return {
        "available": True, "history_months": n, "first_month": months[0], "last_month": months[-1],
        "method": "log-linear trend, damped growth" + (", seasonality" if use_season else ""),
        "seasonality_used": use_season, "monthly_growth_pct": round(g * 100, 1), "growth_capped": capped,
        "r2": round(r2, 2), "confidence": confidence, "interval": "80%", "months": out,
        "next12_tokens": total, "next12_low": sum(m["low"] for m in out), "next12_high": sum(m["high"] for m in out),
        "last12_tokens": int(last12) if last12 is not None else None,
        "change_vs_last12_pct": round(100 * (total / last12 - 1), 1) if last12 else None,
    }


def _dim(ym):
    return calendar.monthrange(int(ym[:4]), int(ym[5:7]))[1]


def build_forecast_12m(q, T, today, monthly_budget_total=None):
    """Whole-history forecast for the next 12 months (tokens, then cost via the recent blended rate)."""
    rows = q(f"""
        SELECT FORMAT_DATE('%Y-%m', DATE_TRUNC(DATE(event_timestamp,'Asia/Singapore'), MONTH)) ym,
               SUM(total_tokens) tokens, SUM(estimated_cost_usd) cost, SUM(COALESCE(request_count,1)) requests
        FROM `{T}`
        WHERE DATE(event_timestamp,'Asia/Singapore') < DATE_TRUNC(CURRENT_DATE('Asia/Singapore'), MONTH)
        GROUP BY ym ORDER BY ym""")
    months = [r["ym"] for r in rows]
    tokens = [float(r["tokens"] or 0) for r in rows]
    fc = fit_forecast(months, tokens)
    if not fc["available"]:
        return fc
    cost3 = sum(float(r["cost"] or 0) for r in rows[-3:])
    tok3 = sum(float(r["tokens"] or 0) for r in rows[-3:])
    rate = cost3 / tok3 if tok3 else 0.0                      # USD per token, last 3 complete months
    for m in fc["months"]:
        m["cost_usd"] = round(m["tokens"] * rate, 2)
    fc["blended_usd_per_1k_tokens"] = round(rate * 1000, 4)
    fc["next12_cost_usd"] = round(fc["next12_tokens"] * rate, 2)
    fc["next12_cost_low_usd"] = round(fc["next12_low"] * rate, 2)
    fc["next12_cost_high_usd"] = round(fc["next12_high"] * rate, 2)
    fc["cost_assumption"] = "Cost = forecast tokens x blended $/token of the last 3 complete months (current model mix and prices)."
    if monthly_budget_total and monthly_budget_total > 0:
        first_over = next((m["month"] for m in fc["months"] if m["cost_usd"] > monthly_budget_total), None)
        fc["budget"] = {"monthly_budget_usd": round(monthly_budget_total, 2),
                        "annual_budget_usd": round(monthly_budget_total * 12, 2),
                        "forecast_pct_of_annual_budget": round(100 * fc["next12_cost_usd"] / (monthly_budget_total * 12), 1),
                        "first_month_over_monthly_budget": first_over,
                        "note": "Compared with today's total monthly budgets held flat for 12 months."}
    fc["top_projects"] = _project_forecasts(q, T)
    return fc


def _project_forecasts(q, T, top_n=5):
    try:
        rows = q(f"""
            SELECT project_id, FORMAT_DATE('%Y-%m', DATE_TRUNC(DATE(event_timestamp,'Asia/Singapore'), MONTH)) ym,
                   SUM(total_tokens) tokens, SUM(estimated_cost_usd) cost
            FROM `{T}`
            WHERE DATE(event_timestamp,'Asia/Singapore') >= DATE_SUB(DATE_TRUNC(CURRENT_DATE('Asia/Singapore'), MONTH), INTERVAL 24 MONTH)
              AND DATE(event_timestamp,'Asia/Singapore') <  DATE_TRUNC(CURRENT_DATE('Asia/Singapore'), MONTH)
            GROUP BY project_id, ym""")
    except Exception:
        return []
    by_p = {}
    for r in rows:
        by_p.setdefault(r["project_id"], {})[r["ym"]] = (float(r["tokens"] or 0), float(r["cost"] or 0))

    def recent_cost(series):
        return sum(c for _, (_, c) in sorted(series.items())[-3:])

    out = []
    for pid in sorted(by_p, key=lambda p: -recent_cost(by_p[p]))[:top_n]:
        s = sorted(by_p[pid].items())
        months, toks = [m for m, _ in s], [v[0] for _, v in s]
        tok3, cost3 = sum(toks[-3:]), sum(v[1] for _, v in s[-3:])
        rate = cost3 / tok3 if tok3 else 0.0
        if len(s) >= 6:
            f = fit_forecast(months, toks)
            if f["available"]:
                out.append({"project": pid, "next12_tokens": f["next12_tokens"], "next12_cost_usd": round(f["next12_tokens"] * rate, 2),
                            "monthly_growth_pct": f["monthly_growth_pct"], "confidence": f["confidence"], "method": "trend"})
                continue
        run = tok3 / max(len(toks[-3:]), 1) * 12
        out.append({"project": pid, "next12_tokens": int(run), "next12_cost_usd": round(run * rate, 2),
                    "monthly_growth_pct": None, "confidence": "low", "method": "run-rate (short history)"})
    return out


# ----------------------------------------------------------------------------
# Overuse risk (pure scoring + data assembly)
# ----------------------------------------------------------------------------
def _level(score):
    return "critical" if score >= 70 else "high" if score >= 45 else "medium" if score >= 25 else "low"


def score_risk(budget, mtd, live_projected, accel, anomaly_multiple, trip_threshold, run_rate_daily, day, dim):
    """Explainable 0-100 overuse risk for one project. Returns (score, drivers, trip_in_days)."""
    drivers, score = [], 0
    if budget <= 0:
        return 0, ["no budget set"], None
    proj_pct = 100 * live_projected / budget
    used_pct = 100 * mtd / budget
    if used_pct >= 100:
        score += 55
        drivers.append(f"already at {used_pct:.0f}% of budget")
    elif used_pct >= trip_threshold * 100:
        score += 45
        drivers.append(f"{used_pct:.0f}% of budget used (KillSwitch threshold {trip_threshold * 100:.0f}%)")
    elif proj_pct >= 130:
        score += 50
        drivers.append(f"projected {proj_pct:.0f}% of budget by month end")
    elif proj_pct >= 100:
        score += 40
        drivers.append(f"projected {proj_pct:.0f}% of budget by month end")
    elif proj_pct >= 85:
        score += 28
        drivers.append(f"projected {proj_pct:.0f}% of budget by month end")
    elif proj_pct >= 70:
        score += 15
        drivers.append(f"projected {proj_pct:.0f}% of budget by month end")
    if accel is not None and accel >= 1.5:
        score += 15
        drivers.append(f"daily spend is {accel:.1f}x the prior week")
    elif accel is not None and accel >= 1.2:
        score += 8
        drivers.append(f"daily spend is {accel:.1f}x the prior week")
    if anomaly_multiple:
        score += 15
        drivers.append(f"recent spike {anomaly_multiple:.1f}x baseline")
    trip_days = None
    if run_rate_daily > 0 and mtd < budget * trip_threshold:
        trip_days = (budget * trip_threshold - mtd) / run_rate_daily
        if trip_days <= dim - day:
            score += 10 if trip_days <= 7 else 5
            drivers.append(f"reaches the KillSwitch threshold in about {math.ceil(trip_days)} day(s) at the current rate")
        else:
            trip_days = None
    elif mtd >= budget * trip_threshold:
        trip_days = 0
    return min(100, score), drivers, trip_days


def build_risk(project_rows, daily_rows, anomalies, trip_threshold, states, today):
    """Rank projects by overuse risk using real budgets (KillSwitch tables) and the last 17 days of spend."""
    day, dim = max(today.day, 1), calendar.monthrange(today.year, today.month)[1]
    left = dim - today.day
    series = {}
    for r in daily_rows:
        series.setdefault(r["project_id"], {})[r["d"]] = float(r["cost"] or 0)
    an = {a["project"]: a["multiple"] for a in anomalies}
    items = []
    for r in project_rows:
        b = float(r.get("monthly_budget_usd") or 0)
        if b <= 0:
            continue
        pid, mtd = r["project_id"], float(r.get("mtd_spend_usd") or 0)
        s = series.get(pid, {})
        last7 = [s.get(today - timedelta(days=i), 0.0) for i in range(1, 8)]
        prev7 = [s.get(today - timedelta(days=i), 0.0) for i in range(8, 15)]
        rr = sum(last7) / 7
        accel = (rr / (sum(prev7) / 7)) if sum(prev7) > 0 else None
        live_proj = mtd + rr * left if s else (float(r["projected_spend_usd"]) if r.get("projected_spend_usd") is not None else mtd / day * dim)
        score, drivers, trip_days = score_risk(b, mtd, live_proj, accel, an.get(pid), trip_threshold, rr, day, dim)
        mtd_tokens = float(r.get("mtd_tokens") or 0)
        items.append({
            "project": pid, "risk_score": score, "risk_level": _level(score), "drivers": drivers,
            "budget_usd": round(b, 2), "mtd_spend_usd": round(mtd, 2), "budget_used_pct": round(100 * mtd / b, 1),
            "projected_month_end_usd": round(live_proj, 2), "projected_pct_of_budget": round(100 * live_proj / b, 1),
            "projected_month_end_tokens": int(mtd_tokens / day * dim),
            "killswitch_state": states.get(pid, "armed"),
            "reaches_killswitch_threshold_on": (today + timedelta(days=max(1, math.ceil(trip_days)))).isoformat()
            if trip_days not in (None, 0) else ("already reached" if trip_days == 0 else None),
        })
    items.sort(key=lambda x: (-x["risk_score"], -x["projected_pct_of_budget"]))
    counts = {}
    for i in items:
        counts[i["risk_level"]] = counts.get(i["risk_level"], 0) + 1
    return {"basis": "live run-rate of the last 7 days vs real project budgets; KillSwitch is simulation-only",
            "killswitch_threshold_pct": round(trip_threshold * 100), "level_counts": counts,
            "top": items[:8], "scoring": "0-100: projected budget use, KillSwitch threshold proximity, acceleration, spikes"}


# ----------------------------------------------------------------------------
# Per-project / per-user profiles and tips
# ----------------------------------------------------------------------------
def _agg(rows):
    cost = sum(float(r["cost"] or 0) for r in rows)
    tokens = sum(float(r["tokens"] or 0) for r in rows)
    req = sum(float(r["req"] or 0) for r in rows)
    prompt = sum(float(r["prompt"] or 0) for r in rows)
    comp = sum(float(r["completion"] or 0) for r in rows)
    think = sum(float(r["thoughts"] or 0) for r in rows)
    cached = sum(float(r["cached"] or 0) for r in rows)
    by_model = {}
    for r in rows:
        d = by_model.setdefault(r["model"], {"cost": 0.0, "tokens": 0.0})
        d["cost"] += float(r["cost"] or 0)
        d["tokens"] += float(r["tokens"] or 0)
    mix = sorted(by_model.items(), key=lambda kv: -kv[1]["cost"])
    return {"cost": cost, "tokens": tokens, "req": req, "prompt": prompt, "completion": comp,
            "thoughts": think, "cached": cached, "by_model": by_model,
            "mix": [{"model": m, "share_pct": round(100 * d["cost"] / cost, 1) if cost else 0.0} for m, d in mix[:3]]}


def _metrics(a):
    tpr = a["tokens"] / a["req"] if a["req"] else 0
    return {"cost_usd": round(a["cost"], 2), "tokens": int(a["tokens"]), "requests": int(a["req"]),
            "tokens_per_request": int(tpr),
            "usd_per_1k_tokens": round(1000 * a["cost"] / a["tokens"], 4) if a["tokens"] else 0,
            "model_mix": a["mix"],
            "cached_input_pct": round(100 * a["cached"] / a["prompt"], 1) if a["prompt"] else 0.0,
            "thinking_token_pct": round(100 * a["thoughts"] / a["tokens"], 1) if a["tokens"] else 0.0,
            "output_to_input_ratio": round(a["completion"] / a["prompt"], 2) if a["prompt"] else 0.0}


def build_tips(scope, entity, a, mf, flash_cpt, total_cost):
    """Deterministic, evidence-backed tips for one project or user. `mf` scales period cost to 30 days."""
    tips, cost = [], a["cost"]
    if cost <= 0 or a["tokens"] <= 0:
        return tips
    A = ASSUMPTIONS
    tpr = a["tokens"] / a["req"] if a["req"] else 0
    pid = lambda kind: f"{scope}:{entity}:{kind}"
    label = f"{'project' if scope == 'project' else 'user'} {entity}"

    pro = {m: d for m, d in a["by_model"].items() if "pro" in m.lower() and d["tokens"] > 0}
    pro_cost = sum(d["cost"] for d in pro.values())
    if pro and pro_cost / cost >= 0.25 and tpr < 30000 and flash_cpt:
        top = max(pro, key=lambda m: pro[m]["cost"])
        p_cpt = pro[top]["cost"] / pro[top]["tokens"]
        if p_cpt > flash_cpt:
            saving = pro_cost * A["downshift_share"] * (1 - flash_cpt / p_cpt) * mf
            tips.append({"id": pid("downshift"), "scope": scope, "entity": entity, "priority": 2,
                         "title": f"Move routine {top} traffic for {label} to a Flash model",
                         "est_saving_usd": round(saving, 2), "risk": "medium - validate quality on a sample first",
                         "evidence": f"{100 * pro_cost / cost:.0f}% of spend is on Pro; only {tpr:,.0f} tokens/request, so most calls look routine"})
    think_pct = a["thoughts"] / a["tokens"]
    if think_pct >= 0.25:
        tips.append({"id": pid("thinking"), "scope": scope, "entity": entity, "priority": 2,
                     "title": f"Cap the thinking budget for {label}",
                     "est_saving_usd": round(cost * min(think_pct, 0.6) * A["thinking_cut"] * mf, 2),
                     "risk": "low - set thinking_budget per task; keep it high only for hard reasoning",
                     "evidence": f"{100 * think_pct:.0f}% of tokens are reasoning (thinking) tokens"})
    cached_pct = a["cached"] / a["prompt"] if a["prompt"] else 0
    ppr = a["prompt"] / a["req"] if a["req"] else 0
    if a["prompt"] > 0 and cached_pct < 0.20 and ppr >= 2000:
        tips.append({"id": pid("cache"), "scope": scope, "entity": entity, "priority": 2,
                     "title": f"Use context caching for {label}",
                     "est_saving_usd": round(cost * A["cache_saving"] * mf, 2),
                     "risk": "low - cache shared system prompts and reference documents",
                     "evidence": f"only {100 * cached_pct:.0f}% of input tokens are cached; {ppr:,.0f} input tokens/request"})
    ratio = a["completion"] / a["prompt"] if a["prompt"] else 0
    if ratio >= 1.0 and a["completion"] > 0:
        tips.append({"id": pid("verbose"), "scope": scope, "entity": entity, "priority": 2,
                     "title": f"Constrain answer length for {label}",
                     "est_saving_usd": round(cost * A["verbose_output_saving"] * mf, 2),
                     "risk": "low - set max_output_tokens and ask for concise answers",
                     "evidence": f"output tokens are {ratio:.1f}x input tokens (output is the expensive side)"})
    if tpr >= 50000:
        tips.append({"id": pid("context"), "scope": scope, "entity": entity, "priority": 2,
                     "title": f"Trim context or use retrieval for {label}",
                     "est_saving_usd": round(cost * A["big_context_saving"] * mf, 2),
                     "risk": "medium - check answer quality after trimming",
                     "evidence": f"{tpr:,.0f} tokens per request is very large"})
    if scope == "user" and total_cost and cost / total_cost >= 0.10:
        tips.append({"id": pid("concentration"), "scope": scope, "entity": entity, "priority": 1,
                     "title": f"Review usage of {label} and consider a per-user cap",
                     "est_saving_usd": 0.0, "risk": "governance - talk to the user before limiting",
                     "evidence": f"this one user accounts for {100 * cost / total_cost:.0f}% of all spend in the period"})
    tips.sort(key=lambda t: (-t["priority"], -t["est_saving_usd"]))
    return tips[:2]


def build_profiles(q, T, cond, mf, flash_cpt, total_cost):
    """Top projects and users with their usage profile + tips (what the admin asked for: advice from past usage)."""
    base = f"""
        SELECT {{key}} AS k, model, SUM(estimated_cost_usd) cost, SUM(total_tokens) tokens,
               SUM(COALESCE(request_count,1)) req, SUM(prompt_tokens) prompt, SUM(completion_tokens) completion,
               SUM(thoughts_tokens) thoughts, SUM(cached_content_tokens) cached
        FROM `{T}` WHERE {cond} GROUP BY k, model"""
    result = {"projects": [], "users": []}
    all_tips = []
    for key, scope, n in (("project_id", "project", 5), ("user_id", "user", 5)):
        try:
            rows = q(base.replace("{key}", key))
        except Exception:
            continue
        grouped = {}
        for r in rows:
            grouped.setdefault(r["k"], []).append(r)
        ranked = sorted(grouped.items(), key=lambda kv: -sum(float(x["cost"] or 0) for x in kv[1]))[:n]
        for ent, rs in ranked:
            a = _agg(rs)
            tips = build_tips(scope, ent, a, mf, flash_cpt, total_cost)
            result["projects" if scope == "project" else "users"].append(
                {"id": ent, **_metrics(a), "tip_ids": [t["id"] for t in tips]})
            all_tips.extend(tips)
    all_tips.sort(key=lambda t: (-t["priority"], -t["est_saving_usd"]))
    return result, all_tips[:10]


# ----------------------------------------------------------------------------
# Model lifecycle exposure (last 30 days)
# ----------------------------------------------------------------------------
def build_lifecycle(q, T, today):
    rows = q(f"""
        SELECT project_id, model, SUM(estimated_cost_usd) cost, SUM(total_tokens) tokens
        FROM `{T}`
        WHERE DATE(event_timestamp,'Asia/Singapore') >= DATE_SUB(CURRENT_DATE('Asia/Singapore'), INTERVAL 30 DAY)
        GROUP BY project_id, model""")
    total = sum(float(r["cost"] or 0) for r in rows) or 0.0
    by_model, tips = {}, []
    for r in rows:
        lc = lifecycle_for(r["model"], today)
        if not lc:
            continue
        d = by_model.setdefault(r["model"], {"model": r["model"], **lc, "cost_30d_usd": 0.0, "projects": {}})
        c = float(r["cost"] or 0)
        d["cost_30d_usd"] += c
        d["projects"][r["project_id"]] = d["projects"].get(r["project_id"], 0.0) + c
    models = []
    for d in sorted(by_model.values(), key=lambda x: -x["cost_30d_usd"]):
        top = sorted(d.pop("projects").items(), key=lambda kv: -kv[1])[:3]
        d["cost_30d_usd"] = round(d["cost_30d_usd"], 2)
        d["share_of_30d_cost_pct"] = round(100 * d["cost_30d_usd"] / total, 1) if total else 0.0
        d["top_projects"] = [p for p, _ in top]
        models.append(d)
    for d in models[:3]:
        tips.append({"id": f"model:{d['model']}:migrate", "scope": "portfolio", "entity": d["model"], "priority": 3,
                     "title": f"Migrate workloads off {d['model']}: {d['note']}",
                     "est_saving_usd": 0.0, "risk": "medium - test prompts on the replacement model first",
                     "evidence": f"{d['share_of_30d_cost_pct']}% of the last 30 days' cost; top projects: {', '.join(d['top_projects'])}"})
    exposed = round(sum(m["share_of_30d_cost_pct"] for m in models), 1)
    return {"source": MODEL_LIFECYCLE_SOURCE, "exposed_pct_of_30d_cost": exposed, "models": models[:6]}, tips
