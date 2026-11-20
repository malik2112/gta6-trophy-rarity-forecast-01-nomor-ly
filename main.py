#!/usr/bin/env python3
"""Entry for make.fun gta6-trophy-rarity-forecast-01: Pooled logit trend in log-time.

Reads every <id>.in case (schema makefun/gta6-trophies-case/v1) from the
input directory and writes <id>.out with five quantiles (5/25/50/75/95%)
per trophy. Standard library only, no network, deterministic.
Outside the grader: MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for
/input and /output.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

"""Shared helpers for GTA VI trophy rarity forecasters (stdlib only).

Model family (from dev leaderboard titles): counts + dilution.
Earn rate r_i(t) = E_i(t) / P(t), where E_i = earners of trophy i and
P = players tracked. Both counts are monotone and grow roughly linearly
in log-time since launch, then saturate. We extrapolate log-counts in
log-time with per-trophy slopes shrunk toward the pooled slope, cap
earners at players, and put an honest logit-normal spread around the
median that widens with the horizon.
"""
import math
from datetime import datetime

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LAUNCH = datetime.fromisoformat("2026-11-19T00:00:00+00:00")


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    c = clamp(pct)
    return math.log(c / (100 - c))


def inv_logit(lg):
    return 100 / (1 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def log_time_hours(t):
    """log1p of hours since launch (launch = 2026-11-19T00:00Z)."""
    hrs = max(0.0, (t - LAUNCH).total_seconds() / 3600.0)
    return math.log1p(hrs)


def wls(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return (ys[-1] if ys else 0.0, 0.0)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b)


def extract(case):
    """Return (xs, players, earners) with xs log-time per read."""
    hist = case["history"]
    xs = [log_time_hours(parse_time(h["slot"] if h.get("slot") else h["at"])) for h in hist]
    players = [float(h["players_tracked"]) for h in hist]
    earners = []
    for i in range(len(case["trophies"])):
        col = []
        for j, h in enumerate(hist):
            r = h["rates"][i]
            col.append(None if r is None else float(r) / 100.0 * players[j])
        earners.append(col)
    return xs, players, earners


def forecast_counts(case, shrink=0.5, sigma_per_sqrt_day=0.18, recent_boost=False,
                    saturate=False, logit_trend=False):
    xs, players, earners = extract(case)
    target_x = log_time_hours(parse_time(case["target"]))
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    n = len(xs)
    weights = [1.0] * n
    if recent_boost and n >= 4:
        weights = [2.0 if j >= n // 2 else 1.0 for j in range(n)]

    # Players forecast: log-counts in log-time, slope floored at 0 (counts never fall),
    # capped and damped so a steep launch ramp cannot explode over a long horizon.
    lp = [math.log(max(1.0, p)) for p in players]
    a_p, b_p = wls(xs, lp, weights)
    b_p = min(3.0, max(0.0, b_p))
    dx_all = max(0.0, target_x - xs[-1])
    dx_damped_all = dx_all / (1.0 + 0.4 * dx_all)
    p_fit_last = math.exp(a_p + b_p * xs[-1])
    p_target = max(players[-1], p_fit_last * math.exp(b_p * dx_damped_all)) if n >= 2 else players[-1]

    # Pooled earner slope (across trophies with >=2 non-null reads).
    slopes = []
    fits = []
    for col in earners:
        pts = [(xs[j], math.log(max(0.5, col[j])), weights[j]) for j in range(n) if col[j] is not None]
        if len(pts) >= 2:
            a, b = wls([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
        elif pts:
            a, b = pts[0][1], 0.0
        else:
            a, b = math.log(0.5), 0.0
        fits.append((a, b))
        slopes.append(b)
    pooled = sorted(slopes)[len(slopes) // 2] if slopes else 0.0

    spread = max(0.02, sigma_per_sqrt_day * math.sqrt(horizon_days))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = None
        for j in range(n - 1, -1, -1):
            if case["history"][j]["rates"][i] is not None:
                last_rate = float(case["history"][j]["rates"][i])
                break
        if last_rate is None:
            last_rate = 0.0
        a, b = fits[i]
        b_shrunk = min(2.0, max(0.0, (1 - shrink) * b + shrink * pooled))
        dx = max(0.0, target_x - xs[-1])
        dx_damped = dx / (1.0 + 0.4 * dx)
        if logit_trend:
            # Trend on the logit of the rate itself, shrunk toward flat.
            pts = [(xs[j], logit(float(case["history"][j]["rates"][i])), weights[j])
                   for j in range(n) if case["history"][j]["rates"][i] is not None]
            if len(pts) >= 2:
                aa, bb = wls([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
                bb = max(-1.5, min(1.5, (1 - shrink) * bb))
                med_lg = logit(last_rate) + bb * dx_damped
            else:
                med_lg = logit(last_rate)
            median = inv_logit(med_lg)
        else:
            e_fit_last = math.exp(a + b_shrunk * xs[-1])
            e_trend = e_fit_last * math.exp(b_shrunk * dx_damped)
            if saturate:
                e_last0 = max(0.5, earners[i][-1] if earners[i][-1] is not None else 0.5)
                e_target = math.exp(0.5 * math.log(e_trend) + 0.5 * math.log(e_last0))
            else:
                e_target = e_trend
            e_last = earners[i][-1]
            if e_last is not None:
                e_target = max(e_last, e_target)
            e_target = min(e_target, p_target)
            median = 100.0 * e_target / p_target if p_target > 0 else last_rate
        lg = logit(median)
        q = [round4(inv_logit(lg + z * spread)) for z in Z]
        # enforce non-decreasing
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_counts(case, shrink=0.5, sigma_per_sqrt_day=0.20, logit_trend=True)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
