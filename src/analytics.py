"""
analytics.py  (B6)  - temporal waste patterns, TWCI, recovery, forecast,
menu recommendations.  THE HEADLINE MODULE.

Input
    dataset/generated/events.csv     (one row per plate x dish)
    dataset/menu.csv                 (to keep only dishes that were on the menu)
    dataset/generated/answer_key.csv (planted patterns, for scoring)
    optional --pred results/waste_predictions.csv  (event_id,dish,pred_waste_pct
                                     from waste.py / B5). Without it the TRUE
                                     waste is used ("oracle" run).

What it computes
  N1  TWCI per cell (dish, weekday, meal):
        mu  = mean waste %          R = share of weeks with weekly mean > 30 %
        TWCI = mu x R
      pattern classes:
        chronic          - every slot of the dish is high (mu>=30, R>=0.75)
        weekly_recurring - high only in some slots of the dish
        drifting         - waste rises week by week (slope >= 2 %/wk, p<0.05)
        serving_decay    - late plates (minutes > breakpoint) waste more (N2)
        pairing          - dish wasted more when partner dish is on the menu
        sporadic         - occasional spikes (>=10 % of plates high) not
                           explained by anything above
        healthy          - none of the above
  N2  serving-time decay: per-dish regression waste ~ minutes + best breakpoint
  N5  forecast (seasonal-naive = same slot last week) vs baselines,
      pairing lift table, quantity plan per slot, plain-English
      recommendations.
  Pattern-recovery precision / recall against answer_key.csv.

Outputs (results/analytics/ by default) + results/metrics.json (merged)
    heatmap_waste.png, heatmap_twci.png, drift.png, serving_decay.png,
    pairing.png, cells.csv, detected_patterns.csv, recovery.csv,
    pairing_lift.csv, forecast.csv, quantity_plan.csv, recommendations.md

Run
    python src/analytics.py --dataset dataset
    python src/analytics.py --dataset dataset --pred results/waste_predictions.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import MEAL_ORDER, WEEKDAYS  # noqa: E402
from simulate_timeline import load_menu  # noqa: E402

# ---------------------------------------------------------------- thresholds
HIGH = 30.0          # % : "high waste"
R_HIGH = 0.75        # recurrence needed for a high cell
DRIFT_SLOPE = 2.0    # % per week
DECAY_DIFF = 8.0     # % late-minus-early
PAIR_LIFT = 8.0      # % with-minus-without
SPORADIC_FRAC = 0.10 # share of plates above HIGH: at least this ...
SPORADIC_MAX = 0.50  # ... but below this (more than half high = not 'occasional')
ALPHA = 0.05
TARGET_WASTE = 10.0  # % healthy waste we plan for
SHORT = {"breakfast": "bfast", "lunch": "lunch", "snacks": "snack", "dinner": "dinner"}


# ================================================================ data
def load_events(dataset, pred=None, keep_off_menu=False, noise_sd=0.0, seed=0):
    ds = Path(dataset)
    ev = pd.read_csv(ds / "generated" / "events.csv")
    if pred:
        p = pd.read_csv(pred)
        ev = ev.merge(p[["event_id", "dish", "pred_waste_pct"]],
                      on=["event_id", "dish"], how="inner")
        ev["waste"] = ev["pred_waste_pct"].clip(0, 100)
        source = f"predicted ({Path(pred).name})"
    else:
        ev["waste"] = ev["true_waste_pct"]
        source = "true waste (oracle)"
    if noise_sd > 0:   # robustness test: pretend the measurement has this error
        rng = np.random.default_rng(seed)
        ev["waste"] = (ev["waste"] + rng.normal(0, noise_sd, len(ev))).clip(0, 100)
        source += f" + noise sd {noise_sd}"
    ev["date"] = pd.to_datetime(ev["date"])
    ev["week"] = (ev["date"] - ev["date"].min()).dt.days // 7
    menu = load_menu(ds / "menu.csv")
    on = [d in menu.get((w, m), []) for d, w, m in zip(ev.dish, ev.weekday, ev.meal)]
    ev["on_menu"] = on
    if not keep_off_menu:
        ev = ev[ev.on_menu]
    ev = ev[ev.dish != "other"].copy()
    return ev, menu, source


def slot_order(ev):
    slots = ev[["weekday", "meal"]].drop_duplicates()
    slots["k"] = [WEEKDAYS.index(w) * 10 + (MEAL_ORDER.index(m) if m in MEAL_ORDER else 9)
                  for w, m in zip(slots.weekday, slots.meal)]
    return [(w, m) for w, m in slots.sort_values("k")[["weekday", "meal"]].values]


# ================================================================ N1 TWCI
def cell_table(ev, high=HIGH):
    """One row per (dish, weekday, meal): mu, R, TWCI, sd, n."""
    weekly = ev.groupby(["dish", "weekday", "meal", "week"])["waste"].mean().reset_index()
    rec = weekly.groupby(["dish", "weekday", "meal"])["waste"].agg(
        R=lambda s: float((s > high).mean()), weeks="count").reset_index()
    base = ev.groupby(["dish", "weekday", "meal"])["waste"].agg(
        mu="mean", sd="std", n="count").reset_index()
    cells = base.merge(rec, on=["dish", "weekday", "meal"])
    cells["TWCI"] = cells["mu"] * cells["R"]
    cells["high"] = (cells["mu"] >= high) & (cells["R"] >= R_HIGH)
    return cells.sort_values("TWCI", ascending=False)


# ================================================================ detectors
def trimmed_mean(x, cut=0.2):
    return float(stats.trim_mean(np.asarray(x, float), cut)) if len(x) else float("nan")


def detect_drift(ev, min_per_week=8):
    """Waste rising week by week. Robust to the occasional plate where a whole
    dish is left: the weekly level is a 20 %-trimmed mean, the per-plate
    regression must be significant after Bonferroni (one test per dish), and
    the trimmed weekly means must themselves rise >= DRIFT_SLOPE per week."""
    out = []
    dishes = list(ev.groupby("dish"))
    n_tests = max(1, len(dishes))
    for dish, g in dishes:
        counts = g.groupby("week").size()
        if (counts >= min_per_week).sum() < 3:
            continue
        r = stats.linregress(g.week, g.waste)
        wk = g.groupby("week").waste.apply(trimmed_mean)
        slope_tm = stats.linregress(wk.index.values, wk.values).slope
        if r.pvalue < ALPHA / n_tests and min(r.slope, slope_tm) >= DRIFT_SLOPE:
            out.append({"dish": dish, "weekday": "all", "meal": "all",
                        "pattern_type": "drifting", "effect": round(slope_tm, 2),
                        "p_value": r.pvalue,
                        "evidence": f"+{slope_tm:.1f}%/week (trimmed weekly means "
                                    f"{[round(float(v), 1) for v in wk.values]})"})
    return out


def decay_table(ev, breakpoints=range(30, 100, 10)):
    """Per dish: slope per hour and the best late/early breakpoint."""
    rows = []
    for dish, g in ev.groupby("dish"):
        if len(g) < 20:
            continue
        r = stats.linregress(g.minutes_since_start, g.waste)
        best = None
        for b in breakpoints:
            late, early = g[g.minutes_since_start > b].waste, g[g.minutes_since_start <= b].waste
            if len(late) < 8 or len(early) < 8:
                continue
            t = stats.ttest_ind(late, early, equal_var=False)
            diff = late.mean() - early.mean()
            if best is None or diff > best[1]:
                best = (b, diff, t.pvalue)
        rows.append({"dish": dish, "slope_per_hour": 60 * r.slope, "slope_p": r.pvalue,
                     "breakpoint": best[0] if best else np.nan,
                     "late_minus_early": best[1] if best else np.nan,
                     "p_value": best[2] if best else np.nan})
    return pd.DataFrame(rows, columns=["dish", "slope_per_hour", "slope_p", "breakpoint",
                                       "late_minus_early", "p_value"])


def detect_decay(dec):
    out = []
    n_tests = max(1, len(dec))
    for _, r in dec.iterrows():
        if r.late_minus_early >= DECAY_DIFF and r.p_value < ALPHA / n_tests:
            out.append({"dish": r.dish, "weekday": "all", "meal": "all",
                        "pattern_type": "serving_decay", "effect": round(r.late_minus_early, 1),
                        "p_value": r.p_value,
                        "evidence": f"+{r.late_minus_early:.1f}% after {int(r.breakpoint)} min"
                                    f" ({r.slope_per_hour:+.1f}%/hour)"})
    return out


def pairing_table(ev, menu):
    """lift(A | B) = mean waste of A when B is on the menu - when it is not."""
    rows = []
    dishes = sorted(ev.dish.unique())
    for a in dishes:
        g = ev[ev.dish == a]
        slot_menu = [set(menu.get((w, m), [])) for w, m in zip(g.weekday, g.meal)]
        for b in dishes:
            if b == a:
                continue
            with_b = np.array([b in s for s in slot_menu])
            if with_b.sum() < 8 or (~with_b).sum() < 8:
                continue
            x, y = g.waste.values[with_b], g.waste.values[~with_b]
            t = stats.ttest_ind(x, y, equal_var=False)
            rows.append({"dish": a, "partner": b, "with": x.mean(), "without": y.mean(),
                         "lift": x.mean() - y.mean(), "n_with": int(with_b.sum()),
                         "p_value": t.pvalue,
                         "n_slots_with": len({(w, m) for (w, m), f in
                                              zip(zip(g.weekday, g.meal), with_b) if f})})
    return pd.DataFrame(rows, columns=["dish", "partner", "with", "without", "lift", "n_with",
                                       "p_value", "n_slots_with"])


def pairing_consistent(ev, menu, dish, partner, without_mean, min_share=0.6):
    """A real pairing effect shows up in MOST slots where the partner is
    served - not just one slot (that would be a weekly-recurring pattern)."""
    g = ev[ev.dish == dish]
    slot_means = g.groupby(["weekday", "meal"]).waste.mean()
    with_slots = [s for s in slot_means.index if partner in menu.get(s, [])]
    if len(with_slots) == 0:
        return False
    raised = sum(slot_means[s] > without_mean + PAIR_LIFT / 2 for s in with_slots)
    return raised / len(with_slots) >= min_share


def detect_pairing(pairs, cells, ev=None, menu=None):
    """Best partner per dish. A partner only counts if it explains the dish's
    high waste better than any other partner (stops 'salad' being blamed
    because it happens to share slots with biryani), and if the effect is
    consistent across the partner's slots."""
    out = []
    if pairs.empty:
        return out
    n_tests = len(pairs)
    for dish, g in pairs.groupby("dish"):
        g = g.sort_values("lift", ascending=False)
        if ev is not None:
            ok = [pairing_consistent(ev, menu, dish, r.partner, r.without) for r in g.itertuples()]
            g = g[np.array(ok, bool)]
            if g.empty:
                continue
        top = g.iloc[0]
        if top.lift < PAIR_LIFT or top.p_value >= ALPHA / n_tests:
            continue
        if len(g) > 1 and g.iloc[1].lift > 0.8 * top.lift and g.iloc[1].p_value < ALPHA:
            # two partners almost equally good: pick the one present in fewer
            # slots (more specific); report the other as evidence
            cand = g[(g.lift > 0.8 * top.lift)]
            top = cand.sort_values(["n_slots_with", "lift"], ascending=[True, False]).iloc[0]
        out.append({"dish": dish, "weekday": "all", "meal": "all",
                    "pattern_type": "pairing", "partner": top.partner,
                    "effect": round(top.lift, 1), "p_value": top.p_value,
                    "evidence": f"{top['with']:.1f}% with {top.partner} vs "
                                f"{top.without:.1f}% without"})
    return out


def detect_cells_and_pairing(cells, menu):
    """Look at the HIGH cells (mu >= 30 %, recurring >= 75 % of weeks) of each dish:
      all of its slots high              -> chronic
      >= 3 slots RAISED above the dish's own normal level (>= +8 points over its
         lower-quartile slot) that share a partner dish: raised in >= 60 % of the
         slots with that partner and <= 20 % of the slots without it
                                         -> pairing (dish with partner)
      otherwise each high slot           -> weekly_recurring
    A pairing has to explain SEVERAL high slots; one or two high slots are a
    day/meal-specific (weekly-recurring) pattern, even if some other dish
    happens to be on the same menu."""
    out = []
    for dish, g in cells.groupby("dish"):
        S = [(w, m) for w, m in zip(g.weekday, g.meal)]
        H = {(w, m) for w, m, h in zip(g.weekday, g.meal, g.high) if h}
        # "elevated" = clearly above THIS dish's normal level (its lower-quartile
        # slot), used for pairing: a partner effect need not cross the absolute
        # 30 % line in every slot
        base = float(np.percentile(g.mu, 25)) if len(g) else 0.0
        E = {(w, m) for w, m, mu in zip(g.weekday, g.meal, g.mu) if mu >= base + PAIR_LIFT}
        if not H and len(E) < 3:
            continue
        info = {(r.weekday, r.meal): r for r in g.itertuples()}
        if H and len(H) == len(S):
            for sl in sorted(H):
                r = info[sl]
                out.append({"dish": dish, "weekday": sl[0], "meal": sl[1], "pattern_type": "chronic",
                            "effect": round(r.mu, 1), "p_value": np.nan,
                            "evidence": f"mu {r.mu:.1f}%, R {r.R:.2f}, TWCI {r.TWCI:.1f}"})
            continue
        best = None
        partners = sorted({d for sl in S for d in menu.get(sl, [])} - {dish, "other"})
        for b in partners:
            W = {sl for sl in S if b in menu.get(sl, [])}
            notW = set(S) - W
            if len(W) < 3 or not notW:
                continue
            hits = len(E & W)
            hit_rate, fp_rate = hits / len(W), len(E & notW) / len(notW)
            if hits >= 3 and hit_rate >= 0.6 and fp_rate <= 0.2:
                key = (hit_rate - fp_rate, -len(W))
                if best is None or key > best[0]:
                    best = (key, b, W)
        explained = set()
        if best:
            _, b, W = best
            w_mu = g[[(x, y) in W for x, y in zip(g.weekday, g.meal)]].mu.mean()
            o_mu = g[[(x, y) not in W for x, y in zip(g.weekday, g.meal)]].mu.mean()
            out.append({"dish": dish, "weekday": "all", "meal": "all", "pattern_type": "pairing",
                        "partner": b, "effect": round(w_mu - o_mu, 1), "p_value": np.nan,
                        "evidence": f"{w_mu:.1f}% with {b} vs {o_mu:.1f}% without "
                                    f"({len(E & W)} of {len(W)} {b} slots raised)"})
            explained = H & W
        for sl in sorted(H - explained):
            r = info[sl]
            out.append({"dish": dish, "weekday": sl[0], "meal": sl[1],
                        "pattern_type": "weekly_recurring", "effect": round(r.mu, 1),
                        "p_value": np.nan,
                        "evidence": f"mu {r.mu:.1f}%, R {r.R:.2f}, TWCI {r.TWCI:.1f}"})
    return out


def detect_sporadic(ev, explained_dishes):
    """Sporadic = this dish is left in large amounts clearly MORE OFTEN than
    dishes usually are (at least twice the typical rate), but not on most
    plates and without a fixed slot. With realistic eating, every dish is
    sometimes left - that alone is not a pattern."""
    out = []
    fr = ev.groupby("dish").waste.apply(lambda x: float((x > HIGH).mean()))
    typical = float(fr.median()) if len(fr) else 0.0
    for dish, g in ev.groupby("dish"):
        if dish in explained_dishes:
            continue
        frac = fr[dish]
        if max(SPORADIC_FRAC, 2 * typical) <= frac < SPORADIC_MAX and g.waste.mean() < HIGH:
            out.append({"dish": dish, "weekday": "all", "meal": "all",
                        "pattern_type": "sporadic", "effect": round(100 * frac, 1),
                        "p_value": np.nan,
                        "evidence": f"{100 * frac:.0f}% of plates above {HIGH:.0f}% "
                                    f"(typical dish: {100 * typical:.0f}%)"})
    return out


def detect_all(ev, menu, cells):
    det = detect_drift(ev)
    dec = decay_table(ev)
    det += detect_decay(dec)
    det += detect_cells_and_pairing(cells, menu)
    # pairing that is real but too weak to make whole slots "high": lift test
    # on the dish's plates OUTSIDE its high slots, partner in >= 3 slots
    high_slots = {(r.dish, r.weekday, r.meal) for r in cells.itertuples() if r.high}
    found = {d["dish"] for d in det}
    rest = ev[[(d not in found) and ((d, w, m) not in high_slots)
               for d, w, m in zip(ev.dish, ev.weekday, ev.meal)]]
    pairs = pairing_table(ev, menu)
    weak = pairing_table(rest, menu) if len(rest) else pairs.iloc[0:0]
    weak = weak[weak.n_slots_with >= 3]
    det += detect_pairing(weak, cells, rest, menu)
    det += detect_sporadic(ev, {d["dish"] for d in det})
    found = {d["dish"] for d in det}
    for dish in sorted(ev.dish.unique()):
        if dish not in found:
            det.append({"dish": dish, "weekday": "all", "meal": "all",
                        "pattern_type": "healthy", "effect": round(ev[ev.dish == dish].waste.mean(), 1),
                        "p_value": np.nan, "evidence": "no recurring pattern"})
    det = pd.DataFrame(det)
    if "partner" not in det:
        det["partner"] = np.nan
    return det, dec, pairs


def _slot_match(a, b):
    return a == "all" or b == "all" or a == b


def recovery(detected, key):
    """Precision / recall of non-healthy patterns. A detection matches a key
    row when dish + pattern_type agree and the slots are compatible."""
    key = key.copy()
    key["weekday"] = key["weekday"].fillna("all")
    key["meal"] = key["meal"].fillna("all")
    K = key[key.pattern_type != "healthy"].reset_index(drop=True)
    D = detected[detected.pattern_type != "healthy"].reset_index(drop=True)

    def match(d, k):
        return (d.dish == k.dish and d.pattern_type == k.pattern_type
                and _slot_match(d.weekday, k.weekday) and _slot_match(d.meal, k.meal))

    d_ok = [any(match(d, k) for _, k in K.iterrows()) for _, d in D.iterrows()]
    k_ok = [any(match(d, k) for _, d in D.iterrows()) for _, k in K.iterrows()]
    prec = float(np.mean(d_ok)) if len(D) else float("nan")
    rec = float(np.mean(k_ok)) if len(K) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    table = K.assign(recovered=k_ok)
    fps = D[~np.array(d_ok, bool)] if len(D) else D
    # healthy dishes correctly left alone
    healthy_key = set(key[key.pattern_type == "healthy"].dish)
    flagged = set(D.dish)
    healthy_ok = (len(healthy_key - flagged) / len(healthy_key)) if healthy_key else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1,
            "healthy_specificity": healthy_ok}, table, fps


# ================================================================ forecast
def forecast(ev):
    """Weekly mean per cell. Predict week w from weeks < w.
       seasonal_naive : same cell, last week
       cell_mean      : same cell, mean of all earlier weeks
       global_mean    : mean of ALL earlier plates (baseline)"""
    wk = ev.groupby(["dish", "weekday", "meal", "week"])["waste"].mean().reset_index()
    rows = []
    for w in sorted(wk.week.unique())[1:]:
        past_ev = ev[ev.week < w]
        gmean = past_ev.waste.mean()
        cur = wk[wk.week == w]
        for _, r in cur.iterrows():
            hist = wk[(wk.dish == r.dish) & (wk.weekday == r.weekday) &
                      (wk.meal == r.meal) & (wk.week < w)].sort_values("week")
            if hist.empty:
                continue
            rows.append({"dish": r.dish, "weekday": r.weekday, "meal": r.meal, "week": w,
                         "actual": r.waste, "seasonal_naive": hist.waste.iloc[-1],
                         "cell_mean": hist.waste.mean(), "global_mean": gmean})
    fc = pd.DataFrame(rows, columns=["dish", "weekday", "meal", "week", "actual",
                                     "seasonal_naive", "cell_mean", "global_mean"])
    mae = {m: float((fc[m] - fc.actual).abs().mean())
           for m in ("seasonal_naive", "cell_mean", "global_mean")} if len(fc) else {}
    return fc, mae


def next_week_plan(ev, cells, det):
    """Forecast next week per cell (cell mean, plus drift slope if the dish
    drifts) and turn it into a quantity factor:
        food eaten per plate  ~ (1 - waste)
        new quantity factor   = (1 - forecast) / (1 - TARGET_WASTE)"""
    drift = {r.dish: r.effect for r in det.itertuples() if r.pattern_type == "drifting"}
    last_week = ev.week.max()
    wk = ev.groupby(["dish", "weekday", "meal", "week"])["waste"].mean().reset_index()
    rows = []
    for (dish, w, m), g in wk.groupby(["dish", "weekday", "meal"]):
        if dish in drift:
            f = g[g.week == g.week.max()].waste.iloc[0] + drift[dish] * (last_week + 1 - g.week.max())
        else:
            f = g.waste.mean()
        f = float(np.clip(f, 0, 95))
        factor = (1 - f / 100) / (1 - TARGET_WASTE / 100)
        rows.append({"dish": dish, "weekday": w, "meal": m, "forecast_waste_pct": round(f, 1),
                     "quantity_factor": round(min(factor, 1.0), 2),
                     "change_pct": round(100 * (min(factor, 1.0) - 1), 0)})
    return pd.DataFrame(rows).sort_values("change_pct")


# ================================================================ recommendations
def recommendations(det, plan):
    lines = ["# Menu and quantity recommendations", ""]
    order = ["chronic", "weekly_recurring", "pairing", "serving_decay", "drifting", "sporadic"]
    for t in order:
        for r in det[det.pattern_type == t].itertuples():
            d = r.dish.capitalize()
            if t == "chronic":
                cut = plan[(plan.dish == r.dish) & (plan.weekday == r.weekday) & (plan.meal == r.meal)]
                c = int(-cut.change_pct.iloc[0]) if len(cut) else 0
                lines.append(f"- **{d}, {r.weekday} {r.meal} - chronic** ({r.evidence}). "
                             f"Cut the prepared quantity by about {c}%, or swap it for a more popular dish.")
            elif t == "weekly_recurring":
                cut = plan[(plan.dish == r.dish) & (plan.weekday == r.weekday) & (plan.meal == r.meal)]
                c = int(-cut.change_pct.iloc[0]) if len(cut) else 0
                lines.append(f"- **{d}, {r.weekday} {r.meal} only - weekly-recurring** ({r.evidence}). "
                             f"Normal on other days, so reduce it by about {c}% in this slot only, "
                             f"or move it to a slot where it is eaten.")
            elif t == "pairing":
                lines.append(f"- **{d} with {r.partner} - pairing effect** ({r.evidence}). "
                             f"Serve a smaller {r.dish} portion (or skip it) on {r.partner} days.")
            elif t == "serving_decay":
                lines.append(f"- **{d} - serving-time decay** ({r.evidence}). Cook it in two "
                             f"batches or keep it hot, so plates served late in the service are fresh.")
            elif t == "drifting":
                lines.append(f"- **{d} - rising waste** ({r.evidence}). Check the recipe and get "
                             f"student feedback now, before it becomes chronic.")
            elif t == "sporadic":
                lines.append(f"- **{d} - sporadic spikes** ({r.evidence}). No fixed slot: "
                             f"check batch quality on the spike days.")
    lines += ["", "## Biggest quantity cuts for next week", "",
              "| dish | slot | forecast waste % | change in quantity |", "|---|---|---|---|"]
    for r in plan[plan.change_pct <= -10].head(12).itertuples():
        lines.append(f"| {r.dish} | {r.weekday} {r.meal} | {r.forecast_waste_pct} | {int(r.change_pct)}% |")
    return "\n".join(lines) + "\n"


# ================================================================ plots
def plot_heatmap(cells, ev, value, path, title):
    slots = slot_order(ev)
    dishes = cells.groupby("dish")[value].max().sort_values(ascending=False).index.tolist()
    M = np.full((len(dishes), len(slots)), np.nan)
    for r in cells.itertuples():
        M[dishes.index(r.dish), slots.index((r.weekday, r.meal))] = getattr(r, value)
    fig, ax = plt.subplots(figsize=(max(8, 0.38 * len(slots)), 0.45 * len(dishes) + 1.8))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(dishes)), dishes)
    ax.set_xticks(range(len(slots)), [f"{w[:3]} {SHORT.get(m, m)}" for w, m in slots], rotation=90, fontsize=8)
    for i in range(len(dishes)):
        for j in range(len(slots)):
            if not np.isnan(M[i, j]) and M[i, j] >= (HIGH if value == "mu" else 15):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=value)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drift(ev, path):
    wk = ev.groupby(["dish", "week"]).waste.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    wk.plot(ax=ax, marker="o")
    ax.set_xlabel("week")
    ax.set_ylabel("mean waste %")
    ax.set_title("Weekly mean waste per dish")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_decay(ev, dec, path):
    if dec.empty:
        return
    bins = np.arange(0, 135, 15)
    ev = ev.assign(mbin=pd.cut(ev.minutes_since_start, bins, include_lowest=True))
    fig, ax = plt.subplots(figsize=(6, 4))
    top = dec.sort_values("slope_per_hour", ascending=False).dish.head(5).tolist()
    for dish in top:
        g = ev[ev.dish == dish].groupby("mbin", observed=True).waste.mean()
        ax.plot([b.mid for b in g.index], g.values, marker="o", label=dish)
    ax.set_xlabel("minutes since service started")
    ax.set_ylabel("mean waste %")
    ax.set_title("Serving-time decay (top 5 dishes by slope)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pairing(pairs, path):
    if pairs.empty:
        return
    top = pairs.sort_values("lift", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([f"{a} | with {b}" for a, b in zip(top.dish, top.partner)][::-1], top.lift[::-1])
    ax.set_xlabel("extra waste when partner is on the menu (%)")
    ax.set_title("Top pairing lifts")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ================================================================ main
def run(dataset, pred=None, out="results/analytics", metrics_path="results/metrics.json",
        noise_sd=0.0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ev, menu, source = load_events(dataset, pred, noise_sd=noise_sd)
    if ev.week.nunique() < 3 or ev.event_id.nunique() < 100:
        print("  WARNING: very little data (fewer than 3 weeks or 100 plates) - patterns and "
              "forecasts will be weak or empty. Generate all events for a real run.")
    print(f"  {len(ev)} plate-dish rows on the menu, {ev.event_id.nunique()} plates, "
          f"{ev.week.nunique()} weeks  |  waste source: {source}")

    cells = cell_table(ev)
    det, dec, pairs = detect_all(ev, menu, cells)
    fc, fmae = forecast(ev)
    plan = next_week_plan(ev, cells, det)

    cells.to_csv(out / "cells.csv", index=False)
    det.to_csv(out / "detected_patterns.csv", index=False)
    dec.to_csv(out / "serving_decay.csv", index=False)
    pairs.sort_values("lift", ascending=False).to_csv(out / "pairing_lift.csv", index=False)
    fc.to_csv(out / "forecast.csv", index=False)
    plan.to_csv(out / "quantity_plan.csv", index=False)
    (out / "recommendations.md").write_text(recommendations(det, plan))

    plot_heatmap(cells, ev, "mu", out / "heatmap_waste.png", "Mean waste % per dish and slot")
    plot_heatmap(cells, ev, "TWCI", out / "heatmap_twci.png", "TWCI = mean waste x recurrence")
    plot_drift(ev, out / "drift.png")
    plot_decay(ev, dec, out / "serving_decay.png")
    plot_pairing(pairs, out / "pairing.png")

    print("\n  detected patterns:")
    for r in det[det.pattern_type != "healthy"].itertuples():
        extra = f" with {r.partner}" if isinstance(r.partner, str) else ""
        print(f"    {r.pattern_type:<16} {r.dish:<8}{extra:<14} {r.weekday:<9} {r.meal:<9} {r.evidence}")
    healthy = det[det.pattern_type == "healthy"].dish.tolist()
    print(f"    healthy: {', '.join(healthy)}")

    metrics = {"analytics_source": source}
    key_path = Path(dataset) / "generated" / "answer_key.csv"
    if key_path.exists():
        key = pd.read_csv(key_path)
        m, table, fps = recovery(det, key)
        table.to_csv(out / "recovery.csv", index=False)
        metrics.update({f"pattern_{k}": round(v, 3) for k, v in m.items()})
        print(f"\n  PATTERN RECOVERY vs answer key: precision {m['precision']:.2f}  "
              f"recall {m['recall']:.2f}  F1 {m['f1']:.2f}  "
              f"(healthy dishes left alone: {m['healthy_specificity']:.2f})")
        for r in table[~table.recovered].itertuples():
            print(f"    MISSED: {r.pattern_type} {r.dish} {r.weekday} {r.meal}")
        for r in fps.itertuples():
            print(f"    FALSE ALARM: {r.pattern_type} {r.dish} {r.weekday} {r.meal} ({r.evidence})")
    if fmae:
        metrics.update({f"forecast_mae_{k}": round(v, 2) for k, v in fmae.items()})
        print(f"\n  FORECAST MAE (weekly cell means): seasonal-naive {fmae['seasonal_naive']:.2f}  "
              f"cell-mean {fmae['cell_mean']:.2f}  global-mean baseline {fmae['global_mean']:.2f}")

    mp = Path(metrics_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    allm = json.loads(mp.read_text()) if mp.exists() else {}
    allm.update(metrics)
    mp.write_text(json.dumps(allm, indent=2))
    print(f"\n  outputs in {out}/  |  metrics merged into {mp}")
    return det, metrics


def robustness(dataset, pred=None, out="results/analytics", levels=(0, 5, 10, 15, 20), seeds=3):
    """Pattern recovery vs measurement noise -> robustness.csv / .png"""
    import contextlib
    import io
    key = pd.read_csv(Path(dataset) / "generated" / "answer_key.csv")
    rows = []
    for sd in levels:
        for s in range(seeds if sd > 0 else 1):
            ev, menu, _ = load_events(dataset, pred, noise_sd=sd, seed=s)
            with contextlib.redirect_stdout(io.StringIO()):
                cells = cell_table(ev)
                det, _, _ = detect_all(ev, menu, cells)
            m, _, _ = recovery(det, key)
            rows.append({"noise_sd": sd, "seed": s, **m})
    df = pd.DataFrame(rows).groupby("noise_sd")[["precision", "recall", "f1"]].mean().reset_index()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "robustness.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for c in ("precision", "recall", "f1"):
        ax.plot(df.noise_sd, df[c], marker="o", label=c)
    ax.set_xlabel("measurement noise sd (% points)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pattern recovery vs measurement error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)
    print(df.round(2).to_string(index=False))
    print(f"  wrote {out / 'robustness.csv'} and robustness.png")
    return df


def main():
    ap = argparse.ArgumentParser(description="Temporal waste analytics (B6)")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--pred", default=None, help="waste_predictions.csv from waste.py")
    ap.add_argument("--out", default="results/analytics")
    ap.add_argument("--metrics", default="results/metrics.json")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="add gaussian measurement noise (sd, %%) - robustness test")
    ap.add_argument("--robustness", action="store_true",
                    help="run recovery for noise sd 0,5,10,15,20 and plot it")
    args = ap.parse_args()
    if args.robustness:
        robustness(args.dataset, args.pred, args.out)
    else:
        run(args.dataset, args.pred, args.out, args.metrics, args.noise)


if __name__ == "__main__":
    main()
