"""
simulate_timeline.py  (B2, part 1)  - makes the 4-week mess log.

Reads
    dataset/classes.txt      class names (YOLO ids)
    dataset/menu.csv         weekday,meal,dishes   (dishes separated by ';')
    dataset/labels/*.txt     to know which dishes each before photo contains
    dataset/patterns.yaml    planted patterns (see patterns_TEMPLATE.yaml)
Writes
    dataset/generated/events.csv
        event_id,before_img,after_img,date,weekday,meal,minutes_since_start,
        lighting,dish,true_waste_pct          (one row per dish per plate)
    dataset/generated/answer_key.csv
        dish,weekday,meal,pattern_type,notes

How it works
    For every date (weeks x 7 days) and every meal on that day's menu,
    it creates a random number of plates. For each plate it picks a real
    before photo that contains one of the menu's dishes (it cycles through
    the menu dishes so every dish gets plates), draws minutes_since_start,
    and samples true_waste_pct for every dish on that photo:
        healthy noise (uniform base_waste)  ->  replaced/raised by any
        planted pattern that matches  ->  + gaussian jitter, clipped 0..100.
    "waste %" = share of the dish's served area still left on the plate.

Run
    python src/simulate_timeline.py --dataset dataset
"""

import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import (MEAL_ORDER, clean_dish, clean_meal, clean_weekday,
                    label_classes, read_classes)

IMG_EXT = (".jpg", ".jpeg", ".png")


# ------------------------------------------------------------------ loading
def load_menu(path):
    """menu.csv -> dict {(weekday, meal): [dish, ...]}"""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    menu = {}
    for _, r in df.iterrows():
        key = (clean_weekday(r["weekday"]), clean_meal(r["meal"]))
        dishes = [clean_dish(d) for d in str(r["dishes"]).split(";") if d.strip()]
        menu.setdefault(key, [])
        menu[key] += [d for d in dishes if d not in menu[key]]
    return menu


def load_patterns(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    pats = []
    for p in cfg.get("patterns", []) or []:
        p = dict(p)
        p["type"] = str(p["type"]).strip().lower()
        p["dish"] = clean_dish(p["dish"])
        p["weekday"] = clean_weekday(p.get("weekday", "all"))
        p["meal"] = clean_meal(p["meal"]) if p.get("meal") not in (None, "", "all", "*") else "all"
        if "with" in p:
            p["with"] = clean_dish(p["with"])
        pats.append(p)
    cfg["patterns"] = pats
    return cfg


def index_images(before_dir, labels_dir, classes):
    """dict before_img_name -> set of dish names found in its label file."""
    idx = {}
    missing = []
    for p in sorted(Path(before_dir).iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        lab = Path(labels_dir) / (p.stem + ".txt")
        if not lab.exists():
            missing.append(p.name)
            continue
        ids = label_classes(lab)
        dishes = {classes[i] for i in ids if 0 <= i < len(classes)}
        if dishes:
            idx[p.name] = dishes
    if missing:
        print(f"  WARNING: {len(missing)} photos have no label file and are skipped "
              f"(e.g. {missing[:3]})")
    return idx


# ------------------------------------------------------------------ waste model
def matches(p, weekday, meal):
    return p["weekday"] in ("all", weekday) and p["meal"] in ("all", meal)


def base_waste(rng, cfg):
    """Normal ('healthy') waste for one dish on one plate.
    If patterns.yaml has `base_waste_mixture` (list of [probability, low, high]),
    students behave differently: most finish a dish, some leave part of it,
    a few leave nearly all of it. Otherwise: uniform base_waste [low, high]."""
    mix = cfg.get("base_waste_mixture")
    if mix:
        probs = np.array([m[0] for m in mix], float)
        k = rng.choice(len(mix), p=probs / probs.sum())
        return rng.uniform(mix[k][1], mix[k][2])
    lo, hi = cfg.get("base_waste", [5, 15])
    return rng.uniform(lo, hi)


def sample_waste(rng, dish, week, weekday, meal, minutes, context, cfg, plate_offset=0.0):
    """True waste % for one dish on one plate. plate_offset = this student's
    general appetite (some students leave more of everything)."""
    value = base_waste(rng, cfg) + plate_offset
    pats = [p for p in cfg["patterns"] if p["dish"] == dish and matches(p, weekday, meal)]

    # 1) "level" patterns replace the healthy value
    for p in pats:
        t = p["type"]
        if t in ("chronic", "weekly_recurring", "level"):
            value = rng.normal(p.get("waste", 40), p.get("sd", 5))
        elif t == "drifting":
            value = p.get("start", 12) + p.get("slope_per_week", 5) * week \
                    + rng.normal(0, p.get("sd", 2))
        elif t == "sporadic":
            if rng.random() < p.get("prob", 0.15):
                value = rng.normal(p.get("waste", 50), p.get("sd", 8))
    # 2) additive effects on top
    for p in pats:
        t = p["type"]
        if t == "serving_decay" and minutes > p.get("after_minutes", 60):
            value += p.get("extra", 15)
        elif t == "pairing" and p.get("with") in context:
            value += p.get("extra", 20)

    value += rng.normal(0, cfg.get("noise_sd", 3))
    return float(np.clip(value, 0, 100))


def answer_key(cfg, menu):
    rows, planted = [], set()
    for p in cfg["patterns"]:
        t = p["type"]
        if t in ("chronic", "weekly_recurring", "level"):
            note = f"mean {p.get('waste', 40)}% sd {p.get('sd', 5)}"
        elif t == "drifting":
            note = f"start {p.get('start', 12)}% +{p.get('slope_per_week', 5)}%/week"
        elif t == "serving_decay":
            note = f"+{p.get('extra', 15)}% when minutes > {p.get('after_minutes', 60)}"
        elif t == "pairing":
            note = f"+{p.get('extra', 20)}% when served with {p.get('with')}"
        elif t == "sporadic":
            note = f"spike to {p.get('waste', 50)}% with prob {p.get('prob', 0.15)}"
        else:
            note = "noise only"
        rows.append([p["dish"], p["weekday"], p["meal"], t, note])
        planted.add(p["dish"])
    menu_dishes = sorted({d for ds in menu.values() for d in ds})
    for d in menu_dishes:
        if d not in planted:
            rows.append([d, "all", "all", "healthy", "auto: no planted pattern"])
    return pd.DataFrame(rows, columns=["dish", "weekday", "meal", "pattern_type", "notes"])


def check_config(cfg, menu, img_index, classes):
    """Print warnings for things that would silently break the patterns."""
    menu_dishes = {d for ds in menu.values() for d in ds}
    photo_dishes = set().union(*img_index.values()) if img_index else set()
    for d in sorted(menu_dishes - set(classes)):
        print(f"  WARNING: menu dish '{d}' is not in classes.txt")
    for d in sorted(menu_dishes - photo_dishes):
        print(f"  WARNING: menu dish '{d}' is in no labelled photo -> it will never be served")
    for p in cfg["patterns"]:
        d = p["dish"]
        if d not in menu_dishes:
            print(f"  WARNING: pattern dish '{d}' ({p['type']}) is not on the menu")
        if p["type"] in ("chronic", "weekly_recurring") and p["weekday"] != "all":
            slot = (p["weekday"], p["meal"]) if p["meal"] != "all" else None
            if slot and d not in menu.get(slot, []):
                print(f"  WARNING: '{d}' is not on the menu at {slot} - its {p['type']} "
                      f"pattern will only show via off-menu photos")
            if p["type"] == "chronic":
                other = [k for k, v in menu.items() if d in v and not matches(p, *k)]
                if other:
                    print(f"  NOTE: chronic dish '{d}' is also served at {other[:3]} with "
                          f"normal waste, so analytics may call it weekly-recurring")
        if p["type"] == "pairing" and p.get("with") not in menu_dishes:
            print(f"  WARNING: pairing partner '{p.get('with')}' is not on the menu")


# ------------------------------------------------------------------ main
def simulate(dataset, out_dir=None):
    ds = Path(dataset)
    out = Path(out_dir) if out_dir else ds / "generated"
    out.mkdir(parents=True, exist_ok=True)

    classes = read_classes(ds / "classes.txt")
    menu = load_menu(ds / "menu.csv")
    cfg = load_patterns(ds / "patterns.yaml")
    img_index = index_images(ds / "before", ds / "labels", classes)
    if not img_index:
        raise SystemExit("No labelled before photos found - nothing to simulate.")
    print(f"  {len(img_index)} labelled photos, {len(menu)} menu slots, "
          f"{len(cfg['patterns'])} patterns")
    check_config(cfg, menu, img_index, classes)

    rng = np.random.default_rng(cfg.get("seed", 42))
    weeks = int(cfg.get("weeks", 4))
    start = pd.to_datetime(str(cfg.get("start_date", "2026-10-05"))).date()
    pmin, pmax = cfg.get("plates_per_slot", [5, 9])
    mmin, mmax = cfg.get("minutes_range", [0, 120])
    light = {"breakfast": "warm_bright", "lunch": "daylight",
             "snacks": "neutral", "dinner": "cool_dim"}
    light.update(cfg.get("lighting", {}) or {})

    strict = bool(cfg.get("strict_menu", False))
    by_dish = defaultdict(list)                     # dish -> photos with it
    for name, dishes in img_index.items():
        for d in dishes:
            by_dish[d].append(name)

    rows, eid, skipped = [], 0, 0
    for day in range(weeks * 7):
        date = start + dt.timedelta(days=day)
        week = day // 7
        weekday = date.strftime("%A")
        meals = sorted([m for (w, m) in menu if w == weekday],
                       key=lambda m: MEAL_ORDER.index(m) if m in MEAL_ORDER else 99)
        for meal in meals:
            menu_dishes = menu[(weekday, meal)]
            if all(d == "other" for d in menu_dishes):   # nothing we analyse: skip slot
                continue
            n = int(rng.integers(pmin, pmax + 1))
            order = list(rng.permutation(menu_dishes))
            used = set()
            # strict_menu: a photo may only be served if EVERY dish on it is on this
            # slot's menu (e.g. a photo with chicken never appears at a veg lunch)
            allowed = ([i for i, s in img_index.items() if s <= set(menu_dishes)]
                       if strict else list(img_index))
            for k in range(n):
                target = order[k % len(order)]
                cands = [i for i in by_dish.get(target, []) if i in allowed]
                if not cands:   # nobody photographed this dish: any photo with overlap
                    cands = [i for i in allowed if img_index[i] & set(menu_dishes)]
                if not cands:
                    skipped += 1
                    continue
                fresh = [c for c in cands if c not in used] or cands
                w = np.array([1 + len(img_index[c] & set(menu_dishes)) ** 2 for c in fresh],
                             float)
                img = fresh[rng.choice(len(fresh), p=w / w.sum())]
                used.add(img)

                eid += 1
                event_id = f"e{eid:05d}"
                minutes = int(rng.integers(mmin, mmax + 1))
                context = set(menu_dishes) | img_index[img]
                plate_offset = rng.normal(0, cfg.get("appetite_sd", 0))
                for dish in sorted(img_index[img]):
                    waste = sample_waste(rng, dish, week, weekday, meal, minutes,
                                         context - {dish}, cfg, plate_offset)
                    rows.append([event_id, img, f"{event_id}.jpg", date.isoformat(),
                                 weekday, meal, minutes, light.get(meal, "neutral"),
                                 dish, round(waste, 1)])

    cols = ["event_id", "before_img", "after_img", "date", "weekday", "meal",
            "minutes_since_start", "lighting", "dish", "true_waste_pct"]
    events = pd.DataFrame(rows, columns=cols)
    events.to_csv(out / "events.csv", index=False)
    key = answer_key(cfg, menu)
    key.to_csv(out / "answer_key.csv", index=False)

    # ---- short summary so you can see the patterns are really there
    print(f"\n  events (plates): {events.event_id.nunique()}   rows (plate x dish): {len(events)}")
    if skipped:
        print(f"  WARNING: {skipped} plates skipped (no photo matched the menu)")
    for p in cfg["patterns"]:
        if p["type"] == "healthy":
            continue
        sel = events.dish == p["dish"]
        if p["weekday"] != "all":
            sel &= events.weekday == p["weekday"]
        if p["meal"] != "all":
            sel &= events.meal == p["meal"]
        rest = (events.dish == p["dish"]) & ~sel
        msg = f"  {p['type']:<16} {p['dish']:<8} {p['weekday']:<9} {p['meal']:<9} rows={sel.sum():4d}"
        if p["type"] == "drifting" and sel.any():
            wk = events[sel].groupby(pd.to_datetime(events[sel].date).dt.isocalendar().week)["true_waste_pct"].mean()
            msg += "  weekly means: " + ", ".join(f"{v:.0f}" for v in wk.values)
        elif p["type"] == "serving_decay" and sel.any():
            late = events[sel].minutes_since_start > p.get("after_minutes", 60)
            msg += (f"  early={events[sel][~late].true_waste_pct.mean():.1f}%"
                    f"  late={events[sel][late].true_waste_pct.mean():.1f}%")
        elif p["type"] == "pairing" and sel.any():
            sub = events[sel]
            menu_has = [p.get("with") in menu.get((w, m), [])
                        for w, m in zip(sub.weekday, sub.meal)]
            menu_has = np.array(menu_has)
            msg += (f"  with {p.get('with')}={sub[menu_has].true_waste_pct.mean():.1f}%"
                    f"  without={sub[~menu_has].true_waste_pct.mean():.1f}%")
        elif sel.any():
            msg += f"  mean={events[sel].true_waste_pct.mean():5.1f}%"
            if rest.any() and p["type"] in ("chronic", "weekly_recurring"):
                msg += f"  (other slots {events[rest].true_waste_pct.mean():.1f}%)"
        print(msg)
    print(f"\n  wrote {out / 'events.csv'} and {out / 'answer_key.csv'}")
    return events, key


def main():
    ap = argparse.ArgumentParser(description="Simulate the 4-week mess log")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default=None, help="default: <dataset>/generated")
    args = ap.parse_args()
    simulate(args.dataset, args.out)


if __name__ == "__main__":
    main()
