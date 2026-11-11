"""
waste.py  (B5)  - waste % per dish + MAE + ablation.

Method ("region" method)
  1. rectify + colour-correct both photos, align after -> before (align.py)
  2. dish regions come from the BEFORE photo (labels now; segment.py later)
  3. find food pixels on the aligned AFTER photo (food vs empty plate)
  4. waste % of dish d = food pixels inside d's region / area of d's region
  So the after photo only needs "food or plate"; the dish name comes from
  where the food sits - that is why alignment matters.

After-photo food mask options (--seg):
  classical : Lab colour distance from the plate colour + Otsu threshold
              (default, no training needed)
  gt        : ground-truth after labels (oracle - tests geometry only)
  (segment.py will add yolo later)

Ablation (--ablation): full / no_align / no_color / neither, MAE overall
and by lighting.  Writes
  results/waste_predictions.csv   event_id,dish,pred_waste_pct,true_waste_pct,lighting
  results/waste_ablation.csv
  results/metrics.json            (merged)

Run
  python src/waste.py --dataset toy_dataset --limit 100 --ablation
  python src/waste.py --dataset dataset                    # all events, full method
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from align import align_pair, warp_mask
from common import class_masks, read_classes, read_yolo_seg, resize_max_side
from preprocess import SIZE, circle_mask, preprocess

YOLO_WEIGHTS = "models/best.pt"
CONFIGS = {  # name: (align, colour correction)
    "full": (True, True),
    "no_align": (False, True),
    "no_color": (True, False),
    "neither": (False, False),
}


# ================================================================ food mask
def food_mask_classical(img, plate_mask=None, min_thr=14, min_area=60, ref_mask=None,
                        l_weight=None):
    """Food = pixels whose Lab colour is far from the plate colour.
    Plate colour = median of ref_mask pixels (e.g. the empty steel right
    around the dishes) or, if not given, of the whole plate (after eating,
    most of the plate is empty). Threshold = max(min_thr, Otsu).
    Steel trays reflect light, so their brightness varies a lot: for them
    brightness (L) gets a lower weight than colour (a, b)."""
    if plate_mask is None:
        plate_mask = circle_mask(img.shape[0]) if img.shape[0] == img.shape[1] \
            else np.full(img.shape[:2], 255, np.uint8)
    inner = cv2.erode(plate_mask, np.ones((25, 25), np.uint8)) > 0   # skip the rim
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_px = lab[(ref_mask > 0) & inner] if ref_mask is not None else lab[inner]
    if len(ref_px) < 100:
        ref_px = lab[inner]
    ref = np.median(ref_px, axis=0)
    if l_weight is None:
        l_weight = 0.5 if img.shape[0] == img.shape[1] else 0.25
    d = lab - ref
    dist = np.sqrt((l_weight * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    dist[~inner] = 0
    d8 = np.clip(dist * 3, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8[inner], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(min_thr, otsu / 3)
    m = ((dist > thr) & inner).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # drop crumbs
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab_cc == i] = 1
    return keep * 255


def plate_ring(regions, shape, width=31):
    """Empty plate right around the dishes = best reference for plate colour."""
    union = np.zeros(shape[:2], np.uint8)
    for m in regions.values():
        union |= m.astype(np.uint8)
    ring = cv2.dilate(union, np.ones((width, width), np.uint8)) & (1 - union)
    return ring * 255


# ================================================================ one pair
def measure(before_img, after_img, before_masks, align=True, color_correct=True,
            seg="classical", after_masks_gt=None, pre_before=None):
    """before_masks: {class_id: mask in ORIGINAL before photo}.
    Returns ({class_id: waste %}, align result dict)."""
    r = align_pair(before_img, after_img, color_correct=color_correct, align=align,
                   pre_before=pre_before)
    shp = r["before"].shape
    regions = {c: warp_mask(m, r["H_before"], shp) > 0 for c, m in before_masks.items()}
    if seg == "gt":
        food = np.zeros(shp[:2], bool)
        for m in after_masks_gt.values():
            food |= warp_mask(m, r["H_after_total"], shp) > 0
    elif seg == "yolo":
        from segment import yolo_food_mask
        food = yolo_food_mask(r["after_warped"], weights=YOLO_WEIGHTS) > 0
    else:
        food = food_mask_classical(r["after_warped"], r["plate_mask"],
                                   ref_mask=plate_ring(regions, shp)) > 0
    out = {}
    for c, reg in regions.items():
        a = int(reg.sum())
        out[c] = 100.0 * int((food & reg).sum()) / a if a else np.nan
    return out, r


# ================================================================ dataset run
def run(dataset, configs=("full",), limit=None, seg="classical", max_side=1280,
        out="results", n_show=6, split=None, subset="val"):
    ds = Path(dataset)
    gen = ds / "generated"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    classes = read_classes(ds / "classes.txt")
    events = pd.read_csv(gen / "events.csv")
    done = {f.name for f in (gen / "after").glob("*.jpg")}
    if split:   # only events whose before photo is in the chosen split (e.g. YOLO val photos)
        keep = set(json.loads(Path(split).read_text())[subset])
        events = events[events.before_img.isin(keep)]
        print(f"  using only {subset} photos from {split}: {sorted(keep)}")
    ids = [e for e in events.event_id.unique() if f"{e}.jpg" in done]
    if limit:
        ids = ids[:limit]
    if not ids:
        raise SystemExit("No generated after images found - run generate_after.py first.")
    print(f"  {len(ids)} events, configs {list(configs)}, after-food mask: {seg}")

    cache, rows, shows = {}, [], []
    for k, eid in enumerate(ids, 1):
        g = events[events.event_id == eid]
        e = g.iloc[0]
        if e.before_img not in cache:
            img = resize_max_side(cv2.imread(str(ds / "before" / e.before_img)), max_side)
            objs = read_yolo_seg(ds / "labels" / (Path(e.before_img).stem + ".txt"))
            bm = class_masks(objs, *img.shape[:2])
            cache[e.before_img] = (img, bm, {cc: preprocess(img, color_correct=cc)
                                             for cc in (True, False)})
        bimg, bmasks, pres = cache[e.before_img]
        aimg = cv2.imread(str(gen / "after" / e.after_img))
        agt = None
        if seg == "gt":
            agt = class_masks(read_yolo_seg(gen / "after_labels" / (Path(e.after_img).stem + ".txt")),
                              *aimg.shape[:2])
        for name in configs:
            al, cc = CONFIGS[name]
            pred, r = measure(bimg, aimg, bmasks, al, cc, seg, agt, pre_before=pres[cc])
            for _, row in g.iterrows():
                cid = classes.index(row.dish) if row.dish in classes else -1
                rows.append({"config": name, "event_id": eid, "dish": row.dish,
                             "lighting": row.lighting, "true_waste_pct": row.true_waste_pct,
                             "pred_waste_pct": round(pred.get(cid, np.nan), 2),
                             "align_method": r["method"]})
            if name == "full" and len(shows) < n_show:
                fm = None
                if seg != "gt":
                    regs = {c: warp_mask(m, r["H_before"], r["before"].shape) > 0
                            for c, m in bmasks.items()}
                    fm = food_mask_classical(r["after_warped"], r["plate_mask"],
                                             ref_mask=plate_ring(regs, r["before"].shape))
                t = 256
                tiles = [cv2.resize(r["before"], (t, t)), cv2.resize(r["after_warped"], (t, t))]
                # (tiles are squashed to squares for the preview only)
                if fm is not None:
                    tiles.append(cv2.resize(cv2.cvtColor(fm, cv2.COLOR_GRAY2BGR), (t, t)))
                strip = np.hstack(tiles)
                txt = " ".join(f"{d}:{tv:.0f}/{pred.get(classes.index(d), np.nan):.0f}"
                               for d, tv in zip(g.dish, g.true_waste_pct) if d in classes)
                cv2.putText(strip, txt[:70], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 255), 1, cv2.LINE_AA)
                shows.append(strip)
        if k % 25 == 0 or k == len(ids):
            print(f"    {k}/{len(ids)}")

    df = pd.DataFrame(rows).dropna(subset=["pred_waste_pct"])
    df["abs_err"] = (df.pred_waste_pct - df.true_waste_pct).abs()

    full = df[df.config == configs[0]]
    full[["event_id", "dish", "pred_waste_pct", "true_waste_pct", "lighting"]].to_csv(
        out / "waste_predictions.csv", index=False)
    abl = df.pivot_table(index="config", columns="lighting", values="abs_err", aggfunc="mean")
    abl["ALL"] = df.groupby("config").abs_err.mean()
    abl = abl.reindex([c for c in CONFIGS if c in abl.index]).round(2)
    abl.to_csv(out / "waste_ablation.csv")
    if shows:
        cv2.imwrite(str(out / "waste_check.jpg"), np.vstack(shows))

    print("\n  WASTE MAE (percentage points), rows = config, columns = lighting")
    print(abl.to_string())
    bias = full.pred_waste_pct.mean() - full.true_waste_pct.mean()
    print(f"\n  {configs[0]}: bias {bias:+.2f} pp (positive = over-estimates waste)")
    print(f"  wrote {out / 'waste_predictions.csv'} (use with analytics.py --pred), "
          f"waste_ablation.csv, waste_check.jpg")

    mp = out / "metrics.json"
    allm = json.loads(mp.read_text()) if mp.exists() else {}
    for c in abl.index:
        allm[f"waste_mae_{c}_{seg}"] = float(abl.loc[c, "ALL"])
    mp.write_text(json.dumps(allm, indent=2))
    return df


def main():
    ap = argparse.ArgumentParser(description="Waste % per dish + ablation (B5)")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seg", default="classical", choices=["classical", "gt", "yolo"])
    ap.add_argument("--weights", default="models/best.pt", help="YOLO weights for --seg yolo")
    ap.add_argument("--split", default=None,
                    help="yolo_ds/split.json -> evaluate only on photos YOLO never saw")
    ap.add_argument("--subset", default="val")
    ap.add_argument("--ablation", action="store_true", help="run all 4 configs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    configs = tuple(CONFIGS) if args.ablation else ("full",)
    global YOLO_WEIGHTS
    YOLO_WEIGHTS = args.weights
    run(args.dataset, configs, args.limit, args.seg, out=args.out, split=args.split,
        subset=args.subset)


if __name__ == "__main__":
    main()
