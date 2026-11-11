"""
app.py  (B7)  - Streamlit demo.

Run from the project root:
    pip install streamlit
    streamlit run app.py

Tab 1 "Measure a plate": pick a generated event OR upload a before + after
photo -> rectified images, alignment overlay, food mask, waste % per dish.
Tabs 2-4 read the analytics outputs (python src/analytics.py ... first):
heatmaps, detected patterns + recovery, recommendations + quantity plan.
Change the folders in the sidebar if you used --out.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from align import align_pair, overlay, warp_mask  # noqa: E402
from common import class_masks, read_classes, read_yolo_seg, resize_max_side  # noqa: E402
from waste import food_mask_classical, plate_ring  # noqa: E402

try:                                      # B4 - used automatically once it exists
    import segment as seg_mod             # noqa: E402
except Exception:
    seg_mod = None

st.set_page_config(page_title="Temporal Food Waste Analytics", layout="wide")
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
           (0, 128, 128)]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")
dataset = Path(st.sidebar.text_input("Dataset folder", "dataset"))
an_dir = Path(st.sidebar.text_input("Analytics output folder", "results/analytics"))
do_align = st.sidebar.checkbox("Alignment (SIFT/ECC)", True)
do_color = st.sidebar.checkbox("Colour correction (white balance + CLAHE)", True)
weights = st.sidebar.text_input("YOLO dish model (before photo)", "models/best.pt")
have_yolo = seg_mod is not None and Path(weights).exists()
_food_default = "models/food_best.pt" if Path("models/food_best.pt").exists() else weights
food_weights = st.sidebar.text_input("YOLO leftover model (after photo)", _food_default,
                                     help="one-class 'food' model trained with real canteen "
                                          "leftovers; falls back to the dish model")
detector = st.sidebar.radio("Leftover detector", ["YOLO", "Classical (colour)"] if have_yolo
                            else ["Classical (colour)"],
                            help="YOLO is used when the weights file exists")


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_upload(f):
    data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def unnamed_regions(img):
    """No labels and no model: food blobs on the before photo become
    'region 1', 'region 2', ... (classical colour segmentation)."""
    from preprocess import preprocess
    p = preprocess(img)
    fm = food_mask_classical(p["image"], p["plate_mask"])
    n, lab = cv2.connectedComponents(fm)
    Hinv = np.linalg.inv(p["H"])
    h, w = img.shape[:2]
    out = {}
    for i in range(1, n):
        m = (lab == i).astype(np.uint8) * 255
        if m.sum() / 255 > 300:
            out[len(out)] = cv2.warpPerspective(m, Hinv, (w, h), flags=cv2.INTER_NEAREST)
    return out, [f"region {i + 1}" for i in range(len(out))]


def color_regions(img, regions, labels):
    """Each region tinted in its own colour + its number written on it
    (same colour as its row in the table)."""
    vis = img.copy()
    h = img.shape[0]
    fs = max(0.6, h / 600)                       # font scale grows with image size
    th = max(2, int(h / 300))
    for k, (c, m) in enumerate(regions.items()):
        col = np.array(PALETTE[k % len(PALETTE)][::-1])
        vis[m] = (0.45 * vis[m] + 0.55 * col).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, col.tolist(), th)
    for k, (c, m) in enumerate(regions.items()):  # numbers on top of everything
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        # a point INSIDE the region (centroid can fall outside odd shapes)
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        txt = str(labels[k])
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        org = (int(x - tw / 2), int(y + tht / 2))
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return vis


def main_tray_only(cmasks, grow=0.06):
    """Keep only dish regions on the MAIN tray: food masks are merged and grown
    a little, the biggest connected group is the tray in front of the camera;
    regions mostly outside it (e.g. a neighbouring tray at the photo edge) are
    dropped."""
    if not cmasks:
        return cmasks
    m0 = next(iter(cmasks.values()))
    h, w = m0.shape[:2]
    union = np.zeros((h, w), np.uint8)
    for m in cmasks.values():
        union |= (m > 0).astype(np.uint8)
    k = max(3, int(grow * max(h, w))) | 1
    grown = cv2.dilate(union, np.ones((k, k), np.uint8))
    n, lab, st_, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    if n <= 2:
        return cmasks
    main = 1 + int(np.argmax(st_[1:, cv2.CC_STAT_AREA]))
    keep = {}
    for c, m in cmasks.items():
        mm = m > 0
        if mm.sum() and (lab[mm] == main).mean() >= 0.5:
            keep[c] = m
    return keep or cmasks


def hex_colour(k):
    r, g, b = PALETTE[k % len(PALETTE)]
    return f"#{r:02x}{g:02x}{b:02x}"


def kitchen_actions(df, slot):
    """'This plate' -> 'this slot over 4 weeks' -> 'what to cook next time'."""
    if slot is None:
        return
    wd, ml = slot
    st.subheader(f"Kitchen action for {wd} {ml}")
    plan_p, cells_p, det_p = an_dir / "quantity_plan.csv", an_dir / "cells.csv", an_dir / "detected_patterns.csv"
    if not plan_p.exists():
        st.info(f"Run analytics first ({plan_p} not found).")
        return
    plan = pd.read_csv(plan_p)
    cells = pd.read_csv(cells_p) if cells_p.exists() else pd.DataFrame()
    det = pd.read_csv(det_p) if det_p.exists() else pd.DataFrame()
    rows, lines = [], []
    for name, w in zip(df["dish"], df["waste % (measured)"]):
        p = plan[(plan.dish == name) & (plan.weekday == wd) & (plan.meal == ml)]
        c = cells[(cells.dish == name) & (cells.weekday == wd) & (cells.meal == ml)] if len(cells) else cells
        pats = []
        if len(det):
            m = det[(det.dish == name) & (det.pattern_type != "healthy") &
                    det.weekday.isin(["all", wd]) & det.meal.isin(["all", ml])]
            for r in m.itertuples():
                label = r.pattern_type.replace("_", " ")
                if isinstance(r.partner, str):
                    label += f" (with {r.partner})"
                pats.append(label)
                lines.append(f"- **{name}** - {label}: {r.evidence}")
        if p.empty:
            rows.append({"dish": name, "this plate %": w, "usual in this slot %": None,
                         "pattern": "not on this slot's menu", "next week: forecast %": None,
                         "cook": "-"})
            continue
        ch = int(p.change_pct.iloc[0])
        rows.append({"dish": name, "this plate %": w,
                     "usual in this slot %": round(float(c.mu.iloc[0]), 1) if len(c) else None,
                     "pattern": ", ".join(pats) if pats else "healthy",
                     "next week: forecast %": p.forecast_waste_pct.iloc[0],
                     "cook": "same amount" if ch > -5 else f"{-ch}% less"})
    out = pd.DataFrame(rows)
    def hl(row):
        return ["background-color: #ffd6d6; color: black" if isinstance(row["cook"], str)
                and row["cook"].endswith("less") else "" for _ in row.index]
    st.dataframe(out.style.apply(hl, axis=1).format(precision=1, na_rep="-"), hide_index=True)
    st.caption("'This plate' = measured from these two photos. 'Usual' and 'next week' come from "
               "4 weeks of plates (analytics). Quantity cut = keep eaten food the same, aim for "
               "10 % waste.")
    if lines:
        st.markdown("**Why:**\n" + "\n".join(lines))


tab1, tab2, tab3, tab4 = st.tabs(["Measure a plate", "Heatmaps", "Patterns", "Recommendations"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.header("Before / after → waste % per dish")
    src = st.radio("Input", ["Pick a generated event", "Upload two photos"], horizontal=True)
    before = after = None
    bmasks, names, truth = {}, [], {}
    slot = None                                   # (weekday, meal) of this plate
    classes = read_classes(dataset / "classes.txt") if (dataset / "classes.txt").exists() else []

    if src == "Pick a generated event":
        evp = dataset / "generated" / "events.csv"
        if not evp.exists():
            st.info(f"No {evp} yet. Run simulate_timeline.py and generate_after.py, "
                    "or upload two photos.")
        else:
            ev = pd.read_csv(evp)
            have = {f.name for f in (dataset / "generated" / "after").glob("*.jpg")}
            ids = [e for e in ev.event_id.unique() if f"{e}.jpg" in have]
            if ids:
                eid = st.selectbox("Event", ids)
                g = ev[ev.event_id == eid]
                e = g.iloc[0]
                st.caption(f"{e.date} · {e.weekday} {e.meal} · {e.minutes_since_start} min "
                           f"into service · lighting {e.lighting}")
                before = resize_max_side(cv2.imread(str(dataset / "before" / e.before_img)), 1280)
                after = cv2.imread(str(dataset / "generated" / "after" / e.after_img))
                objs = read_yolo_seg(dataset / "labels" / (Path(e.before_img).stem + ".txt"))
                bmasks = class_masks(objs, *before.shape[:2])
                names = classes
                truth = dict(zip(g.dish, g.true_waste_pct))
                slot = (e.weekday, e.meal)
            else:
                st.info("No generated after images found.")
    else:
        c1, c2 = st.columns(2)
        fb = c1.file_uploader("Before photo", type=["jpg", "jpeg", "png"])
        fa = c2.file_uploader("After photo", type=["jpg", "jpeg", "png"])
        d1, d2 = st.columns(2)
        slot = (d1.selectbox("Day this meal was served",
                             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
                d2.selectbox("Meal", ["breakfast", "lunch", "dinner"], index=1))
        if fb and fa:
            before = resize_max_side(read_upload(fb), 1280)
            after = resize_max_side(read_upload(fa), 1280)
            if have_yolo:
                bmasks = main_tray_only(seg_mod.predict_class_masks(before, weights))
                names = classes
                st.caption("Dish regions from segment.py")
            else:
                bmasks, names = unnamed_regions(before)
                st.caption("No segmentation model yet: food blobs shown as unnamed regions.")

    if before is not None and after is not None and bmasks:
        with st.spinner("Rectifying, aligning, measuring..."):
            r = align_pair(before, after, color_correct=do_color, align=do_align)
            regions = {c: warp_mask(m, r["H_before"], r["before"].shape) > 0
                       for c, m in bmasks.items()}
            if detector == "YOLO":
                fw = food_weights if Path(food_weights).exists() else weights
                fm = seg_mod.yolo_food_mask(r["after_warped"], weights=fw) > 0
            else:
                fm = food_mask_classical(r["after_warped"], r["plate_mask"],
                                         ref_mask=plate_ring(regions, r["before"].shape)) > 0
            # waste is only measured inside the dish regions -> only show red there
            near = np.zeros(r["before"].shape[:2], np.uint8)
            for reg in regions.values():
                near |= reg.astype(np.uint8)
            near = cv2.dilate(near, np.ones((25, 25), np.uint8)) > 0
            fm_show = fm & near
        failed = r["method"] in ("none", "plate") and r["shape"] == "tray"
        if failed:
            st.error("These two photos could not be aligned, so they are probably NOT the same "
                     "tray (matching points must cover the whole tray, not one cup or logo). "
                     "No waste is measured. Upload a before and after photo of the SAME tray, "
                     "or pick a generated event.")
            c1, c2 = st.columns(2)
            c1.image(rgb(r["before"]), caption="Before")
            c2.image(rgb(r["after_rect"]), caption="After")
        else:
            c2, c3, c4 = st.columns(3)
            c2.image(rgb(r["after_rect"]), caption=f"After ({'rectified' if r['shape'] == 'round' else 'tray, resized'})")
            c3.image(rgb(r["after_warped"]), caption=f"After aligned ({r['method']}, {r['inliers']} inliers)")
            c4.image(rgb(overlay(r["before"], r["after_warped"])), caption="Overlay: grey = aligned")

            fixed = {}
            if src == "Upload two photos" and names:
                IGN = "(not on this tray - ignore)"
                with st.expander("Fix dish names (the model can name a dish wrongly)", expanded=True):
                    st.caption("Regions are numbered as in the picture below. Correct a wrong name, "
                               "or ignore a region that is not on this tray.")
                    cols = st.columns(min(4, max(1, len(regions))))
                    for k, c in enumerate(regions):
                        cur = names[c] if c < len(names) else names[0]
                        fixed[c] = cols[k % len(cols)].selectbox(
                            f"Region {k + 1}", list(names) + [IGN],
                            index=list(names).index(cur) if cur in names else 0, key=f"fix_{c}")
                regions = {c: reg for c, reg in regions.items() if fixed.get(c) != IGN}
                if not regions:
                    st.warning("All regions ignored - nothing to measure.")
            rows = []
            for k, (c, reg) in enumerate(regions.items()):
                a = int(reg.sum())
                name = fixed.get(c) or (names[c] if c < len(names) else f"class {c}")
                pred = 100 * int((fm & reg).sum()) / a if a else np.nan
                row = {"#": k + 1, "dish": name, "waste % (measured)": round(pred, 1)}
                if truth:
                    row["waste % (true)"] = truth.get(name, np.nan)
                rows.append(row)
            df = pd.DataFrame(rows)

            vis = r["after_warped"].copy()
            vis[fm_show] = (0.4 * vis[fm_show] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            for reg in regions.values():                       # thin outline of each dish region
                cnts, _ = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (255, 255, 255), 2)
            e1, e2 = st.columns(2)
            e1.image(rgb(color_regions(r["before"], regions, list(df["#"]))),
                     caption=f"Before ({r['shape']}): numbered dish regions - colours match the table")
            e2.image(rgb(vis), caption=f"Leftover food found by {detector} (red), inside the dish regions (white outlines)")

            def paint(row):
                col = hex_colour(int(row["#"]) - 1)
                return [f"background-color: {col}; color: black; font-weight: bold"
                        if c in ("#", "dish") else "" for c in row.index]
            st.dataframe(df.style.apply(paint, axis=1).format(precision=1), hide_index=True)
            kitchen_actions(df, slot)

# ------------------------------------------------------------------ tabs 2-4
def need(path):
    if not path.exists():
        st.info(f"{path} not found. Run: python src/analytics.py --dataset {dataset} "
                f"--out {an_dir}")
        return False
    return True


with tab2:
    st.header("Where and when food is wasted")
    for f, cap in [("heatmap_waste.png", "Mean waste % per dish × slot"),
                   ("heatmap_twci.png", "TWCI = mean waste × recurrence"),
                   ("drift.png", "Weekly trend"), ("serving_decay.png", "Serving-time decay"),
                   ("pairing.png", "Pairing lift"), ("robustness.png", "Recovery vs noise")]:
        if (an_dir / f).exists():
            st.image(str(an_dir / f), caption=cap)
    need(an_dir / "heatmap_waste.png")

with tab3:
    st.header("Detected patterns")
    if need(an_dir / "detected_patterns.csv"):
        det = pd.read_csv(an_dir / "detected_patterns.csv")
        st.dataframe(det[["pattern_type", "dish", "partner", "weekday", "meal", "evidence"]],
                     hide_index=True)
        if (an_dir / "recovery.csv").exists():
            rec = pd.read_csv(an_dir / "recovery.csv")
            st.subheader(f"Recovered {int(rec.recovered.sum())} of {len(rec)} planted patterns")
            st.dataframe(rec, hide_index=True)
        if (an_dir / "cells.csv").exists():
            with st.expander("All cells (TWCI table)"):
                st.dataframe(pd.read_csv(an_dir / "cells.csv").round(2), hide_index=True)

with tab4:
    st.header("Menu actions")
    if need(an_dir / "recommendations.md"):
        st.markdown((an_dir / "recommendations.md").read_text())
    if (an_dir / "quantity_plan.csv").exists():
        with st.expander("Full quantity plan for next week"):
            st.dataframe(pd.read_csv(an_dir / "quantity_plan.csv"), hide_index=True)
