"""
collect_results.py  (B8)  - gather every number and figure that already exists
into results/report/ for Person A (report + slides). Re-runs nothing heavy.

    python collect_results.py

Missing items are just listed as "not found" - run the step and collect again.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

R = Path("results")
REP = R / "report"
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# (source, name in report/figures, what it shows)
FIGS = [
    ("dataset/generated/preview.jpg", "01_generator_before_after.jpg",
     "Synthetic eating generator: real before photo | generated after photo"),
    ("results/align/align_check.jpg", "02_alignment_check.jpg",
     "Alignment: before | after | after aligned | overlay (grey = aligned)"),
    ("results/seg_examples.jpg", "03_segmentation_examples.jpg",
     "Segmentation on held-out photos: image | ground truth | classical | YOLO"),
    ("results/yolo/waste_check.jpg", "04_waste_check_yolo.jpg",
     "Waste measurement with YOLO: before | aligned after"),
    ("results/classical/waste_check.jpg", "05_waste_check_classical.jpg",
     "Waste measurement with the classical colour mask: before | aligned after | food mask"),
    ("results/analytics/heatmap_waste.png", "06_heatmap_waste_TRUE.png",
     "Mean waste % per dish x slot (true waste)"),
    ("results/analytics_measured/heatmap_waste.png", "07_heatmap_waste_MEASURED.png",
     "Mean waste % per dish x slot (YOLO-measured waste)"),
    ("results/analytics_measured/heatmap_twci.png", "08_heatmap_twci_MEASURED.png",
     "TWCI = mean waste x recurrence (measured)"),
    ("results/analytics_measured/drift.png", "09_drift_MEASURED.png",
     "Weekly mean waste per dish (measured) - rice drift"),
    ("results/analytics_measured/serving_decay.png", "10_serving_decay_MEASURED.png",
     "Waste vs minutes since service start (measured) - pasta decay"),
    ("results/analytics_measured/pairing.png", "11_pairing_MEASURED.png",
     "Pairing lift (measured) - salad with eggs"),
    ("results/analytics/robustness.png", "12_robustness_noise.png",
     "Pattern recovery vs added measurement noise"),
]
# YOLO training curves (newest run that exists)
for run in ("runs_kaggle", "runs/segment/runs_seg/food_more", "runs/segment/runs_seg/food", "runs_seg/food"):
    for f, n in (("results.png", "13_yolo_training_curves.png"),
                 ("confusion_matrix_normalized.png", "14_yolo_confusion_matrix.png"),
                 ("val_batch0_pred.jpg", "15_yolo_val_predictions.jpg")):
        if Path(run, f).exists() and not any(x[1] == n for x in FIGS):
            FIGS.append((f"{run}/{f}", n, f"YOLO training: {f}"))
# leftover (one-class food) model, unzipped training_results.zip
for f, n in (("results.png", "16_leftover_model_training_curves.png"),
             ("val_batch0_pred.jpg", "17_leftover_model_val_predictions.jpg")):
    if Path("runs_kaggle_food", f).exists():
        FIGS.append((f"runs_kaggle_food/{f}", n, f"Leftover model training: {f}"))

found, missing = [], []
for src, dst, desc in FIGS:
    if Path(src).exists():
        shutil.copy(src, FIG / dst)
        found.append((dst, desc))
    else:
        missing.append(src)

for src, dst in (("results/analytics/recommendations.md", "recommendations_TRUE.md"),
                 ("results/analytics_measured/recommendations.md", "recommendations_MEASURED.md"),
                 ("results/analytics_measured/quantity_plan.csv", "quantity_plan_MEASURED.csv"),
                 ("results/analytics_measured/detected_patterns.csv", "detected_patterns_MEASURED.csv")):
    if Path(src).exists():
        shutil.copy(src, REP / dst)


def md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def csv_block(path, title, cols=None):
    p = Path(path)
    if not p.exists():
        missing.append(str(p))
        return []
    d = pd.read_csv(p)
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    return [f"## {title}", "", md(d), ""]


def json_block(path, title):
    p = Path(path)
    if not p.exists():
        missing.append(str(p))
        return []
    return [f"## {title}", "", "```", p.read_text().strip(), "```", ""]


L = ["# Results summary (collected by collect_results.py)", "",
     "All waste numbers are in percentage points (pp). Held-out = the 4 photos YOLO never "
     "saw (yolo_ds/split.json) and their generated after images.", ""]
L += csv_block(R / "segmentation_eval.csv", "Segmentation, held-out photos (scored inside labelled dish areas)",
               ["method", "images", "n", "mIoU", "food_IoU"])
L += csv_block(R / "classical" / "waste_ablation.csv", "Waste MAE, classical food mask, held-out photos")
L += csv_block(R / "yolo" / "waste_ablation.csv", "Waste MAE, YOLO food mask, held-out photos")
L += csv_block(R / "yolo_ablation" / "waste_ablation.csv",
               "Ablation with YOLO, held-out photos (full / no_align / no_color / neither)")
L += csv_block(R / "waste_all_yolo" / "waste_ablation.csv",
               "Waste MAE, YOLO, all 615 plates (includes photos YOLO trained on - not the fair accuracy number)")
L += csv_block(R / "analytics" / "recovery.csv", "Pattern recovery - TRUE waste (upper bound)")
L += json_block(R / "analytics" / "metrics.json", "Analytics metrics - TRUE waste")
L += csv_block(R / "analytics_measured" / "recovery.csv", "Pattern recovery - YOLO-MEASURED waste (end-to-end headline)")
L += json_block(R / "analytics_measured" / "metrics.json", "Analytics metrics - MEASURED waste")
L += csv_block(R / "analytics" / "robustness.csv", "Pattern recovery vs added measurement noise (sd, pp)")
L += ["## Figures (report/figures/)", ""] + [f"- **{n}**: {d}" for n, d in found] + [""]
if missing:
    L += ["## Not found (run these steps, then collect again)", ""] + [f"- {m}" for m in missing]
(REP / "RESULTS.md").write_text("\n".join(L) + "\n")

print(f"Copied {len(found)} figures to {FIG}/ and wrote {REP / 'RESULTS.md'}")
if missing:
    print("Not found (optional - run the step and collect again):")
    for m in missing:
        print("  -", m)
