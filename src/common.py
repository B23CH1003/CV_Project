"""
common.py - small helpers shared by all modules.

- reading classes.txt
- reading / writing YOLO segmentation label files
- turning polygons into masks and masks back into polygons
- cleaning weekday / meal / dish names so menu.csv and patterns.yaml match

YOLO-seg label format (one object per line, coordinates normalised to 0..1):
    <class_id> x1 y1 x2 y2 x3 y3 ...
"""

from pathlib import Path

import cv2
import numpy as np

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
MEAL_ORDER = ["breakfast", "lunch", "snacks", "dinner"]


# ---------------------------------------------------------------- names
def clean_dish(name):
    """'  Rajma ' -> 'rajma'."""
    return str(name).strip().lower()


def clean_meal(name):
    m = str(name).strip().lower()
    if m in ("snack", "evening snacks", "tea"):
        m = "snacks"
    return m


def clean_weekday(name):
    """Accepts 'mon', 'Monday', 'MONDAY' ... returns 'Monday'.
    'all', '*' or empty -> 'all'."""
    s = str(name).strip().lower()
    if s in ("", "all", "*", "any", "none", "nan"):
        return "all"
    for d in WEEKDAYS:
        if d.lower().startswith(s[:3]):
            return d
    raise ValueError(f"Unknown weekday: {name!r}")


# ---------------------------------------------------------------- classes
def read_classes(path):
    """classes.txt -> list of class names (index = YOLO class id)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [clean_dish(l) for l in lines if l.strip()]


# ---------------------------------------------------------------- labels
def read_yolo_seg(path):
    """Return list of (class_id, polygon) where polygon is an (N,2) float
    array of normalised x,y. Lines with fewer than 3 points are skipped.
    (A plain YOLO *box* line 'c cx cy w h' is turned into a rectangle.)"""
    objs = []
    p = Path(path)
    if not p.exists():
        return objs
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        vals = np.array(parts[1:], dtype=np.float32)
        if len(vals) == 4:                      # bounding box fallback
            cx, cy, w, h = vals
            poly = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                             [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
        else:
            if len(vals) % 2 == 1:
                vals = vals[:-1]
            poly = vals.reshape(-1, 2)
        if len(poly) >= 3:
            objs.append((cid, np.clip(poly, 0, 1)))
    return objs


def write_yolo_seg(path, objs):
    """objs: list of (class_id, polygon normalised (N,2))."""
    lines = []
    for cid, poly in objs:
        if len(poly) < 3:
            continue
        coords = " ".join(f"{v:.6f}" for v in np.asarray(poly).reshape(-1))
        lines.append(f"{int(cid)} {coords}")
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def label_classes(path):
    """Set of class ids that appear in a label file."""
    return {cid for cid, _ in read_yolo_seg(path)}


# ---------------------------------------------------------------- masks
def polygon_to_mask(poly_norm, h, w):
    """Normalised polygon -> uint8 mask (0/255) of size h x w."""
    pts = np.round(np.asarray(poly_norm) * [w, h]).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def instance_masks(objs, h, w):
    """Turn polygons into NON-overlapping instance masks.
    Big objects are painted first, small ones on top, so a small dish
    drawn over a big one keeps its pixels.
    Returns list of (class_id, mask)."""
    raw = [(cid, polygon_to_mask(p, h, w)) for cid, p in objs]
    order = sorted(range(len(raw)), key=lambda i: -int((raw[i][1] > 0).sum()))
    label_map = np.zeros((h, w), np.int32)        # 0 = no food
    for k, i in enumerate(order, start=1):
        label_map[raw[i][1] > 0] = k
    out = []
    for k, i in enumerate(order, start=1):
        m = (label_map == k).astype(np.uint8) * 255
        if m.any():
            out.append((raw[i][0], m))
    return out


def class_masks(objs, h, w):
    """dict class_id -> union mask (0/255)."""
    out = {}
    for cid, m in instance_masks(objs, h, w):
        out[cid] = np.maximum(out.get(cid, np.zeros_like(m)), m)
    return out


def mask_to_polygons(mask, min_area=30, eps_frac=0.002):
    """Binary mask -> list of normalised polygons (one per outer contour)."""
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = eps_frac * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        if len(c) >= 3:
            polys.append(c / [w, h])
    return polys


def resize_max_side(img, max_side):
    """Shrink so the longest side is <= max_side (never enlarges)."""
    if not max_side:
        return img
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)
