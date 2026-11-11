"""
preprocess.py  (B1)  - plate detection, rectification, lighting normalisation.

Pipeline for one image (each step can be switched off for the ablation):
  1. detect the plate   : HoughCircles; if the plate looks like an ellipse
                          (camera not straight above) use an ellipse fit on
                          the largest contour instead.
  2. rectify            : homography that maps the plate ellipse to a
                          top-down circle in a fixed SIZE x SIZE image.
  3. denoise            : bilateral (default) or median filter.
  4. glare removal      : glare = very bright (high V) + almost no colour
                          (low S); those pixels are filled with cv2.inpaint.
  5. colour constancy   : white balance on the neutral (grey) plate/steel pixels,
                          gains capped at 1.25 (N3). Falls back to capped gray-world.
  6. CLAHE              : mild contrast equalisation on the L channel (N3).

Round plates are rectified; rectangular steel TRAYS (rim not a circle,
detected automatically, or set env PLATE_SHAPE=tray) are only resized -
align.py then maps the after photo directly onto the before photo.

Use from code:
    from preprocess import preprocess
    out = preprocess(img_bgr)            # dict
    out["image"], out["H"], out["plate_mask"], out["glare_mask"], out["plate"]

Command line (check the plate detection on a folder):
    python src/preprocess.py --input dataset/before --out results/preprocessed
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

SIZE = 768          # output image is SIZE x SIZE
PLATE_FILL = 0.92   # plate diameter = 92% of SIZE (leaves a small border)


# =========================================================== 1. plate detection
def _hough_circle(gray):
    """Strongest big circle, or None. Returns (cx, cy, r) in gray coords."""
    h, w = gray.shape
    m = min(h, w)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=m / 2, param1=100, param2=40,
                               minRadius=int(0.25 * m), maxRadius=int(0.6 * m))
    if circles is None:
        return None
    # take the largest of the 3 strongest circles (the plate rim, not a bowl)
    best = max(circles[0][:3], key=lambda c: c[2])
    return float(best[0]), float(best[1]), float(best[2])


def _ellipse_fit(gray):
    """Ellipse fit on the largest closed contour, or None.
    Returns cv2 ellipse ((cx, cy), (d1, d2), angle)."""
    h, w = gray.shape
    img_area = h * w
    cands = []
    # (a) edges -> closed contours ; (b) Otsu bright region (plates are light)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8),
                             iterations=2)
    _, otsu = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    for bw in (edges, otsu):
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if len(c) < 20:
                continue
            hull = cv2.convexHull(c)
            if len(hull) < 5:
                continue
            e = cv2.fitEllipse(hull)
            (cx, cy), (d1, d2), _ = e
            area = np.pi * d1 * d2 / 4
            if not (0.12 * img_area < area < 1.05 * img_area):
                continue
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            # how well does the contour fill the ellipse? (1 = perfect)
            fill = cv2.contourArea(hull) / max(area, 1)
            cands.append((fill * area, e))
    if not cands:
        return None
    return max(cands, key=lambda t: t[0])[1]


def _ring_hits(edges, plate, scale, n=180):
    (cx, cy), (a, b), ang = plate["center"], plate["axes"], plate["angle"]
    a, b = a * scale, b * scale
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    th = np.deg2rad(ang)
    x = cx + a / 2 * np.cos(t) * np.cos(th) - b / 2 * np.sin(t) * np.sin(th)
    y = cy + a / 2 * np.cos(t) * np.sin(th) + b / 2 * np.sin(t) * np.cos(th)
    h, w = edges.shape
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if inside.mean() < 0.8:                 # runs out of the photo
        return None
    return float(edges[y[inside].astype(int), x[inside].astype(int)].mean())


def rim_support(gray, plate):
    """How strongly the fitted ellipse is a real rim:
        edge hits ON the ellipse  minus  edge hits on rings 12 % smaller/larger.
    A real round plate has a continuous rim and calm surroundings (~0.5+).
    A circle forced onto a tray, or onto busy brushed steel where edges are
    everywhere, scores low."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
    on = _ring_hits(edges, plate, 1.0)
    if on is None:
        return 0.0
    off = [v for v in (_ring_hits(edges, plate, 0.88), _ring_hits(edges, plate, 1.12))
           if v is not None]
    return float(on - (max(off) if off else 0.0))


ROUND_MIN_SUPPORT = 0.35   # rim edges minus background edges
ROUND_MIN_SIZE = 0.6     # plate diameter / shorter photo side


def plate_shape(plate):
    """'round' or 'tray'. Override with environment variable PLATE_SHAPE."""
    forced = os.environ.get("PLATE_SHAPE", "auto").lower()
    if forced in ("round", "tray"):
        return forced
    big = plate.get("rel_size", 1.0) >= ROUND_MIN_SIZE      # a katori bowl is not the plate
    return "round" if plate.get("support", 0) >= ROUND_MIN_SUPPORT and big else "tray"


def detect_plate(img_bgr, work_side=640):
    """Find the plate. Returns dict (coordinates in the ORIGINAL image):
        {"kind": "circle"|"ellipse"|"fallback",
         "center": (cx, cy), "axes": (major_diam, minor_diam), "angle": deg}
    """
    h, w = img_bgr.shape[:2]
    s = min(1.0, work_side / max(h, w))
    small = cv2.resize(img_bgr, (int(w * s), int(h * s))) if s < 1 else img_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    circ = _hough_circle(gray)
    ell = _ellipse_fit(gray)

    result = None
    if ell is not None:
        (cx, cy), (d1, d2), ang = ell
        ratio = min(d1, d2) / max(d1, d2)
        near = circ is None or np.hypot(cx - circ[0], cy - circ[1]) < 0.25 * max(d1, d2)
        # use the ellipse when the plate is clearly squashed (tilted camera)
        # or when Hough found nothing
        if circ is None or (ratio < 0.9 and near):
            if d1 >= d2:
                major, minor, angle = d1, d2, ang
            else:
                major, minor, angle = d2, d1, ang + 90
            result = {"kind": "ellipse", "center": (cx, cy),
                      "axes": (major, minor), "angle": angle}
    if result is None and circ is not None:
        cx, cy, r = circ
        result = {"kind": "circle", "center": (cx, cy),
                  "axes": (2 * r, 2 * r), "angle": 0.0}
    if result is None:                          # last resort: centred circle
        d = 0.9 * min(gray.shape)
        result = {"kind": "fallback", "center": (gray.shape[1] / 2, gray.shape[0] / 2),
                  "axes": (d, d), "angle": 0.0}
    result["support"] = rim_support(gray, result) if result["kind"] != "fallback" else 0.0
    result["rel_size"] = float(max(result["axes"]) / min(gray.shape))

    # back to original resolution
    cx, cy = result["center"]
    result["center"] = (cx / s, cy / s)
    result["axes"] = (result["axes"][0] / s, result["axes"][1] / s)
    return result


def plate_mask_original(shape, plate):
    """Filled plate ellipse in original image coordinates (0/255)."""
    m = np.zeros(shape[:2], np.uint8)
    (cx, cy), (a, b) = plate["center"], plate["axes"]
    cv2.ellipse(m, ((cx, cy), (a, b), plate["angle"]), 255, -1)
    return m


# =========================================================== 2. rectification
def rectify_homography(plate, size=SIZE, fill=PLATE_FILL):
    """Homography H (3x3) mapping the plate ellipse -> circle of diameter
    fill*size centred in a size x size image. Uses 4 point pairs: the two
    ends of the major axis and the two ends of the minor axis."""
    (cx, cy), (a, b), ang = plate["center"], plate["axes"], plate["angle"]
    t = np.deg2rad(ang)
    u = np.array([np.cos(t), np.sin(t)])        # major-axis direction
    v = np.array([-np.sin(t), np.cos(t)])       # minor-axis direction
    c = np.array([cx, cy])
    src = np.float32([c + u * a / 2, c - u * a / 2, c + v * b / 2, c - v * b / 2])
    R = fill * size / 2
    C = np.array([size / 2, size / 2])
    dst = np.float32([C + u * R, C - u * R, C + v * R, C - v * R])
    return cv2.getPerspectiveTransform(src, dst)


def warp_polygons(objs, H, src_hw, size=SIZE):
    """Move YOLO polygons (normalised in the source image) into the
    rectified image (normalised again). objs = [(cid, poly), ...]."""
    h, w = src_hw
    out = []
    for cid, poly in objs:
        pts = (np.asarray(poly, np.float32) * [w, h]).reshape(-1, 1, 2)
        pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2) / size
        out.append((cid, np.clip(pts, 0, 1)))
    return out


def circle_mask(size=SIZE, fill=PLATE_FILL):
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), int(fill * size / 2), 255, -1)
    return m


# =========================================================== 3-6. filters
def denoise(img, method="bilateral"):
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=7)
    if method == "median":
        return cv2.medianBlur(img, 5)
    return img


def glare_mask(img, v_min=225, s_max=40, delta=18, mask=None, max_frac=0.08):
    """Specular glare = very bright + nearly colourless + clearly brighter
    than its neighbourhood (so a plain white plate is NOT glare)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    k = int(max(img.shape[:2]) * 0.08) | 1
    local = cv2.medianBlur(hsv[..., 2], min(k, 255)).astype(np.float32)
    g = (v >= v_min) & (hsv[..., 1] <= s_max) & (v - local >= delta)
    g = g.astype(np.uint8) * 255
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    g = cv2.dilate(g, np.ones((5, 5), np.uint8))
    if mask is not None:
        g &= mask
    # safety: glare is small; if too much is flagged keep only saturated white
    area = (mask > 0).sum() if mask is not None else g.size
    if (g > 0).sum() > max_frac * area:
        g &= ((hsv[..., 2] >= 252).astype(np.uint8) * 255)
    return g


def remove_glare(img, gmask):
    if gmask.any():
        return cv2.inpaint(img, gmask, 5, cv2.INPAINT_TELEA)
    return img


def gray_world(img, mask=None, max_gain=1.25):
    """Scale B, G, R so their means are equal (computed inside mask).
    Gains are capped so a colourful plate does not get over-corrected."""
    f = img.astype(np.float32)
    px = f[mask > 0] if mask is not None and mask.any() else f.reshape(-1, 3)
    means = px.mean(axis=0) + 1e-6
    gain = np.clip(means.mean() / means, 1 / max_gain, max_gain)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def neutral_white_balance(img, mask=None, max_gain=1.25, min_px=500):
    """White balance using the NEUTRAL (grey) pixels as reference - on a
    steel tray / white plate those are the plate itself, so its colour cast
    = the lighting cast. Uses the least-saturated 25 % of mid-bright pixels.
    Falls back to capped gray-world if there are too few."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ok = (hsv[..., 2] > 50) & (hsv[..., 2] < 245)
    if mask is not None:
        ok &= mask > 0
    if ok.sum() < min_px:
        return gray_world(img, mask, max_gain)
    s = hsv[..., 1][ok]
    neutral = ok & (hsv[..., 1] <= np.percentile(s, 25))
    if neutral.sum() < min_px:
        return gray_world(img, mask, max_gain)
    means = img[neutral].astype(np.float32).mean(axis=0) + 1e-6
    gain = np.clip(means.mean() / means, 1 / max_gain, max_gain)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def clahe_l(img, clip=1.5, tiles=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# =========================================================== full pipeline
TRAY_SIDE = 1024    # tray photos are resized so the long side is this


def preprocess(img_bgr, size=SIZE, rectify=True, denoise_method="bilateral",
               fix_glare=True, color_correct=True, plate=None, shape=None):
    """Run the whole B1 pipeline. Returns a dict:
        image       : processed BGR image
                      round plate -> size x size, plate = centred circle
                      tray        -> original photo resized (long side 1024),
                                     NOT rectified: align.py maps the after
                                     photo straight onto the before photo
        H           : 3x3 homography original photo -> output image
        shape       : "round" or "tray" (auto, or env PLATE_SHAPE)
        plate       : detected plate dict (original coords)
        plate_mask  : plate region in the output image (whole image for trays)
        glare_mask  : pixels that were inpainted
    """
    if plate is None:
        plate = detect_plate(img_bgr)
    if shape is None:
        shape = plate_shape(plate)
    if rectify and shape == "round":
        H = rectify_homography(plate, size)
        img = cv2.warpPerspective(img_bgr, H, (size, size),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        pmask = circle_mask(size)
    elif shape == "tray":
        h, w = img_bgr.shape[:2]
        s = min(1.0, TRAY_SIDE / max(h, w))
        H = np.diag([s, s, 1.0])
        img = cv2.resize(img_bgr, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA) if s < 1 else img_bgr.copy()
        pmask = np.full(img.shape[:2], 255, np.uint8)
    else:
        H = np.eye(3)
        img = img_bgr.copy()
        pmask = plate_mask_original(img.shape, plate)

    img = denoise(img, denoise_method)
    gmask = glare_mask(img, mask=pmask) if fix_glare else np.zeros(pmask.shape, np.uint8)
    if fix_glare:
        img = remove_glare(img, gmask)
    if color_correct:
        img = neutral_white_balance(img, pmask)
        img = clahe_l(img)
    return {"image": img, "H": H, "plate": plate, "shape": shape,
            "plate_mask": pmask, "glare_mask": gmask}


def draw_plate(img, plate, color=(0, 255, 0)):
    out = img.copy()
    (cx, cy), (a, b) = plate["center"], plate["axes"]
    t = max(2, int(max(img.shape[:2]) / 300))
    cv2.ellipse(out, ((cx, cy), (a, b), plate["angle"]), color, t)
    cv2.putText(out, plate["kind"], (10, 30 * t), cv2.FONT_HERSHEY_SIMPLEX,
                t * 0.6, color, t)
    return out


# =========================================================== CLI
def main():
    ap = argparse.ArgumentParser(description="Plate detection + rectification")
    ap.add_argument("--input", default="dataset/before")
    ap.add_argument("--out", default="results/preprocessed")
    ap.add_argument("--size", type=int, default=SIZE)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in Path(args.input).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    kinds = {}
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print("  could not read", p.name)
            continue
        r = preprocess(img, size=args.size)
        k = r["shape"] if r["shape"] == "tray" else r["plate"]["kind"]
        kinds[k] = kinds.get(k, 0) + 1
        # side-by-side: original with detected plate | rectified result
        left = draw_plate(img, r["plate"]) if r["shape"] == "round" else img.copy()
        hh = r["image"].shape[0]
        left = cv2.resize(left, (int(left.shape[1] * hh / left.shape[0]), hh))
        cv2.imwrite(str(out / f"{p.stem}_check.jpg"), np.hstack([left, r["image"]]))
        cv2.imwrite(str(out / f"{p.stem}_rect.jpg"), r["image"])
        print(f"  {p.name}: shape={r['shape']} (rim support {r['plate']['support']:.2f}), glare px={int((r['glare_mask'] > 0).sum())}")
    print(f"Done: {len(files)} images -> {out}   plate kinds: {kinds}")


if __name__ == "__main__":
    main()
