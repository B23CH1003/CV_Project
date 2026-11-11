"""
generate_after.py  (B2, part 2)  - synthetic "eating" generator.

For every plate event in dataset/generated/events.csv it takes the real
before photo + its YOLO-seg labels and makes a fake "after" photo:

  1. for each food region (instance) keep ONE connected blob whose area is
     exactly true_waste_pct % of the region - that is the leftover.
     The rest is "eaten".
  2. eaten pixels are filled with the plate colour (sampled from the empty
     plate) + a little texture; seams are blended with cv2.inpaint.
  3. realistic residue on the eaten area (NOT labelled, NOT counted as waste):
     wet dishes leave a streaky gravy film + a ring on the bowl wall, dry
     dishes leave scattered grains/crumbs copied from the dish itself.
  4. small camera change: rotation +-12 deg, scale, shift and a slight
     perspective tilt (one homography for image AND masks).
  5. small brightness change only (food colours stay as in the before photo),
     rare faint glare, sensor noise and JPEG compression.

Writes
    dataset/generated/after/<event_id>.jpg
    dataset/generated/after_labels/<event_id>.txt      (YOLO-seg, leftovers only)
    dataset/generated/generation_log.csv               (realised vs planned waste)
    dataset/generated/preview.jpg                      (first N events, for spot-checks)

Run
    python src/generate_after.py --dataset dataset
    python src/generate_after.py --dataset dataset --limit 20     # quick test
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import (instance_masks, mask_to_polygons, read_classes,
                    read_yolo_seg, resize_max_side, write_yolo_seg)
from preprocess import detect_plate, plate_mask_original, plate_shape

# lighting: BRIGHTNESS ONLY (food colours must stay as in the before photo).
# A small, realistic difference between the two shots, nothing more.
LIGHTING = {"daylight": 1.00, "evening": 0.94, "warm_bright": 1.00,
            "neutral": 0.98, "cool_dim": 0.94}

# dishes that leave a gravy film when eaten vs dishes that leave grains/crumbs
WET = {"soup", "beans", "dal", "sabzi", "chole", "curd", "sweet", "rajma", "kadhi", "kheer"}


# ================================================================ 1. leftover blob
def smooth_noise(rng, h, w, scale=0.08):
    """Low-frequency random field in [0, 1] (makes blob borders irregular)."""
    sh, sw = max(2, int(h * scale)), max(2, int(w * scale))
    n = rng.random((sh, sw)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), max(h, w) * 0.03)
    return (n - n.min()) / (n.max() - n.min() + 1e-6)


def leftover_blob(mask, frac, rng):
    """Inside a binary mask, pick ONE connected region with exactly
    round(frac * area) pixels. Grows from a random seed, ordered by
    distance-from-seed + smooth noise, so the shape is a natural blob."""
    m = mask > 0
    area = int(m.sum())
    target = int(round(np.clip(frac, 0, 1) * area))
    keep = np.zeros_like(m)
    if target <= 0 or area == 0:
        return keep.astype(np.uint8) * 255
    if target >= area:
        return m.astype(np.uint8) * 255

    ys, xs = np.nonzero(m)
    x0, y0, bw, bh = cv2.boundingRect(m.astype(np.uint8))
    crop = m[y0:y0 + bh, x0:x0 + bw]
    # seed: random food pixel, biased towards the edge (people push leftovers aside)
    dist_in = cv2.distanceTransform(crop.astype(np.uint8), cv2.DIST_L2, 3)
    cy, cx = np.nonzero(crop)
    edge_w = 1.0 / (1.0 + dist_in[cy, cx])
    i = rng.choice(len(cy), p=edge_w / edge_w.sum())
    sy, sx = cy[i], cx[i]

    yy, xx = np.mgrid[0:bh, 0:bw]
    d = np.hypot(yy - sy, xx - sx).astype(np.float32)
    d /= d[crop].max() + 1e-6
    field = d + 0.6 * smooth_noise(rng, bh, bw)
    field[~crop] = np.inf

    # first guess: the `target` lowest-field pixels
    flat = field.ravel()
    order = np.argsort(flat, kind="stable")[:target]
    k = np.zeros(bh * bw, bool)
    k[order] = True
    k = k.reshape(bh, bw)
    # keep only the biggest connected piece ...
    n, lab, stats, _ = cv2.connectedComponentsWithStats(k.astype(np.uint8), connectivity=8)
    if n > 2:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        k = lab == big
    # ... then grow it back to the exact target, ring by ring
    kernel = np.ones((3, 3), np.uint8)
    while k.sum() < target:
        ring = (cv2.dilate(k.astype(np.uint8), kernel) > 0) & crop & ~k
        if not ring.any():
            break
        need = target - int(k.sum())
        ry, rx = np.nonzero(ring)
        if len(ry) > need:
            sel = np.argsort(field[ry, rx], kind="stable")[:need]
            ry, rx = ry[sel], rx[sel]
        k[ry, rx] = True
    keep[y0:y0 + bh, x0:x0 + bw] = k
    return keep.astype(np.uint8) * 255


# ================================================================ 2-3. fill + smear
def plate_colour_pixels(img, food_union, plate):
    """BGR pixels of empty plate (inside plate, away from food)."""
    food_d = cv2.dilate(food_union, np.ones((15, 15), np.uint8)) > 0
    if plate_shape(plate) == "tray":
        # steel tray: sample the empty steel in a band around the food
        k = max(31, int(0.04 * max(img.shape[:2])) | 1)
        pm = cv2.dilate(food_union, np.ones((k, k), np.uint8)) > 0
    else:
        pm = plate_mask_original(img.shape, plate) > 0
        # stay off the rim/shadow: shrink the plate mask a bit
        pm = cv2.erode(pm.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    px = img[pm & ~food_d]
    if len(px) < 200:                       # plate fully covered: use all non-food
        px = img[~food_d]
    if len(px) < 50:
        px = np.full((50, 3), 230, np.uint8)
    return px


def surface_mask(img, food_union, region=None):
    """Pixels that look like the EMPTY plate/tray the food sits on.
    For each group of food, take a ring of pixels just around it (that is
    the plate), find its typical colour (median in Lab) and accept every
    pixel of similar colour. Works for steel, white, coloured and patterned
    plates; table-cloth and unlabelled food further away are rejected."""
    h, w = food_union.shape
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = max(15, int(0.05 * max(h, w)))
    inner = cv2.dilate(food_union, np.ones((9, 9), np.uint8))
    groups = cv2.dilate(food_union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    n, comp = cv2.connectedComponents((groups > 0).astype(np.uint8))
    valid = np.zeros((h, w), bool)
    for k in range(1, n):
        ring = (comp == k) & (inner == 0)
        if region is not None and (region & (inner == 0) & (comp == k)).sum() >= 50:
            # known plate/bowl outline: use ALL its visible pixels near this food
            ring = region & (inner == 0) & (comp == k)
        if ring.sum() < 50:
            continue
        # dominant colour of the ring = the plate (k-means, largest cluster);
        # food edges / table-cloth in the ring form the smaller clusters
        px = lab[ring]
        if len(px) > 4000:
            px = px[np.random.default_rng(0).choice(len(px), 4000, replace=False)]
        cv2.setRNGSeed(0)                           # same result every run
        _, lbl, ctr = cv2.kmeans(px, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
        # plates are mostly neutral (white/grey/steel): strongly coloured
        # clusters (food edges, unlabelled food) count half
        chroma = np.hypot(ctr[:, 1] - 128, ctr[:, 2] - 128)
        score = np.bincount(lbl.ravel(), minlength=3) * np.where(chroma < 22, 1.0, 0.5)
        big = int(score.argmax())
        med = ctr[big]
        dc = np.linalg.norm(px[lbl.ravel() == big] - med, axis=1)
        thr = float(np.clip(np.percentile(dc, 90), 10.0, 26.0))
        d = np.linalg.norm(lab - med, axis=2)
        valid |= (d <= thr) & (groups > 0) if region is None else (d <= thr)
    if valid.sum() < 500:                          # fallback: neutral grey pixels
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        valid = hsv[..., 1] <= max(35.0, float(np.percentile(hsv[..., 1], 35)))
    valid &= inner == 0
    if region is not None and (valid & region).sum() >= 500:
        valid &= region
    return cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def empty_plate_estimate(img, food_union, scale=8, surface=None):
    """What the plate/tray looks like WITHOUT food: mask all food, inpaint
    at low resolution (so reflections and light gradients of the steel or
    plate carry on smoothly under the food), then scale back up."""
    h, w = food_union.shape
    sw, sh = max(8, w // scale), max(8, h // scale)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    # build the estimate only from neutral grey (empty steel/plate) pixels, so
    # coloured unlabelled food or table-cloth does not bleed into it
    if surface is None:
        surface = surface_mask(img, food_union)
    food_d = cv2.dilate(food_union, np.ones((15, 15), np.uint8))
    # near the food, also hide pixels that do not look like the plate (unlabelled
    # food, cutlery) - but only if the plate colour was found reliably there
    r = max(15, int(0.05 * max(h, w)))
    near = cv2.dilate(food_union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    near = (near > 0) & (food_d == 0)
    bad = food_d.copy()
    if near.sum() and surface[near].mean() >= 0.3:
        bad[near & ~surface] = 255
    m = cv2.resize(bad, (sw, sh), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    bg = cv2.inpaint(small, m, 5, cv2.INPAINT_TELEA)
    bg = cv2.GaussianBlur(bg, (0, 0), 1.5)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def clone_fill(img, eaten, food_union, rng, tiles=14, max_shift=0.45, surface=None):
    """Fill eaten pixels with REAL empty plate/tray copied from the nearest
    food-free place in the same photo (like a photo editor's clone stamp), so
    the steel texture, scratches and lighting stay real. The eaten area is
    split into tiles; each tile takes the smallest shift whose source pixels
    are all food-free. Seams are feathered."""
    h, w = eaten.shape
    out = img.astype(np.float32)
    # clone only the fine DETAIL (scratches, grain): image minus its blurred version.
    # Brightness/colour come from the smooth empty-plate estimate instead, so a
    # copied patch can never be too bright, too dark or the wrong colour.
    detail_src = out - cv2.GaussianBlur(out, (0, 0), 5)
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    # a valid source pixel is EMPTY PLATE/TRAY: not labelled food AND neutral
    # grey (low saturation) - steel and white plates are grey, while table-
    # cloths and (unlabelled) food are usually coloured
    if surface is None:
        surface = surface_mask(img, food_union)
    val = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surface = surface & (val < 250)                 # no blown-out highlights
    free = (cv2.dilate(food_union, np.ones((11, 11), np.uint8)) == 0) & surface
    # integral image of "free" pixels -> fast "is this box all free?" test
    ii = cv2.integral(free.astype(np.uint8))
    T = max(16, max(h, w) // tiles)
    step = max(4, T // 4)
    R = int(max_shift * max(h, w))
    offsets = [(dy, dx) for dy in range(-R, R + 1, step) for dx in range(-R, R + 1, step)]
    offsets.sort(key=lambda o: o[0] ** 2 + o[1] ** 2)
    for y0 in range(0, h, T):
        for x0 in range(0, w, T):
            cell = eaten[y0:y0 + T, x0:x0 + T] > 0
            if not cell.any():
                continue
            ys, xs = np.nonzero(cell)
            by0, by1 = y0 + ys.min(), y0 + ys.max() + 1
            bx0, bx1 = x0 + xs.min(), x0 + xs.max() + 1
            pad = max(3, T // 8)                        # overlap for soft seams
            by0, bx0 = max(0, by0 - pad), max(0, bx0 - pad)
            by1, bx1 = min(h, by1 + pad), min(w, bx1 + pad)
            bh, bw = by1 - by0, bx1 - bx0
            src = None
            for dy, dx in offsets:
                sy0, sx0 = by0 + dy, bx0 + dx
                if sy0 < 0 or sx0 < 0 or sy0 + bh > h or sx0 + bw > w or (dy == 0 and dx == 0):
                    continue
                n_free = ii[sy0 + bh, sx0 + bw] - ii[sy0, sx0 + bw] - ii[sy0 + bh, sx0] + ii[sy0, sx0]
                if n_free >= 0.98 * bh * bw:
                    src = (sy0, sx0)
                    break
            if src is None:
                continue
            sy0, sx0 = src
            patch = detail_src[sy0:sy0 + bh, sx0:sx0 + bw]
            # feathered weight: 1 inside, falling to 0 at the box border
            wy = np.minimum(np.arange(bh) + 1, bh - np.arange(bh)).astype(np.float32)
            wx = np.minimum(np.arange(bw) + 1, bw - np.arange(bw)).astype(np.float32)
            wgt = np.minimum.outer(np.minimum(wy / pad, 1), np.minimum(wx / pad, 1))
            acc[by0:by1, bx0:bx1] += patch * wgt[..., None]
            wsum[by0:by1, bx0:bx1] += wgt
    got = wsum > 1e-6
    detail = np.zeros_like(out)
    detail[got] = acc[got] / wsum[got][:, None]
    return detail, got


def fill_eaten(img, eaten, plate_px, rng, food_union=None, plate_region=None):
    """Eaten pixels -> empty plate = smooth empty-plate estimate (follows the
    real light) + real steel/plate DETAIL cloned from nearby food-free grey
    areas. Where no detail could be cloned, fine grain is used instead."""
    if not eaten.any():
        return img.copy()
    out = img.astype(np.float32)
    h, w = eaten.shape
    if food_union is None:
        food_union = eaten
    surface = surface_mask(img, food_union, plate_region)
    base = empty_plate_estimate(img, food_union, surface=surface)
    detail, got = clone_fill(img, eaten, food_union, rng, surface=surface)
    sd = np.clip(plate_px.std(axis=0), 1, 6).astype(np.float32) * 0.6
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32), (0, 0), 1.0) * sd
    detail = np.where(got[..., None], detail, grain)
    fill = base + detail
    e = (eaten > 0).astype(np.float32)
    a = np.maximum(cv2.GaussianBlur(e, (0, 0), 1.5), e)[..., None]   # soft edge
    return np.clip(out * (1 - a) + fill * a, 0, 255).astype(np.uint8)


def add_residue(img, before, items, names, rng):
    """What an eaten dish leaves behind (NOT counted as leftover, NOT labelled,
    so the segmentation model learns that residue is not food):
      wet dishes (dal, sabzi, chole, curd, sweet): a thin streaky film of the
        dish itself + a ring where the gravy touched the bowl wall
      dry dishes (rice, roti, puri, salad, other): scattered grains / crumbs,
        real pieces copied from the dish (about 0.5-1 % of the eaten area)
    items: list of (class_id, eaten_mask, original_instance_mask)."""
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    soft = cv2.GaussianBlur(before, (0, 0), 2.5).astype(np.float32)
    unit = max(h, w) / 1000.0
    for cid, eaten, inst in items:
        e = eaten > 0
        if e.sum() < 50:
            continue
        name = names[cid] if 0 <= cid < len(names) else "other"
        if name in WET:
            n = smooth_noise(rng, h, w, 0.18)
            streak = np.clip((n - rng.uniform(0.35, 0.55)) * 3.0, 0, 1)
            alpha = (0.10 + 0.40 * streak) * rng.uniform(0.6, 1.0)
            dist = cv2.distanceTransform((inst > 0).astype(np.uint8), cv2.DIST_L2, 3)
            ring = (dist < rng.uniform(2, 5) * unit) & (smooth_noise(rng, h, w, 0.3) > 0.45)
            alpha = np.where(ring, np.maximum(alpha, rng.uniform(0.25, 0.40)), alpha)
            alpha = cv2.GaussianBlur((alpha * e).astype(np.float32), (0, 0), 1.5)[..., None]
            film = soft * rng.uniform(0.80, 0.92)                # thinner, a bit darker
            out = out * (1 - alpha) + film * alpha
        else:
            ys, xs = np.nonzero(e)
            fy, fx = np.nonzero(inst > 0)
            if len(fy) == 0:
                continue
            ps = max(2, int(round(rng.uniform(3, 6) * unit)))   # grain / crumb size
            n = int(e.sum() * rng.uniform(0.004, 0.010) / (ps * ps))
            for j in rng.choice(len(ys), size=min(n, len(ys)), replace=True):
                k = rng.integers(len(fy))
                sy, sx = int(fy[k]), int(fx[k])
                ty, tx = int(ys[j]), int(xs[j])
                if sy + ps >= h or sx + ps >= w or ty + ps >= h or tx + ps >= w:
                    continue
                patch = before[sy:sy + ps, sx:sx + ps].astype(np.float32)
                yy, xx = np.mgrid[0:ps, 0:ps]
                m = (((yy - ps / 2) ** 2 + (xx - ps / 2) ** 2) <= (ps / 2) ** 2)[..., None]
                out[ty:ty + ps, tx:tx + ps] = np.where(m, patch, out[ty:ty + ps, tx:tx + ps])
    return np.clip(out, 0, 255).astype(np.uint8)


# ================================================================ 4. camera change
def random_homography(h, w, rng, rot=12, shift=0.04, scale=(0.95, 1.05), persp=0.02):
    ang = rng.uniform(-rot, rot)
    s = rng.uniform(*scale)
    A = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), ang, s), [0, 0, 1]])
    A[0, 2] += rng.uniform(-shift, shift) * w
    A[1, 2] += rng.uniform(-shift, shift) * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-persp, persp, (4, 2)) * [w, h]).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst)
    return P @ A


# ================================================================ 5. lighting
def apply_lighting(img, preset, rng, plate=None, glare=True):
    """Small brightness change + very soft uneven light + rare faint glare.
    No colour gains and no gamma, so the food looks the same as before."""
    f = img.astype(np.float32) / 255
    f = f * LIGHTING.get(preset, 1.0) * rng.uniform(0.96, 1.04)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lx, ly = rng.uniform(0, w), rng.uniform(0, h)
    f *= (1 - 0.06 * (np.hypot(xx - lx, yy - ly) / np.hypot(w, h)))[..., None]
    if glare and rng.random() < 0.25:
        gx, gy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        sx, sy = rng.uniform(0.01, 0.03) * w, rng.uniform(0.01, 0.02) * h
        blob = np.exp(-(((xx - gx) / sx) ** 2 + ((yy - gy) / sy) ** 2))
        f = f + (1 - f) * (blob[..., None] * rng.uniform(0.3, 0.5))
    f += rng.normal(0, 0.006, f.shape).astype(np.float32)     # sensor noise
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


# ================================================================ one event
def make_after(before, objs, waste_by_class, lighting, rng, plate=None, names=(),
               plate_region=None):
    """Return (after_img, after_objs, realised dict class_id->(before_px, left_px))."""
    h, w = before.shape[:2]
    insts = instance_masks(objs, h, w)
    if plate is None:
        plate = detect_plate(before)
    food_union = np.zeros((h, w), np.uint8)
    for _, m in insts:
        food_union |= m

    keep_masks, eaten_all, smear_info, realised = [], np.zeros((h, w), np.uint8), [], {}
    for cid, m in insts:
        frac = waste_by_class.get(cid, None)
        if frac is None:          # dish on the photo but not in the event: leave as is
            frac = 1.0
        keep = leftover_blob(m, frac, rng)
        eaten = cv2.subtract(m, keep)
        keep_masks.append((cid, keep))
        eaten_all |= eaten
        smear_info.append((cid, eaten, m))
        b, l = realised.get(cid, (0, 0))
        realised[cid] = (b + int((m > 0).sum()), l + int((keep > 0).sum()))

    # hand-drawn labels are a bit smaller than the food: grow the eaten area a
    # few pixels outward (not into kept food or other dishes) so no thin rim
    # of food is left behind as an outline
    grow = max(3, int(round(max(h, w) / 250)))
    keep_all = np.zeros((h, w), np.uint8)
    for _, k in keep_masks:
        keep_all |= k
    ring = cv2.dilate(eaten_all, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    ring = ring & ~food_union & ~cv2.dilate(keep_all, np.ones((5, 5), np.uint8))
    eaten_all = eaten_all | ring
    plate_px = plate_colour_pixels(before, food_union, plate)
    img = fill_eaten(before, eaten_all, plate_px, rng, food_union, plate_region)
    img = add_residue(img, before, smear_info, list(names), rng)

    # camera change (same homography for image and masks)
    Hm = random_homography(h, w, rng)
    img = cv2.warpPerspective(img, Hm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    c = np.float32([[plate["center"]]])
    new_c = cv2.perspectiveTransform(c, Hm)[0, 0]
    plate_after = dict(plate, center=(float(new_c[0]), float(new_c[1])))

    after_objs = []
    for cid, keep in keep_masks:
        wk = cv2.warpPerspective(keep, Hm, (w, h), flags=cv2.INTER_NEAREST)
        for poly in mask_to_polygons(wk):
            after_objs.append((cid, poly))

    img = apply_lighting(img, lighting, rng, plate_after)
    return img, after_objs, realised


# ================================================================ preview sheet
def preview_sheet(rows, path, tile=256):
    tiles = []
    for before, after, text in rows:
        b = cv2.resize(before, (tile, tile))
        a = cv2.resize(after, (tile, tile))
        pair = np.hstack([b, a])
        cv2.putText(pair, text, (5, tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pair, text, (5, tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(pair)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), grid)


# ================================================================ main
def generate(dataset, limit=None, max_side=1280, seed=123, n_preview=20, overwrite=True):
    ds = Path(dataset)
    gen = ds / "generated"
    (gen / "after").mkdir(parents=True, exist_ok=True)
    (gen / "after_labels").mkdir(parents=True, exist_ok=True)

    classes = read_classes(ds / "classes.txt")
    cid_of = {c: i for i, c in enumerate(classes)}
    events = pd.read_csv(gen / "events.csv")
    groups = list(events.groupby("event_id", sort=True))
    if limit:
        groups = groups[:limit]

    rng = np.random.default_rng(seed)
    cache, log, prev = {}, [], []
    for n, (eid, g) in enumerate(groups, 1):
        r0 = g.iloc[0]
        out_img = gen / "after" / r0["after_img"]
        if out_img.exists() and not overwrite:
            continue
        name = r0["before_img"]
        if name not in cache:                      # load + detect plate once per photo
            img = cv2.imread(str(ds / "before" / name))
            if img is None:
                print(f"  cannot read {name}, skipping {eid}")
                continue
            img = resize_max_side(img, max_side)
            objs = read_yolo_seg(ds / "labels" / (Path(name).stem + ".txt"))
            # plate / bowl outlines (dataset/plates/, from the public dataset) tell
            # the filler exactly which surface is under the food
            region = None
            pf = ds / "plates" / (Path(name).stem + ".txt")
            if pf.exists():
                region = np.zeros(img.shape[:2], np.uint8)
                for _, m in instance_masks(read_yolo_seg(pf), *img.shape[:2]):
                    region |= m
                region = region > 0
            cache[name] = (img, objs, detect_plate(img), region)
        before, objs, plate, region = cache[name]

        waste = {cid_of[d]: v / 100.0 for d, v in zip(g.dish, g.true_waste_pct) if d in cid_of}
        after, after_objs, realised = make_after(before, objs, waste, r0["lighting"],
                                                 rng, plate, classes, region)
        cv2.imwrite(str(out_img), after, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(82, 95))])
        write_yolo_seg(gen / "after_labels" / (Path(r0["after_img"]).stem + ".txt"), after_objs)

        for d, v in zip(g.dish, g.true_waste_pct):
            b, l = realised.get(cid_of.get(d, -1), (0, 0))
            log.append([eid, d, v, round(100 * l / b, 2) if b else np.nan, b])
        if len(prev) < n_preview:
            txt = f"{eid} {r0['meal']} " + " ".join(f"{d}:{v:.0f}" for d, v in
                                                     zip(g.dish, g.true_waste_pct))
            prev.append((before, after, txt[:60]))
        if n % 50 == 0 or n == len(groups):
            print(f"  {n}/{len(groups)} events done")

    logdf = pd.DataFrame(log, columns=["event_id", "dish", "true_waste_pct",
                                       "realised_pct_before_warp", "before_px"])
    logdf.to_csv(gen / "generation_log.csv", index=False)
    if prev:
        preview_sheet(prev, gen / "preview.jpg")
    err = (logdf.true_waste_pct - logdf.realised_pct_before_warp).abs()
    print(f"\n  realised vs planned waste: mean abs diff {err.mean():.3f} pct-points "
          f"(max {err.max():.3f})")
    print(f"  wrote {len(groups)} after images to {gen / 'after'}  |  preview: {gen / 'preview.jpg'}")


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic after images")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--limit", type=int, default=None, help="only the first N events")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="shrink before photos so the longest side is at most this")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--preview", type=int, default=20, help="events in preview.jpg")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    generate(args.dataset, args.limit, args.max_side, args.seed, args.preview,
             overwrite=not args.skip_existing)


if __name__ == "__main__":
    main()
