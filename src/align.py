"""
align.py  (B3)  - put the "after" photo exactly on top of the "before" photo.

Steps
  1. rectify both photos with preprocess.py  (plate -> top-down circle, N4)
  2. SIFT keypoints on both rectified images (inside the plate + rim),
     Lowe ratio test (0.75), RANSAC homography  after -> before.
  3. sanity check the homography (enough inliers, no crazy scale/tilt).
  4. if SIFT fails: ECC (rotation + shift) on the rectified images;
     if that also fails: plate-circle alignment only (identity in the
     rectified frame - the circles already line up, only rotation is lost).

Use from code:
    from align import align_pair
    r = align_pair(before_bgr, after_bgr)
    r["before"], r["after_warped"], r["H"], r["method"], r["inliers"]

Command line (check on generated data, needs events.csv + labels):
    python src/align.py --dataset toy_dataset --limit 40
  prints how often each method was used and a "containment" score:
  % of the after-leftover pixels (ground truth) that land inside the same
  dish's before region. 100% = perfect alignment. Compared against the
  plate-only fallback = the ablation "with / without alignment".
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import class_masks, read_classes, read_yolo_seg, resize_max_side
from preprocess import SIZE, circle_mask, preprocess


# ================================================================ SIFT + RANSAC
_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=4000)
    return _SIFT


def _feature_mask(img, mask=None):
    """Where to look for keypoints: the plate circle grown a little (round
    plates, square image) or the given mask / whole image (trays)."""
    if mask is not None:
        return mask
    h, w = img.shape[:2]
    if h == w:
        return circle_mask(h, fill=0.98)
    return np.full((h, w), 255, np.uint8)


def sift_homography(before, after, ratio=0.75, min_inliers=15, mask=None, loose=False,
                    min_spread=0.12):
    """Homography that maps AFTER -> BEFORE (both rectified), or None.
    Returns (H, n_good_matches, n_inliers)."""
    gb = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    ga = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    kb, db = _sift().detectAndCompute(gb, _feature_mask(before, mask))
    ka, da = _sift().detectAndCompute(ga, _feature_mask(after, None if mask is None else
                                                         np.full(ga.shape, 255, np.uint8)))
    if db is None or da is None or len(kb) < 8 or len(ka) < 8:
        return None, 0, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < min_inliers:
        return None, len(good), 0
    src = np.float32([ka[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=3000,
                                confidence=0.995)
    n_inl = int(inl.sum()) if inl is not None else 0
    if H is None or n_inl < min_inliers or not homography_ok(H, before.shape, loose):
        return None, len(good), n_inl
    # the matches must cover the plate/tray, not one small object (e.g. the same
    # paper cup appearing in photos of two DIFFERENT trays)
    pts = dst[inl.ravel() > 0].reshape(-1, 2)
    spread = cv2.contourArea(cv2.convexHull(pts)) / float(before.shape[0] * before.shape[1])
    if spread < min_spread:
        return None, len(good), n_inl
    return H, len(good), n_inl


def homography_ok(H, shape=(SIZE, SIZE), loose=False):
    """Reject homographies that do not look like 'same plate, camera moved a bit'.
    Rectified round plates: H is close to a rotation (strict limits).
    Trays (loose=True): photos are not rectified, so allow more scale,
    tilt and shift - any rotation is fine in both cases."""
    if isinstance(shape, (int, float)):
        shape = (shape, shape)
    max_scale, max_persp, max_shift = (2.0, 0.004, 0.5) if loose else (1.4, 0.002, 0.35)
    size = max(shape[:2]) * 1.0
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-8:
        return False
    H = H / H[2, 2]
    A = H[:2, :2]
    det = np.linalg.det(A)
    if det <= 0:                               # mirror image -> wrong
        return False
    s = np.sqrt(det)
    if not (1 / max_scale < s < max_scale):
        return False
    # where does the plate centre go?
    if loose:                                    # perspective terms scale with image size
        max_persp = max_persp * 1024 / size
    if np.abs(H[2, :2]).max() > max_persp:
        return False
    h, w = shape[:2]
    c = np.array([w / 2, h / 2, 1.0])
    c2 = H @ c
    c2 = c2[:2] / c2[2]
    return np.hypot(*(c2 - [w / 2, h / 2])) < max_shift * size


# ================================================================ ECC fallback
def _edge_map(img, scale):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (0, 0), 2)
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    e = cv2.GaussianBlur(e, (0, 0), 2)
    return cv2.normalize(e, None, 0, 1, cv2.NORM_MINMAX)


def ecc_homography(before, after, mask=None, loose=False, min_cc=0.5,
                   scales=(0.125, 0.25, 0.5)):
    """Fallback when SIFT fails (e.g. the after photo's fine texture changed).
    Coarse-to-fine ECC with a full homography on blurred EDGE maps: it aligns
    the plate/tray SHAPE (rim, compartment walls), not fine texture.
    Several starting guesses (scale x rotation) are tried; the best
    correlation wins. Returns (H after->before, correlation) or (None, cc)."""
    h, w = before.shape[:2]
    a = cv2.resize(after, (w, h)) if after.shape[:2] != (h, w) else after
    inits = []
    for sc in (0.9, 1.0, 1.1, 1.2):
        for ang in (-10, 0, 10):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            inits.append(np.vstack([M, [0, 0, 1]]).astype(np.float32))
    edges = {s: (_edge_map(before, s), _edge_map(a, s)) for s in scales}
    best = None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    for W in inits:                       # W maps before coords -> after coords
        cc, ok = 0.0, True
        for s in scales:
            S = np.diag([s, s, 1]).astype(np.float32)
            Ws = (S @ W @ np.linalg.inv(S)).astype(np.float32)
            eb, ea = edges[s]
            try:
                cc, Ws = cv2.findTransformECC(eb, ea, Ws, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
            except cv2.error:
                ok = False
                break
            W = np.linalg.inv(S) @ Ws @ S
        if ok and (best is None or cc > best[0]):
            best = (cc, W)
    if best is None or best[0] < min_cc:
        return None, (best[0] if best else 0.0)
    H = np.linalg.inv(best[1])            # after(resized) -> before
    return (H if homography_ok(H, before.shape, loose) else None), float(best[0])


# ================================================================ main API
def align_rectified(before, after, use_sift=True, use_ecc=True, mask=None, loose=False):
    """Map AFTER onto BEFORE. Round plates: both rectified (same size).
    Trays: both just resized photos (sizes may differ). Returns dict."""
    h, w = before.shape[:2]
    H, n_good, n_inl, method = None, 0, 0, "plate"
    if use_sift:
        H, n_good, n_inl = sift_homography(before, after, mask=mask, loose=loose)
        if H is not None:
            method = "sift"
    if H is None and use_ecc:
        H, _ = ecc_homography(before, after, mask=mask, loose=loose)
        if H is not None:
            method = "ecc"
            sa = np.diag([w / after.shape[1], h / after.shape[0], 1.0])
            H = H @ sa                    # ECC worked on after resized to before size
    if H is None:
        H = np.diag([w / after.shape[1], h / after.shape[0], 1.0])
        method = "plate" if not loose else "none"
    warped = cv2.warpPerspective(after, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return {"H": H, "after_warped": warped, "method": method,
            "matches": n_good, "inliers": n_inl}


def align_pair(before_bgr, after_bgr, color_correct=True, align=True,
               pre_before=None, size=SIZE):
    """Full B3 step on two raw photos.
    align=False gives the 'plate-circle only' result (ablation).
    Returns dict:
        before, after_rect, after_warped : rectified images (size x size)
        H            : after_rect -> before (rectified frame)
        H_before     : original before photo -> rectified
        H_after      : original after photo  -> rectified
        H_after_total: original after photo  -> rectified BEFORE frame
        method, matches, inliers, plate_mask
    """
    pb = pre_before or preprocess(before_bgr, size=size, color_correct=color_correct)
    pa = preprocess(after_bgr, size=size, color_correct=color_correct, shape=pb["shape"])
    tray = pb["shape"] == "tray"
    if align:
        # mask=None -> plate circle grown to include the rim (round) / whole photo (tray)
        r = align_rectified(pb["image"], pa["image"], loose=tray)
    else:
        hb, wb = pb["image"].shape[:2]
        S = np.diag([wb / pa["image"].shape[1], hb / pa["image"].shape[0], 1.0])
        r = {"H": S, "after_warped": cv2.resize(pa["image"], (wb, hb)),
             "method": "plate" if not tray else "none", "matches": 0, "inliers": 0}
    r.update({"before": pb["image"], "after_rect": pa["image"],
              "H_before": pb["H"], "H_after": pa["H"],
              "H_after_total": r["H"] @ pa["H"], "plate_mask": pb["plate_mask"],
              "shape": pb["shape"]})
    return r


def warp_mask(mask, H, shape=None):
    """Warp a mask from an original photo into the processed frame with H.
    shape = (h, w) of the target image (default SIZE x SIZE)."""
    h, w = (SIZE, SIZE) if shape is None else shape[:2]
    return cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST)


def overlay(before, after_warped):
    """Red/cyan check image: before in red channel, after in green+blue.
    Grey where they agree, coloured fringes where they do not."""
    gb = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    ga = cv2.cvtColor(after_warped, cv2.COLOR_BGR2GRAY)
    return cv2.merge([ga, ga, gb])


# ================================================================ evaluation CLI
def containment(after_masks, before_masks):
    """Share of after-leftover pixels lying inside the same class's before
    region (all classes pooled). 1.0 = perfect alignment."""
    inside = total = 0
    for cid, am in after_masks.items():
        a = am > 0
        total += int(a.sum())
        if cid in before_masks:
            inside += int((a & (before_masks[cid] > 0)).sum())
    return inside / total if total else np.nan


def evaluate(dataset, limit=40, max_side=1280, out_dir="results/align_check", n_show=8):
    ds = Path(dataset)
    gen = ds / "generated"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(gen / "events.csv").drop_duplicates("event_id")
    events = events[events.after_img.apply(lambda f: (gen / "after" / f).exists())]
    events = events.head(limit)
    if events.empty:
        raise SystemExit("No generated after images found - run generate_after.py first.")

    cache, rows, shows = {}, [], []
    for _, e in events.iterrows():
        if e.before_img not in cache:
            img = resize_max_side(cv2.imread(str(ds / "before" / e.before_img)), max_side)
            objs = read_yolo_seg(ds / "labels" / (Path(e.before_img).stem + ".txt"))
            pre = preprocess(img)
            h, w = img.shape[:2]
            bm = {c: warp_mask(m, pre["H"], pre["image"].shape)
                  for c, m in class_masks(objs, h, w).items()}
            cache[e.before_img] = (img, pre, bm)
        bimg, pre, bmasks = cache[e.before_img]
        aimg = cv2.imread(str(gen / "after" / e.after_img))
        aobjs = read_yolo_seg(gen / "after_labels" / (Path(e.after_img).stem + ".txt"))
        h, w = aimg.shape[:2]
        araw = class_masks(aobjs, h, w)

        r = align_pair(bimg, aimg, pre_before=pre)
        shp = r["before"].shape
        am_full = {c: warp_mask(m, r["H_after_total"], shp) for c, m in araw.items()}
        am_plate = {c: warp_mask(m, r["H_after"], shp) for c, m in araw.items()}
        rows.append({"event_id": e.event_id, "method": r["method"],
                     "matches": r["matches"], "inliers": r["inliers"],
                     "contain_aligned": containment(am_full, bmasks),
                     "contain_plate_only": containment(am_plate, bmasks)})
        if len(shows) < n_show:
            t = 256
            tiles = [cv2.resize(x, (t, t)) for x in
                     (r["before"], r["after_rect"], r["after_warped"],
                      overlay(r["before"], r["after_warped"]))]
            row = np.hstack(tiles)
            cv2.putText(row, f"{e.event_id} {r['method']} inl={r['inliers']}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            shows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "align_eval.csv", index=False)
    if shows:
        cv2.imwrite(str(out / "align_check.jpg"), np.vstack(shows))
    print(f"  pairs: {len(df)}   methods used: {df.method.value_counts().to_dict()}")
    print(f"  containment (after leftovers inside before regions):")
    print(f"     with alignment : {100 * df.contain_aligned.mean():5.1f}%")
    print(f"     plate only     : {100 * df.contain_plate_only.mean():5.1f}%")
    print(f"  wrote {out / 'align_eval.csv'} and {out / 'align_check.jpg'}")
    print("  (check image columns: before | after rectified | after aligned | overlay -"
          " grey = match, red/cyan fringes = mismatch)")
    return df


def main():
    ap = argparse.ArgumentParser(description="Before/after alignment (SIFT + RANSAC)")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="results/align_check")
    args = ap.parse_args()
    evaluate(args.dataset, args.limit, out_dir=args.out)


if __name__ == "__main__":
    main()
