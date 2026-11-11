"""
segment.py  (B4)  - food segmentation: classical baseline + YOLOv8-seg.

Sub-commands (run from the project root):

  1. prepare   build the YOLO training set (split BY PHOTO, so no after image
               of a validation photo is ever seen in training)
       python src/segment.py prepare --dataset dataset
       -> yolo_ds/ (images/, labels/, data.yaml, split.json) + yolo_ds.zip

  2. train     (on Colab GPU - see notebooks/train_yolo_colab.ipynb) or here:
       python src/segment.py train --epochs 50
       -> models/best.pt

  3. baseline  classical segmentation on the validation photos
       python src/segment.py baseline --dataset dataset

  4. evaluate  mIoU of classical vs YOLO on the validation photos
       python src/segment.py evaluate --dataset dataset --weights models/best.pt
       -> results/segmentation_eval.csv, results/seg_examples.jpg, metrics.json

Metrics
  mIoU      = mean over food classes of IoU (class maps, all val images pooled)
  food IoU  = IoU of "any food" vs "no food" - this is what waste.py needs

Classical baseline (course topics: clustering, graph-cut, LBP)
  food mask : Lab distance from the plate/steel colour (low-saturation pixels)
              -> GrabCut refinement
  classes   : k-means in (L, a, b, x, y) inside the food mask -> each cluster
              gets the class whose prototype (mean Lab + LBP texture
              histogram, learned from the TRAIN photos' labels) is nearest.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import (instance_masks, mask_to_polygons, read_classes,
                    read_yolo_seg, resize_max_side, write_yolo_seg)

IMG_EXT = (".jpg", ".jpeg", ".png")


# ================================================================ helpers
def label_map(objs, h, w):
    """Semantic map: 0 = no food, k = class k-1."""
    m = np.zeros((h, w), np.int32)
    for cid, mask in instance_masks(objs, h, w):
        m[mask > 0] = cid + 1
    return m


def masks_to_map(cmasks, h, w):
    """{cid: mask} -> semantic map (bigger regions painted first)."""
    m = np.zeros((h, w), np.int32)
    for cid, mask in sorted(cmasks.items(), key=lambda t: -int((t[1] > 0).sum())):
        m[mask > 0] = cid + 1
    return m


# ================================================================ 1. prepare
def choose_split(photos, photo_classes, n_val=4, seed=0, tries=200):
    """Pick validation photos that cover many classes while every class
    keeps >= 2 photos in training."""
    rng = random.Random(seed)
    best, best_score = None, -1
    all_c = set().union(*photo_classes.values())
    for _ in range(tries):
        val = rng.sample(photos, n_val)
        train = [p for p in photos if p not in val]
        cnt = {c: sum(c in photo_classes[p] for p in train) for c in all_c}
        if min(cnt.values()) < 2:
            continue
        score = len(set().union(*[photo_classes[p] for p in val]))
        if score > best_score:
            best, best_score = val, score
    if best is None:
        best = rng.sample(photos, n_val)
    return sorted(p for p in photos if p not in best), sorted(best)


def prepare(dataset, out="yolo_ds", n_val=0, afters_per_photo=15, max_side=1024, seed=0):
    ds = Path(dataset)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    classes = read_classes(ds / "classes.txt")
    photos = sorted(p.name for p in (ds / "before").iterdir()
                    if p.suffix.lower() in IMG_EXT and (ds / "labels" / (p.stem + ".txt")).exists())
    pcls = {p: {c for c, _ in read_yolo_seg(ds / "labels" / (Path(p).stem + ".txt"))} for p in photos}
    if not n_val:                                   # default: hold out 15 % of the photos
        n_val = max(4, int(round(0.15 * len(photos))))
    train, val = choose_split(photos, pcls, n_val, seed)

    ev_path = ds / "generated" / "events.csv"
    ev = pd.read_csv(ev_path).drop_duplicates("event_id") if ev_path.exists() else pd.DataFrame()
    have = {f.name for f in (ds / "generated" / "after").glob("*.jpg")} if ev_path.exists() else set()

    rng = random.Random(seed)
    n = {"train": 0, "val": 0}
    for split, plist in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p in plist:
            items = [(ds / "before" / p, ds / "labels" / (Path(p).stem + ".txt"), "b_" + Path(p).stem)]
            if len(ev):
                afters = [a for a in ev[ev.before_img == p].after_img if a in have]
                rng.shuffle(afters)
                for a in afters[:afters_per_photo]:
                    items.append((ds / "generated" / "after" / a,
                                  ds / "generated" / "after_labels" / (Path(a).stem + ".txt"),
                                  "a_" + Path(a).stem))
            for img_p, lab_p, name in items:
                img = resize_max_side(cv2.imread(str(img_p)), max_side)
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                shutil.copy(lab_p, out / "labels" / split / f"{name}.txt")
                n[split] += 1

    (out / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        f"nc: {len(classes)}\nnames: {json.dumps(classes)}\n")
    (out / "split.json").write_text(json.dumps({"train": train, "val": val}, indent=1))
    zip_path = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print(f"  photos: {len(train)} train / {len(val)} val  (val = {val})")
    print(f"  images: {n['train']} train / {n['val']} val  (before photos + up to "
          f"{afters_per_photo} generated after images per photo)")
    print(f"  wrote {out}/ and {zip_path}  -> upload the zip to Colab")


def prepare_food(yolo_ds="yolo_ds", dataset="dataset", out="food_ds",
                 empty_val=0.15, max_side=1024, seed=0):
    """ONE-class "food" set for the LEFTOVER model (on the after photo we only
    need 'food or not'; the dish name comes from the before photo):
      - yolo_ds (real before photos + generated after photos, same photo split,
        held-out photos stay in val), every dish class -> 0. Residue in the
        generated afters is unlabelled, so the model learns residue = not food.
      - dataset/empty/: real finished plates from the public dataset (crumbs,
        sauce smears, peels, no food) as negatives with empty labels."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    def put(img_p, lab_p, split, name):
        img = cv2.imread(str(img_p))
        if img is None:
            return False
        img = resize_max_side(img, max_side)
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        if lab_p is not None and Path(lab_p).exists():
            for line in Path(lab_p).read_text().splitlines():
                p_ = line.split()
                if len(p_) >= 7:
                    lines.append("0 " + " ".join(p_[1:]))
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        return True

    n = {"ours_train": 0, "ours_val": 0, "empty_train": 0, "empty_val": 0}
    yd = Path(yolo_ds)
    for sp in ("train", "val"):
        for ip in sorted((yd / "images" / sp).glob("*.jpg")):
            if put(ip, yd / "labels" / sp / (ip.stem + ".txt"), sp, "o_" + ip.stem):
                n[f"ours_{sp}"] += 1
    ed = Path(dataset) / "empty"
    ims = sorted(p for p in ed.iterdir() if p.suffix.lower() in IMG_EXT) if ed.exists() else []
    rng = random.Random(seed)
    rng.shuffle(ims)
    n_val = int(round(empty_val * len(ims)))
    for k, ip in enumerate(ims):
        sp = "val" if k < n_val else "train"
        if put(ip, None, sp, "e_" + ip.stem):
            n[f"empty_{sp}"] += 1
    (out / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [\"food\"]\n")
    zip_path = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print(f"  plates with food: {n['ours_train']} train / {n['ours_val']} val (same photo split as yolo_ds)")
    print(f"  finished plates (no food): {n['empty_train']} train / {n['empty_val']} val")
    print(f"  wrote {out}/ and {zip_path}  -> train with notebooks/train_yolo.ipynb, "
          f"save the result as models/food_best.pt")


# ================================================================ 2. train
def train(data="yolo_ds/data.yaml", epochs=50, imgsz=640, model="yolov8n-seg.pt",
          out="models", device=None):
    from ultralytics import YOLO
    # YOLO reads "path: ." relative to the folder you run from, not the yaml's
    # folder -> write a copy with the absolute dataset path
    data = Path(data).resolve()
    lines = [l for l in data.read_text().splitlines() if not l.startswith("path:")]
    abs_yaml = data.parent / "data_abs.yaml"
    abs_yaml.write_text(f"path: {data.parent}\n" + "\n".join(lines) + "\n")
    m = YOLO(model)
    kw = dict(data=str(abs_yaml), epochs=epochs, imgsz=imgsz, batch=8,
              patience=20, project="runs_seg", name="food", exist_ok=True, seed=0,
              degrees=30, fliplr=0.5, flipud=0.5, hsv_v=0.4, mosaic=1.0, verbose=False)
    if device is not None:
        kw["device"] = device
    m.train(**kw)
    # newer Ultralytics versions put runs under runs/segment/<project>/<name>
    best = Path(m.trainer.save_dir) / "weights" / "best.pt"
    Path(out).mkdir(exist_ok=True)
    shutil.copy(best, Path(out) / "best.pt")
    print(f"  saved {Path(out) / 'best.pt'}   (training curves etc. in {m.trainer.save_dir})")


# ================================================================ YOLO inference
_MODEL = {}


def load_yolo(weights="models/best.pt"):
    if weights not in _MODEL:
        from ultralytics import YOLO
        _MODEL[weights] = YOLO(weights)
    return _MODEL[weights]


def yolo_class_masks(img, weights="models/best.pt", conf=0.25, imgsz=640):
    """{class_id: mask (0/255)} in the input image's size."""
    h, w = img.shape[:2]
    res = load_yolo(weights).predict(img, imgsz=imgsz, conf=conf, verbose=False,
                                     retina_masks=True)[0]
    out = {}
    if res.masks is None:
        return out
    # paint low-confidence first so confident masks win overlaps
    order = np.argsort(res.boxes.conf.cpu().numpy())
    cls = res.boxes.cls.cpu().numpy().astype(int)
    data = res.masks.data.cpu().numpy()
    for i in order:
        m = data[i]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out.setdefault(cls[i], np.zeros((h, w), np.uint8))
        out[cls[i]][m > 0.5] = 255
    return out


def yolo_food_mask(img, **kw):
    m = np.zeros(img.shape[:2], np.uint8)
    for v in yolo_class_masks(img, **kw).values():
        m |= v
    return m


def predict_class_masks(img, weights="models/best.pt"):
    """Used by app.py for uploaded photos (dish regions on the before photo)."""
    return yolo_class_masks(img, weights)


# ================================================================ classical baseline
def lbp(gray):
    """8-neighbour local binary pattern code per pixel (0..255)."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, np.uint8)
    nbrs = [g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:], g[1:-1, 2:],
            g[2:, 2:], g[2:, 1:-1], g[2:, :-2], g[1:-1, :-2]]
    for k, n in enumerate(nbrs):
        code |= ((n >= c).astype(np.uint8) << k)
    return np.pad(code, 1)


def region_features(lab, lbpc, mask, bins=16):
    px = lab[mask]
    hist = np.bincount((lbpc[mask] // (256 // bins)).astype(int), minlength=bins).astype(float)
    return px.mean(axis=0), hist / (hist.sum() + 1e-9)


class Classical:
    """Food mask (colour distance + GrabCut) and class by nearest prototype."""

    def __init__(self, n_classes, work_side=512):
        self.n = n_classes
        self.side = work_side
        self.protos = {}

    def _prep(self, img):
        small = resize_max_side(img, self.side)
        lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        return small, lab, lbp(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

    def fit(self, items):
        """items: list of (img, objs) from TRAIN photos."""
        feats = {}
        for img, objs in items:
            small, lab, lc = self._prep(img)
            h, w = small.shape[:2]
            for cid, m in instance_masks(objs, h, w):
                mm = m > 0
                if mm.sum() < 50:
                    continue
                feats.setdefault(cid, []).append(region_features(lab, lc, mm))
        for cid, fl in feats.items():
            self.protos[cid] = (np.mean([f[0] for f in fl], axis=0),
                                np.mean([f[1] for f in fl], axis=0))
        return self

    def food_mask(self, small, lab):
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1].astype(float)
        neutral = s <= np.percentile(s, 30)                  # steel / plate / table
        ref = np.median(lab[neutral], axis=0)
        d = lab - ref
        dist = np.sqrt((0.3 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
        d8 = np.clip(dist * 3, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = dist > max(12, t / 3)
        # GrabCut: probable fg/bg from the colour mask, sure fg = eroded core
        gc = np.where(fg, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        core = cv2.erode(fg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        gc[core] = cv2.GC_FGD
        if 0 < fg.mean() < 0.95:
            bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
            try:
                cv2.grabCut(small, gc, None, bgm, fgm, 3, cv2.GC_INIT_WITH_MASK)
                fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
            except cv2.error:
                pass
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return fg > 0

    def predict_map(self, img, k=8):
        """Semantic map at the INPUT image size (0 = no food, c+1 = class c)."""
        small, lab, lc = self._prep(img)
        h, w = small.shape[:2]
        fg = self.food_mask(small, lab)
        out = np.zeros((h, w), np.int32)
        ys, xs = np.nonzero(fg)
        if len(ys) > k * 20 and self.protos:
            X = np.column_stack([lab[ys, xs] / 4, xs / w * 40, ys / h * 40]).astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, lbl, _ = cv2.kmeans(X, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
            lbl = lbl.ravel()
            for j in range(k):
                sel = lbl == j
                if sel.sum() < 30:
                    continue
                m = np.zeros((h, w), bool)
                m[ys[sel], xs[sel]] = True
                mu, hist = region_features(lab, lc, m)
                best, bd = None, 1e9
                for cid, (pm, ph) in self.protos.items():
                    chi = 0.5 * np.sum((hist - ph) ** 2 / (hist + ph + 1e-9))
                    dd = np.linalg.norm(mu - pm) / 20 + 2 * chi
                    if dd < bd:
                        best, bd = cid, dd
                out[m] = best + 1
        H, W = img.shape[:2]
        return cv2.resize(out.astype(np.uint16), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    def predict_food(self, img):
        small, lab, _ = self._prep(img)
        fg = self.food_mask(small, lab).astype(np.uint8) * 255
        H, W = img.shape[:2]
        return cv2.resize(fg, (W, H), interpolation=cv2.INTER_NEAREST)


# ================================================================ evaluation
def val_items(dataset, split_path="yolo_ds/split.json", afters_per_photo=10, seed=0):
    """(kind, img_path, label_path, before_photo_path) for val before photos
    + their generated after images."""
    ds = Path(dataset)
    split = json.loads(Path(split_path).read_text())
    items = []
    ev_path = ds / "generated" / "events.csv"
    ev = pd.read_csv(ev_path).drop_duplicates("event_id") if ev_path.exists() else pd.DataFrame()
    have = {f.name for f in (ds / "generated" / "after").glob("*.jpg")}
    rng = random.Random(seed)
    for p in split["val"]:
        items.append(("before", ds / "before" / p, ds / "labels" / (Path(p).stem + ".txt"),
                      ds / "before" / p))
        if len(ev):
            afters = sorted(a for a in ev[ev.before_img == p].after_img if a in have)
            rng.shuffle(afters)
            for a in afters[:afters_per_photo]:
                items.append(("after", ds / "generated" / "after" / a,
                              ds / "generated" / "after_labels" / (Path(a).stem + ".txt"),
                              ds / "before" / p))
    return split, items


def iou_scores(pairs, n_classes):
    """pairs: list of (gt_map, pred_map, care_mask). Only pixels inside
    care_mask count (= the labelled dish areas, where waste is measured;
    A did not label every food item, so food elsewhere has no ground truth).
    Returns (mIoU over food classes present in GT, food-vs-not IoU, per-class)."""
    inter = np.zeros(n_classes + 1)
    union = np.zeros(n_classes + 1)
    fi = fu = 0
    present = set()
    for gt, pr, care in pairs:
        gt = np.where(care, gt, 0)
        pr = np.where(care, pr, 0)
        for c in range(1, n_classes + 1):
            g, p = gt == c, pr == c
            if g.any():
                present.add(c)
            inter[c] += (g & p).sum()
            union[c] += (g | p).sum()
        g, p = gt > 0, pr > 0
        fi += (g & p).sum()
        fu += (g | p).sum()
    per = {c - 1: inter[c] / union[c] for c in sorted(present) if union[c] > 0}
    return float(np.mean(list(per.values()))) if per else float("nan"), \
        float(fi / fu) if fu else float("nan"), per


def evaluate(dataset, weights=None, split_path="yolo_ds/split.json", out="results",
             max_side=1024, n_show=4):
    ds = Path(dataset)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    classes = read_classes(ds / "classes.txt")
    split, items = val_items(dataset, split_path)

    train_items = []
    for p in split["train"]:
        img = resize_max_side(cv2.imread(str(ds / "before" / p)), max_side)
        train_items.append((img, read_yolo_seg(ds / "labels" / (Path(p).stem + ".txt"))))
    clf = Classical(len(classes)).fit(train_items)

    methods = {"classical": lambda im: clf.predict_map(im)}
    if weights and Path(weights).exists():
        methods["yolo"] = lambda im: masks_to_map(yolo_class_masks(im, weights), *im.shape[:2])
    elif weights:
        print(f"  (no YOLO weights at {weights} - evaluating the classical baseline only)")

    pairs = {m: {"before": [], "after": []} for m in methods}
    shows = []
    from align import align_pair
    bcache = {}
    for kind, ip, lp, bp in items:
        img = resize_max_side(cv2.imread(str(ip)), max_side)
        gt = label_map(read_yolo_seg(lp), *img.shape[:2])
        if bp not in bcache:
            bimg = resize_max_side(cv2.imread(str(bp)), max_side)
            bl = label_map(read_yolo_seg(ds / "labels" / (Path(bp).stem + ".txt")), *bimg.shape[:2])
            bcache[bp] = (bimg, bl)
        bimg, bl = bcache[bp]
        preds = {m: f(img) for m, f in methods.items()}
        if kind == "after":
            # bring the after image's maps into the before photo's frame
            r = align_pair(bimg, img, pre_before=None)
            Hb_inv = np.linalg.inv(r["H_before"])
            Hfull = Hb_inv @ r["H_after_total"]          # after photo -> before photo
            wh = (bimg.shape[1], bimg.shape[0])
            warp = lambda a: cv2.warpPerspective(a.astype(np.uint16), Hfull, wh,
                                                 flags=cv2.INTER_NEAREST).astype(np.int32)
            gt = warp(gt)
            preds = {m: warp(p) for m, p in preds.items()}
        care = cv2.dilate((bl > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        row = [img]
        for m, pr in preds.items():
            pairs[m][kind].append((gt, pr, care))
            row.append(colour_map(pr if kind == "before" else methods[m](img), img))
        if len(shows) < n_show and kind == "after":
            t = 256
            shows.append(np.hstack([cv2.resize(x, (t, t)) for x in
                                    [row[0], colour_map(gt, img)] + row[1:]]))

    rows = []
    metrics = {}
    for m in methods:
        for kind in ("before", "after", "all"):
            pl = pairs[m]["before"] + pairs[m]["after"] if kind == "all" else pairs[m][kind]
            miou, fiou, per = iou_scores(pl, len(classes))
            r = {"method": m, "images": kind, "n": len(pl), "mIoU": round(miou, 3),
                 "food_IoU": round(fiou, 3)}
            r.update({f"IoU_{classes[c]}": round(v, 3) for c, v in per.items()})
            rows.append(r)
            if kind == "all":
                metrics[f"seg_mIoU_{m}"] = round(miou, 3)
                metrics[f"seg_foodIoU_{m}"] = round(fiou, 3)
    df = pd.DataFrame(rows)
    df.to_csv(out / "segmentation_eval.csv", index=False)
    if shows:
        cv2.imwrite(str(out / "seg_examples.jpg"), np.vstack(shows))
    print(f"  validation photos: {split['val']}  ({len(items)} images incl. generated afters)")
    print(df[["method", "images", "n", "mIoU", "food_IoU"]].to_string(index=False))
    print(f"  per-class IoU in {out / 'segmentation_eval.csv'}; examples (image | truth | "
          f"{' | '.join(methods)}) in {out / 'seg_examples.jpg'}")
    mp = out / "metrics.json"
    allm = json.loads(mp.read_text()) if mp.exists() else {}
    allm.update(metrics)
    mp.write_text(json.dumps(allm, indent=2))
    return df


PAL = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245),
       (180, 30, 145), (240, 240, 70), (230, 50, 240), (60, 245, 210), (190, 190, 250),
       (128, 128, 0), (0, 128, 128)]


def colour_map(sem, img):
    vis = img.copy()
    for c in np.unique(sem):
        if c == 0:
            continue
        m = sem == c
        vis[m] = (0.4 * vis[m] + 0.6 * np.array(PAL[(c - 1) % len(PAL)])).astype(np.uint8)
    return vis


# ================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Food segmentation (B4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", default="yolo_ds")
    p.add_argument("--val-photos", type=int, default=0, help="0 = 15 %% of the photos")
    p.add_argument("--afters-per-photo", type=int, default=15)
    pf = sub.add_parser("prepare_food", help="one-class leftover-food set")
    pf.add_argument("--yolo-ds", default="yolo_ds")
    pf.add_argument("--dataset", default="dataset")
    pf.add_argument("--out", default="food_ds")
    t = sub.add_parser("train")
    t.add_argument("--data", default="yolo_ds/data.yaml")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--device", default=None)
    for name in ("baseline", "evaluate"):
        e = sub.add_parser(name)
        e.add_argument("--dataset", default="dataset")
        e.add_argument("--split", default="yolo_ds/split.json")
        e.add_argument("--out", default="results")
        if name == "evaluate":
            e.add_argument("--weights", default="models/best.pt")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.dataset, a.out, a.val_photos, a.afters_per_photo)
    elif a.cmd == "prepare_food":
        prepare_food(a.yolo_ds, a.dataset, a.out)
    elif a.cmd == "train":
        train(a.data, a.epochs, a.imgsz, device=a.device)
    elif a.cmd == "baseline":
        evaluate(a.dataset, None, a.split, a.out)
    else:
        evaluate(a.dataset, a.weights, a.split, a.out)


if __name__ == "__main__":
    main()
