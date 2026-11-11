"""
import_public.py - build dataset/ from the PUBLIC canteen dataset
(Roboflow "PROJ3 - Food waste detection", Piccoli et al. 2025,
Applied Sciences 15(13):7137, CC BY 4.0; YOLOv8 segmentation export).

    python tools/import_public.py "PATH/TO/PROJ3 - Food waste detection" --out dataset

The 87 original classes are grouped into 15 dishes (GROUPS below).
Non-food objects (plate, cutlery, cups, napkins/garbage, bones, drinks) are dropped.

Output (the project's data contract):
    dataset/before/img_001.jpg ...   photos that contain food (long side <= 1280 px)
    dataset/labels/img_001.txt ...   YOLO-seg polygons, class ids = classes.txt order
    dataset/classes.txt              the 15 dish names
    dataset/empty/empty_001.jpg ...  photos with NO food (finished trays) -> extra
                                     negatives for the leftover model
    dataset/plates/img_001.txt ...   plate / bowl / tray polygons (used by generate_after.py)
    dataset/source_map.csv           new name -> original file, split, dishes
"""
import argparse
import csv
import glob
import os
import shutil
import sys
from collections import Counter

import cv2
import yaml

GROUPS = {
    "rice":       ["rice", "cabidela rice"],
    "pasta":      ["pasta", "spaghetti", "lasagna", "pizza"],
    "potato":     ["french fries", "chips", "baked potatoes", "mashed potatoes"],
    "vegetables": ["vegetables", "mushrooms", "carrot", "brocolis", "asparagus",
                   "cabbage", "greens", "beans", "onion"],
    "salad":      ["tomato", "lettuce", "cucumber", "olives"],
    "beans":      ["black bean"],
    "soup":       ["soup"],
    "bread":      ["bread", "toasted bread", "cereals"],
    "fruit":      ["banana", "strawberry", "apple", "grape", "melon", "pineapple",
                   "lime", "orange", "watermelon", "blueberries"],
    "dessert":    ["cake", "gelatin"],
    "cheese":     ["cheese"],
    "eggs":       ["fried egg", "scrambled eggs", "scrambled eggs with bacon",
                   "omelet", "boiled egg"],
    "chicken":    ["chicken", "chicken steak", "turkey steak", "breaded"],
    "meat":       ["stewed veal", "pork", "pork loin", "grilled steak", "steak",
                   "steaks with mushrooms", "minced meat", "meatballs", "cutlet",
                   "grilled chop", "ham", "bacon", "chorizo", "pork belly",
                   "pork intestines"],
    "fish":       ["salmon", "tuna", "fried cod", "fish hake", "fish", "mussel",
                   "tuna with mushrooms", "tuna with pasta"],
}
DROP = {"plate", "fork", "knife", "spoon", "cup", "water cup", "coffee cup", "soupbowl",
        "bowl", "board", "garbage", "bones", "coffee", "water"}
CLASSES = list(GROUPS)
LOOKUP = {orig: g for g, origs in GROUPS.items() for orig in origs}
PLATE_LIKE = {"plate", "soupbowl", "bowl", "board"}   # saved to dataset/plates/ (the
MIN_AREA = 0.0015                                      # generator refills eaten food from these)          # ignore tiny food fragments (< 0.15 % of the image)


def poly_area(vals):
    xs, ys = vals[0::2], vals[1::2]
    n = len(xs)
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="unzipped Roboflow export (folder with data.yaml)")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--max-side", type=int, default=1280)
    a = ap.parse_args()

    yml = glob.glob(os.path.join(a.src, "**", "data.yaml"), recursive=True)
    if not yml:
        sys.exit("data.yaml not found - is this the YOLOv8 export?")
    root = os.path.dirname(yml[0])
    names = yaml.safe_load(open(yml[0]))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    unknown = [n for n in names if n.lower() not in LOOKUP and n.lower() not in DROP]
    if unknown:
        print("  WARNING: classes not in GROUPS or DROP (ignored):", unknown)
    to_group = {i: (CLASSES.index(LOOKUP[n.lower()]) if n.lower() in LOOKUP else None)
                for i, n in enumerate(names)}

    out = a.out
    for d in ("before", "labels", "empty", "plates"):
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)
        os.makedirs(os.path.join(out, d))
    open(os.path.join(out, "classes.txt"), "w").write("\n".join(CLASSES) + "\n")

    rows, cnt, n_food, n_empty = [], Counter(), 0, 0
    for im in sorted(glob.glob(os.path.join(root, "**", "images", "*"), recursive=True)):
        if os.path.splitext(im)[1].lower() not in (".jpg", ".jpeg", ".png"):
            continue
        split = os.path.basename(os.path.dirname(os.path.dirname(im)))
        lab = os.path.join(os.path.dirname(os.path.dirname(im)), "labels",
                           os.path.splitext(os.path.basename(im))[0] + ".txt")
        lines, dishes, plates = [], set(), []
        if os.path.exists(lab):
            for line in open(lab):
                p = line.split()
                if len(p) < 7:                       # need a polygon (>= 3 points)
                    continue
                if names[int(float(p[0]))].lower() in PLATE_LIKE:
                    plates.append("0 " + " ".join(p[1:]))
                    continue
                g = to_group.get(int(float(p[0])))
                vals = [float(v) for v in p[1:]]
                if g is None or poly_area(vals) < MIN_AREA:
                    continue
                lines.append(f"{g} " + " ".join(p[1:]))
                dishes.add(CLASSES[g])
        img = cv2.imread(im)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = a.max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        if lines:
            n_food += 1
            name = f"img_{n_food:03d}"
            cv2.imwrite(os.path.join(out, "before", name + ".jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            open(os.path.join(out, "labels", name + ".txt"), "w").write("\n".join(lines) + "\n")
            if plates:
                open(os.path.join(out, "plates", name + ".txt"), "w").write("\n".join(plates) + "\n")
            cnt.update(dishes)
        else:
            n_empty += 1
            name = f"empty_{n_empty:03d}"
            cv2.imwrite(os.path.join(out, "empty", name + ".jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append([name + ".jpg", os.path.basename(im), split, ";".join(sorted(dishes))])

    with open(os.path.join(out, "source_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_name", "original_file", "roboflow_split", "dishes"])
        w.writerows(rows)
    print(f"{n_food} photos with food -> {out}/before + {out}/labels")
    print(f"{n_empty} photos without food -> {out}/empty (finished trays)")
    print("photos per dish:", ", ".join(f"{c} {cnt[c]}" for c in CLASSES))


if __name__ == "__main__":
    main()
