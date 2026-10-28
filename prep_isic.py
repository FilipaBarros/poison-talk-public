"""
Turn an ISIC download (images + one CSV) into the ImageFolder layout run_experiment.py expects:
  OUT/benign/*.jpg, OUT/malignant/*.jpg

Defaults match the ISIC 2020 training CSV (columns image_name, benign_malignant).
Check your CSV header and pass --image-col / --label-col if it differs.
For releases with one column per diagnosis (e.g. MEL, NV, ...), add a column yourself that maps
each image to benign or malignant before running this.

  python prep_isic.py --csv train.csv --images jpeg/train --out isic_folder
"""
import argparse
import csv
import os
import shutil

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--images", required=True)
p.add_argument("--out", required=True)
p.add_argument("--image-col", default="image_name")
p.add_argument("--label-col", default="benign_malignant")
p.add_argument("--ext", default=".jpg")
a = p.parse_args()

counts = {}
with open(a.csv) as f:
    for row in csv.DictReader(f):
        label = row[a.label_col].strip().lower()
        if label not in ("benign", "malignant"):
            continue
        src = os.path.join(a.images, row[a.image_col] + a.ext)
        if not os.path.exists(src):
            continue
        dst_dir = os.path.join(a.out, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst_dir)
        counts[label] = counts.get(label, 0) + 1
print(counts)
