"""
Download ISIC 2020 Challenge training images from the ISIC Archive API (v2) into the
ImageFolder layout run_experiment.py expects:  OUT/benign/*.jpg, OUT/malignant/*.jpg

Uses the 256 px thumbnails (the experiments resize to 64 or 128 px anyway).
Also writes OUT/manifest.csv with each image's id, label, license and attribution.

  python download_isic.py --out isic_folder --malignant all --benign 1500
"""
import argparse
import csv
import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.isic-archive.com/api/v2"

p = argparse.ArgumentParser()
p.add_argument("--out", default="isic_folder")
p.add_argument("--malignant", default="all", help="number of malignant images, or 'all'")
p.add_argument("--benign", type=int, default=1500)
p.add_argument("--collection", default=None, help="collection id; default: find 'Challenge 2020' training")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %d for %s\n%s" % (e.code, url, e.read().decode(errors="replace")[:1000]))


def find_collection():
    url = API + "/collections/?limit=100"
    found = []
    while url:
        d = get(url)
        for c in d["results"]:
            n = c["name"].lower()
            if "2020" in n and "challenge" in n:
                found.append((c["id"], c["name"]))
        url = d.get("next")
    for cid, name in found:
        print("collection %s: %s" % (cid, name))
    train = [cid for cid, name in found if "train" in name.lower()]
    if not train:
        raise SystemExit("No ISIC 2020 training collection found; pass --collection <id> from the list above.")
    return train[0]


def search(label, collection):
    # the archive's current diagnosis field is diagnosis_1 (Benign / Malignant / Indeterminate);
    # older releases used benign_malignant. Use whichever the server accepts.
    for query in ('diagnosis_1:"%s"' % label.capitalize(), "benign_malignant:%s" % label):
        url = API + "/images/search/?" + urllib.parse.urlencode(
            {"query": query, "collections": collection, "limit": 100})
        try:
            with urllib.request.urlopen(url, timeout=60):
                break
        except urllib.error.HTTPError as e:
            print("query %r rejected (HTTP %d): %s" % (query, e.code, e.read().decode(errors="replace")[:300]))
    else:
        raise SystemExit("No label query accepted; the ISIC API rejected both diagnosis field names.")
    print("query:", query)
    out = []
    while url:
        d = get(url)
        out.extend(d["results"])
        url = d.get("next")
        print("  %s: %d listed" % (label, len(out)), end="\r")
    print()
    return out


collection = a.collection or find_collection()
print("using collection", collection)
rng = random.Random(a.seed)
rows = []
for label, want in (("malignant", a.malignant), ("benign", a.benign)):
    imgs = search(label, collection)
    rng.shuffle(imgs)
    if want != "all":
        imgs = imgs[:int(want)]
    d = os.path.join(a.out, label)
    os.makedirs(d, exist_ok=True)
    for i, im in enumerate(imgs):
        path = os.path.join(d, im["isic_id"] + ".jpg")
        if not os.path.exists(path):
            urllib.request.urlretrieve(im["files"]["thumbnail_256"]["url"], path)
        rows.append({"isic_id": im["isic_id"], "label": label, "license": im.get("copyright_license", ""),
                     "attribution": im.get("attribution", "")})
        print("  %s: %d/%d downloaded" % (label, i + 1, len(imgs)), end="\r")
    print()

with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["isic_id", "label", "license", "attribution"])
    w.writeheader()
    w.writerows(rows)
lic = {}
for r in rows:
    lic[r["license"]] = lic.get(r["license"], 0) + 1
print("done:", {l: sum(r["label"] == l for r in rows) for l in ("malignant", "benign")}, "licenses:", lic)
