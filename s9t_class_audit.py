"""Per-class duplicate audit of Malimg, using the same sampling the loader does, plus the
training-split size and a feature-degeneracy check on the candidate base classes."""
import os
import random
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from poison import Net, load_folder, split, train, get_features

ROOT, SIZE, CAP, SEED = "malimg_all", 64, 300, 0
cls = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))

X, y, classes = load_folder(ROOT, SIZE, CAP, SEED)
Xtr, ytr, Xte, yte = split(X, y, 0.25, SEED)

rows = []
for c in cls:
    ci = classes.index(c)
    fs = sorted(f for f in os.listdir(os.path.join(ROOT, c))
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    rng = random.Random(SEED)
    fs2 = list(fs)
    rng.shuffle(fs2)
    fs2 = fs2[:CAP]
    a = np.stack([np.asarray(Image.open(os.path.join(ROOT, c, f)).convert("L").resize((SIZE, SIZE), Image.BILINEAR),
                             dtype=np.uint8) for f in fs2])
    nd = len(np.unique(a.reshape(len(fs2), -1), axis=0))
    ntr = int((ytr == ci).sum())
    nte = int((yte == ci).sum())
    rows.append((c, len(fs), len(fs2), nd, nd / len(fs2), ntr, nte))

print("%-22s %7s %8s %9s %7s %7s %6s" % ("class", "files", "sampled", "distinct", "share", "train", "test"))
for r in rows:
    print("%-22s %7d %8d %9d %6.1f%% %7d %6d" % (r[0], r[1], r[2], r[3], 100 * r[4], r[5], r[6]))
print("\nclasses below 90%% distinct:")
for r in rows:
    if r[4] < 0.9:
        print("  %-22s %6.1f%% distinct" % (r[0], 100 * r[4]))

cand = [r for r in rows if r[3] == r[2]]
cand.sort(key=lambda r: (-r[5], -r[1]))
print("\nfully distinct classes, by training-split size then files:")
for r in cand[:6]:
    print("  %-22s train %d, files %d" % (r[0], r[5], r[1]))

sur = train(Net("resnet18", len(classes), True), Xtr, ytr, 5, 1e-3, 64, "cpu", "linear", seed=SEED)
print("\nfeature degeneracy check (pairwise cosine of frozen feature directions, training split):")
for c in [cand[0][0], cand[1][0], "Yuner.A"]:
    ci = classes.index(c)
    fb = get_features(sur, Xtr[(ytr == ci).nonzero().flatten()], "cpu")
    fd = F.normalize(fb, dim=1)
    s = fd @ fd.t()
    n = len(fb)
    off = s.fill_diagonal_(0).sum() / (n * n - n)
    print("  %-22s n=%4d  norm min %.3f max %.3f  pairwise cosine mean %.4f" % (c, n, fb.norm(dim=1).min(), fb.norm(dim=1).max(), off))
