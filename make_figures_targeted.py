"""
Slide figures for the targeted-sample experiment (run_experiment.py --n-targets k).

  python make_figures_targeted.py OUT_PREFIX FOLDER [FOLDER ...]

Trials from all folders are pooled (for example the eps 8/16 run and the eps 0/2/4 run).
Writes:
  OUT_PREFIX_success_vs_eps.png   share of target images missed vs eps, one line per poison count
  OUT_PREFIX_poison_examples.png  target image, clean base images, poisoned versions, difference x20
  OUT_PREFIX_defenses.png         per defense: share of poisons caught vs share of clean data removed
                                  (only if a folder has defense results)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SERIES = ["#2a78d6", "#eb6834"]  # reference palette slots 1-2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})

prefix, folders = sys.argv[1], sys.argv[2:]
R = [json.load(open(os.path.join(f, "results.json"))) for f in folders]
trials = [t for r in R for t in r["trials"]]
target, base = R[0]["target"], R[0]["base"]

# 1. success vs eps
ns = sorted({t["n_poison"] for t in trials})
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.grid(axis="y", color=GRID, lw=1)
ax.set_axisbelow(True)
for i, n in enumerate(ns[:2]):
    eps = sorted({t["eps255"] for t in trials if t["n_poison"] == n})
    ys = [np.mean([t["success"] for t in trials if t["n_poison"] == n and t["eps255"] == e]) for e in eps]
    k = [sum(1 for t in trials if t["n_poison"] == n and t["eps255"] == e) for e in eps]
    frac = trials[[t["n_poison"] for t in trials].index(n)]["frac_of_base_class"]
    lab = "%d poisons (%.0f%% of %s)" % (n, 100 * frac, base)
    ax.plot(eps, ys, color=SERIES[i], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=lab)
ax.set_xlabel("perturbation budget eps (L-inf, out of 255)")
ax.set_ylabel("share of target images missed")
ax.set_ylim(-0.03, 1.05)
ax.set_xticks(sorted({t["eps255"] for t in trials}))
ax.set_title("%s samples classified as %s after retraining (%d trials per point)" % (target, base, min(k)),
             fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(prefix + "_success_vs_eps.png", dpi=200)
plt.close()

# 2. poison examples
ex = next((torch.load(os.path.join(f, "examples.pt")) for f in folders
           if os.path.exists(os.path.join(f, "examples.pt"))), None)
if ex is not None:
    Xb, Xp, e = ex["examples"]
    k = min(4, len(Xb))
    fig, ax = plt.subplots(3, k + 1, figsize=(2.3 * (k + 1), 7.2))
    for a in ax.flat:
        a.axis("off")
    ax[0, 0].imshow(ex["target_images"][0].permute(1, 2, 0).numpy())
    ax[0, 0].set_title("target (%s)" % target, fontsize=9, loc="left")
    for i in range(k):
        d = (Xp[i] - Xb[i]).permute(1, 2, 0).numpy()
        ax[0, i + 1].imshow(Xb[i].permute(1, 2, 0).numpy())
        ax[1, i + 1].imshow(Xp[i].permute(1, 2, 0).numpy())
        ax[2, i + 1].imshow(np.clip(0.5 + 20 * d, 0, 1))
    ax[0, 1].set_title("clean %s" % base, fontsize=9, loc="left")
    ax[1, 1].set_title("poisoned, eps=%g/255" % e, fontsize=9, loc="left")
    ax[2, 1].set_title("difference x20", fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(prefix + "_poison_examples.png", dpi=200)
    plt.close()

# 3. defenses
ds = [t for t in trials if "defenses" in t]
if ds:
    names = [d["defense"] for d in ds[0]["defenses"]]
    caught = [np.mean([d["tpr"] for t in ds for d in t["defenses"] if d["defense"] == n]) for n in names]
    removed = [np.mean([d["fpr"] for t in ds for d in t["defenses"] if d["defense"] == n]) for n in names]
    after = [np.mean([d["target_missed_after_filter"] for t in ds for d in t["defenses"] if d["defense"] == n]) for n in names]
    before = np.mean([t["success"] for t in ds])
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    b1 = ax.bar(x - w / 2 - 0.01, caught, w, color=SERIES[0], label="poisons caught")
    b2 = ax.bar(x + w / 2 + 0.01, removed, w, color=SERIES[1], label="clean training images removed")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate("%.0f%%" % (100 * b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, ["%s\nmissed after: %.0f%%" % (n.replace("deep_knn", "Deep k-NN").replace("spectral@", "spectral "), 100 * a)
                      for n, a in zip(names, after)], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("share")
    ax.set_title("Defenses on the poisoned training set (targets missed before filtering: %.0f%%)" % (100 * before),
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(prefix + "_defenses.png", dpi=200)
    plt.close()
print("wrote", prefix + "_*.png")
