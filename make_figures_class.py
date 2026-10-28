"""
Slide figures for the class-level experiment repeated over seeds (run_experiment.py --victim-reps r).

  python make_figures_class.py OUT_PREFIX FOLDER [FOLDER ...]

One folder per seed, each with runs at eps 0 and the largest eps, at the largest poison count.
Writes:
  OUT_PREFIX_recall_by_seed.png   target recall per seed: unperturbed replacements vs poisons
  OUT_PREFIX_poison_examples.png  clean base images, poisoned versions, difference x20
  OUT_PREFIX_defenses.png         per defense: poisons caught vs clean removed, and target recall after
                                  filtering (poisoned set and unperturbed control); only if folders have defenses
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SERIES = ["#2a78d6", "#eb6834", "#8a8984"]  # reference palette slots 1-2, neutral for the control
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})

prefix, folders = sys.argv[1], sys.argv[2:]
R = [json.load(open(os.path.join(f, "results.json"))) for f in folders]
target, base = R[0]["target"], R[0]["base"]
n = max(r["n_poison"] for r in R[0]["runs"])
eps = max(r["eps255"] for r in R[0]["runs"])
frac = next(r["frac_of_base_class"] for r in R[0]["runs"] if r["n_poison"] == n)
cap = R[0]["cfg"].get("target_train_cap") or 0


def recall(res, e):
    return next(r["recall_target_mean"] for r in res["runs"] if r["eps255"] == e and r["n_poison"] == n)


ctrl = [recall(r, 0) for r in R]
pois = [recall(r, eps) for r in R]
seeds = [r["cfg"]["seed"] for r in R]
reps = R[0]["cfg"].get("victim_reps", 1)

# 1. recall per seed, control vs poisoned
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.grid(axis="y", color=GRID, lw=1)
ax.set_axisbelow(True)
x = np.arange(len(R))
for i in x:
    ax.plot([i, i], [ctrl[i], pois[i]], color=GRID, lw=2, zorder=1)
ax.scatter(x, ctrl, s=70, color=SERIES[2], edgecolor=SURFACE, lw=2, zorder=2,
           label="%d unperturbed %s images (mean %.2f)" % (n, base, np.mean(ctrl)))
ax.scatter(x, pois, s=70, color=SERIES[1], edgecolor=SURFACE, lw=2, zorder=3,
           label="%d poisoned, eps=%g/255 (mean %.2f)" % (n, eps, np.mean(pois)))
ax.set_xticks(x, ["seed %d" % s for s in seeds])
ax.set_ylim(0, 1.05)
ax.set_ylabel("%s recall on clean test set" % target)
title = "%s recall after retraining\n(%d%% of %s replaced%s; mean of %d victims per point)" % (
    target, round(100 * frac), base, ", %d %s training images" % (cap, target) if cap else "", reps)
ax.set_title(title, fontsize=9, loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower left")
plt.tight_layout()
plt.savefig(prefix + "_recall_by_seed.png", dpi=200)
plt.close()

# 2. poison examples (clean, poisoned, difference)
ex = torch.load(os.path.join(folders[0], "examples.pt"))
Xb, Xp, e = ex["examples"]
k = min(4, len(Xb))
fig, ax = plt.subplots(3, k, figsize=(2.3 * k, 7.2))
for a in ax.flat:
    a.axis("off")
for i in range(k):
    d = (Xp[i] - Xb[i]).permute(1, 2, 0).numpy()
    ax[0, i].imshow(Xb[i].permute(1, 2, 0).numpy())
    ax[1, i].imshow(np.clip(Xp[i].permute(1, 2, 0).numpy(), 0, 1))
    ax[2, i].imshow(np.clip(0.5 + 20 * d, 0, 1))
ax[0, 0].set_title("clean %s" % base, fontsize=9, loc="left")
ax[1, 0].set_title("poisoned, eps=%g/255" % e, fontsize=9, loc="left")
ax[2, 0].set_title("difference x20", fontsize=9, loc="left")
plt.tight_layout()
plt.savefig(prefix + "_poison_examples.png", dpi=200)
plt.close()

# 3. defenses, averaged over seeds
D = [r for r in R if "defenses" in r]
if D:
    names = [d["defense"] for d in D[0]["defenses"]]

    def avg(key, field, name):
        return np.mean([d[field] for r in D for d in r.get(key, []) if d["defense"] == name] or [np.nan])

    caught = [avg("defenses", "tpr", m) for m in names]
    removed = [avg("defenses", "fpr", m) for m in names]
    after = [avg("defenses", "recall_target_after_filter", m) for m in names]
    after_ctrl = [avg("defenses_control", "recall_target_after_filter", m) for m in names]
    before = np.mean([recall(r, eps) for r in D])
    before_ctrl = np.mean([recall(r, 0) for r in D])
    labels = [m.replace("deep_knn", "Deep k-NN").replace("spectral@", "spectral ") for m in names]
    xs = np.arange(len(names))
    w = 0.36
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for a in (a1, a2):
        a.grid(axis="y", color=GRID, lw=1)
        a.set_axisbelow(True)
    b1 = a1.bar(xs - w / 2 - 0.01, caught, w, color=SERIES[0], label="poisons caught")
    b2 = a1.bar(xs + w / 2 + 0.01, removed, w, color=SERIES[1], label="clean training images removed")
    for bars in (b1, b2):
        for b in bars:
            a1.annotate("%.0f%%" % (100 * b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    a1.set_xticks(xs, labels, fontsize=9)
    a1.set_ylim(0, 1.1)
    a1.set_ylabel("share")
    a1.set_title("What each filter removes", fontsize=10, loc="left")
    a1.legend(frameon=False, fontsize=9, loc="upper left")

    a2.bar(xs - w / 2 - 0.01, after_ctrl, w, color=SERIES[2], label="unperturbed set, after filter")
    a2.bar(xs + w / 2 + 0.01, after, w, color=SERIES[1], label="poisoned set, after filter")
    a2.axhline(before_ctrl, color=SERIES[2], lw=1.5, ls="--", label="unperturbed set, no filter (%.2f)" % before_ctrl)
    a2.axhline(before, color=SERIES[1], lw=1.5, ls="--", label="poisoned set, no filter (%.2f)" % before)
    a2.set_xticks(xs, labels, fontsize=9)
    a2.set_ylim(0, 1.25)
    a2.set_ylabel("%s recall after retraining" % target)
    a2.set_title("%s recall after filtering and retraining (mean of %d seeds)" % (target, len(D)),
                 fontsize=10, loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(prefix + "_defenses.png", dpi=200)
    plt.close()
print("wrote", prefix + "_*.png")
