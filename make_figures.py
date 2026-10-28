"""
Make slide figures from a results folder written by run_experiment.py.

  python make_figures.py out_malware
Writes into the same folder:
  fig_poison_examples.png   clean image, poisoned image, difference amplified 20x
  fig_recall_vs_poison.png  target-class recall on the clean test set vs number of poisons, per eps
  fig_features.png          2D PCA of target and base features: clean model vs poisoned victim
  fig_confusion.png         confusion matrices: clean model vs the strongest-attack victim
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

d = sys.argv[1]
R = json.load(open(os.path.join(d, "results.json")))
E = torch.load(os.path.join(d, "examples.pt"))
classes, T, B = R["classes"], R["classes"].index(R["target"]), R["classes"].index(R["base"])

# 1. poison examples
Xb, Xp, eps = E["examples"]
k = min(4, len(Xb))
fig, ax = plt.subplots(3, k, figsize=(2.6 * k, 8))
ax = np.atleast_2d(ax).reshape(3, -1)
for i in range(k):
    ax[0, i].imshow(Xb[i].permute(1, 2, 0).numpy())
    ax[1, i].imshow(Xp[i].permute(1, 2, 0).numpy())
    diff = (Xp[i] - Xb[i]).permute(1, 2, 0).numpy()
    ax[2, i].imshow(np.clip(0.5 + 20 * diff, 0, 1))
    for r in range(3):
        ax[r, i].axis("off")
ax[0, 0].set_title("clean", loc="left")
ax[1, 0].set_title(f"poisoned (eps={eps}/255)", loc="left")
ax[2, 0].set_title("difference x20", loc="left")
plt.tight_layout()
plt.savefig(os.path.join(d, "fig_poison_examples.png"), dpi=200)
plt.close()

# 2. recall vs number of poisons
fig, ax = plt.subplots(figsize=(6, 4))
ax.axhline(R["clean"]["recall"][T], color="gray", ls="--", label="no poisoning")
for e in sorted({r["eps255"] for r in R["runs"]}):
    rs = sorted([r for r in R["runs"] if r["eps255"] == e], key=lambda r: r["n_poison"])
    ax.plot([r["n_poison"] for r in rs], [r["recall"][T] for r in rs], marker="o", label=f"eps={e:g}/255")
ax.set_xlabel("number of poisoned training images")
ax.set_ylabel(f"recall on '{R['target']}' (clean test set)")
ax.set_ylim(0, 1.02)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(d, "fig_recall_vs_poison.png"), dpi=200)
plt.close()

# 3. features, PCA fitted on the clean model features
F = E["features"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a_, (t, b, title) in zip(ax, [("clean_model_target", "clean_model_base", "clean model"),
                                  ("victim_target", "victim_base", "poisoned victim")]):
    # each model has its own feature space, so each panel gets its own PCA
    A = torch.cat([F[t], F[b]]).numpy()
    mu = A.mean(0)
    _, _, Vt = np.linalg.svd(A - mu, full_matrices=False)
    proj = lambda Z, mu=mu, Vt=Vt: (Z.numpy() - mu) @ Vt[:2].T
    P = proj(F[t]); Q = proj(F[b])
    a_.scatter(Q[:, 0], Q[:, 1], s=8, alpha=0.5, label=f"{R['base']} (test)")
    a_.scatter(P[:, 0], P[:, 1], s=8, alpha=0.5, label=f"{R['target']} (test)")
    if title != "clean model":
        S = proj(F["victim_poison"])
        a_.scatter(S[:, 0], S[:, 1], s=14, marker="x", color="k", label="poisons")
    a_.set_title(title)
    a_.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(d, "fig_features.png"), dpi=200)
plt.close()

# 4. confusion matrices
worst = min(R["runs"], key=lambda r: r["recall"][T])
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
for a_, (cm, title) in zip(ax, [(R["clean"]["confusion"], "clean model"),
                                (worst["confusion"], f"poisoned (eps={worst['eps255']:g}, n={worst['n_poison']})")]):
    cm = np.array(cm, dtype=float)
    cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    a_.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    a_.set_title(title)
    a_.set_xlabel("predicted")
    a_.set_ylabel("true")
    if len(classes) <= 12:
        a_.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
        a_.set_yticks(range(len(classes)), classes, fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(d, "fig_confusion.png"), dpi=200)
plt.close()
print("wrote figures to", d)
