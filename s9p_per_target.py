import csv
import json
import os
from collections import defaultdict

TF = ["none", "uint8", "jpeg75", "resize32"]
RUNS = [("A", "restarts1_steps100", "out_s9m_r1"), ("A", "restarts4_steps100", "out_s9m_r4"),
        ("A", "restarts8_steps100", "out_s9m_r8"), ("A", "restarts1_steps250", "out_s9m_s250")]
for ds in ("mal", "isic"):
    for arm in ("rand", "sel"):
        for s in (0, 1, 2):
            RUNS.append(("C", "%s_%s" % (ds, arm), "out_s9o_%s_%s_s%d" % (ds, arm, s)))

rows = []
for part, label, d in RUNS:
    p = os.path.join(d, "results.json")
    if not os.path.exists(p):
        print("missing", d)
        continue
    r = json.load(open(p))
    T = r["classes"].index(r["target"])
    for t in r["trials"]:
        if t["n_poison"] != 50:
            continue
        byt = {x["transform"]: x["pred"] for x in t["transforms"]}
        for i, (idx, pred, conf) in enumerate(zip(t["target_test_idx"], t["pred"], t["clean_conf_true"])):
            row = {"part": part, "setting": label, "run": d, "seed": r["cfg"]["seed"],
                   "eps255": t["eps255"], "trial": t["trial"], "target_test_idx": idx,
                   "clean_conf_true_class": round(conf, 4), "missed_main": int(pred != T)}
            for k in TF:
                row["missed_" + k] = int(byt[k][i] != T) if k in byt else ""
            rows.append(row)

with open("results/per_target.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote results/per_target.csv, %d rows" % len(rows))

BINS = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]
for eps in (8.0, 0.0):
    print("\n== miss rate by clean-model confidence on the true class, eps %g" % eps)
    print("confidence | targets | " + " | ".join(TF))
    for lo, hi in BINS:
        sel = [r for r in rows if r["eps255"] == eps and lo <= r["clean_conf_true_class"] < hi]
        if not sel:
            continue
        cols = []
        for k in TF:
            v = [r["missed_" + k] for r in sel if r["missed_" + k] != ""]
            cols.append("%d/%d" % (sum(v), len(v)) if v else "-")
        print("%.2f to %.2f | %d | %s" % (lo, hi, len(sel), " | ".join(cols)))
