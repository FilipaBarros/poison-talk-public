import json
from collections import defaultdict

FILTERS = ["deep_knn", "spectral@0.05", "spectral@0.1", "spectral@0.2"]
acc = defaultdict(list)
pre = []
for s in range(3):
    r = json.load(open("out_s9v_def_s%d/results.json" % s))
    T = r["classes"].index(r["target"])
    for t in r["trials"]:
        if "defenses" not in t:
            continue
        pre.append((sum(1 for q in t["pred"] if q != T), len(t["pred"])))
        for d in t["defenses"]:
            acc[d["defense"]].append(d)

print("missed before filtering: %d/%d over %d trials" % (sum(a for a, _ in pre), sum(b for _, b in pre), len(pre)))
print("%-14s %8s %9s %14s %10s" % ("filter", "caught", "clean cut", "missed after", "acc after"))
for k in FILTERS:
    v = acc[k]
    if not v:
        continue
    n = len(v)
    print("%-14s %7.0f%% %8.0f%% %10.1f/%d %10.3f" % (
        k, 100 * sum(d["tpr"] for d in v) / n, 100 * sum(d["fpr"] for d in v) / n,
        sum(d["target_missed_after_filter"] for d in v), n,
        sum(d["acc_others_after_filter"] for d in v) / n))

print("\nper-seed missed after filtering, out of 5:")
for k in FILTERS:
    per = defaultdict(float)
    for s in range(3):
        r = json.load(open("out_s9v_def_s%d/results.json" % s))
        for t in r["trials"]:
            for d in t.get("defenses", []):
                if d["defense"] == k:
                    per[s] += d["target_missed_after_filter"]
    print("  %-14s %s" % (k, " ".join("%.0f" % per[s] for s in range(3))))

print("\npoisons caught per seed, out of 50 per trial:")
for k in FILTERS:
    for s in range(3):
        r = json.load(open("out_s9v_def_s%d/results.json" % s))
        c = [(d["poisons_flagged"], d["poisons_total"]) for t in r["trials"] for d in t.get("defenses", [])
             if d["defense"] == k]
        print("  %-14s s%d  %d/%d" % (k, s, sum(a for a, _ in c), sum(b for _, b in c)))
