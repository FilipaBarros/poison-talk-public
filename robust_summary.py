import json
import os
import sys
from collections import defaultdict

ORDER = ["none", "uint8", "jpeg90", "jpeg75", "jpeg50", "resize32"]

for name, dirs in [("Malimg gray", ["out_robust_malimg_s%d" % s for s in range(3)] + ["out_robust_malimg_ctrl_s%d" % s for s in range(3)]),
                   ("ISIC", ["out_robust_isic_s0", "out_robust_isic_ctrl_s0", "out_robust_isic_s1", "out_robust_isic_s2"])] + \
                  [(d, [d]) for d in sys.argv[1:]]:
    cells = defaultdict(list)
    for d in dirs:
        p = os.path.join(d, "results.json")
        if not os.path.exists(p):
            print("missing", p)
            continue
        r = json.load(open(p))
        for t in r["trials"]:
            k = (t["eps255"], t["n_poison"])
            cells[k + ("none",)].append((t["success"], t["acc_others"], t["psnr_db_used"], r["cfg"]["seed"]))
            for x in t["transforms"]:
                cells[k + (x["transform"],)].append((x["success"], x["acc_others"], x["psnr_db"], r["cfg"]["seed"]))
    print("==", name)
    print("eps n transform | missed/total | acc_others | PSNR dB | seeds")
    for k in sorted(cells, key=lambda k: (k[0], k[1], ORDER.index(k[2]) if k[2] in ORDER else 99)):
        v = cells[k]
        print("%g %d %-8s | %d/%d | %.3f | %.2f | %s" % (
            k[0], k[1], k[2], round(sum(a for a, _, _, _ in v)), len(v), sum(b for _, b, _, _ in v) / len(v),
            sum(c for _, _, c, _ in v) / len(v), sorted({s for *_, s in v})))
