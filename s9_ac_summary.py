import json
import os
import sys
from collections import defaultdict

ORDER = ["none", "uint8", "jpeg75", "resize32"]


def load(d):
    p = os.path.join(d, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def cells(dirs, eps):
    """(transform) -> list of (missed_count, n_targets, acc_others, psnr, seed)"""
    out = defaultdict(list)
    meta = defaultdict(list)
    for d in dirs:
        r = load(d)
        if r is None:
            print("missing", d)
            continue
        T = r["classes"].index(r["target"])
        s = r["cfg"]["seed"]
        for t in r["trials"]:
            if t["eps255"] != eps or t["n_poison"] != 50:
                continue
            miss = sum(1 for p in t["pred"] if p != T)
            out["main"].append((miss, len(t["pred"]), t["acc_others"], t["psnr_db_used"], s))
            for x in t["transforms"]:
                miss = sum(1 for p in x["pred"] if p != T)
                out[x["transform"]].append((miss, len(x["pred"]), x["acc_others"], x["psnr_db"], s))
            if t["cosine"] is not None:
                meta["cos"].append(t["cosine"])
            if t.get("base_sel_cosine"):
                meta["selcos"].append(t["base_sel_cosine"])
    return out, meta


def report(name, dirs, eps=8.0):
    c, meta = cells(dirs, eps)
    if not c:
        return
    print("==", name, "eps", eps)
    if meta.get("cos"):
        print("   crafting cosine %.4f to %.4f" % (min(meta["cos"]), max(meta["cos"])))
    if meta.get("selcos"):
        print("   base selection cosine: chosen %.4f, all %.4f (means over trials)" % (
            sum(a for a, _ in meta["selcos"]) / len(meta["selcos"]),
            sum(b for _, b in meta["selcos"]) / len(meta["selcos"])))
    print("   transform | missed/total | per seed | acc_others | PSNR dB")
    for k in sorted(c, key=lambda k: ORDER.index(k) if k in ORDER else 99):
        v = c[k]
        per = defaultdict(lambda: [0, 0])
        for m, n, _, _, s in v:
            per[s][0] += m
            per[s][1] += n
        ps = " ".join("s%d %d/%d" % (s, per[s][0], per[s][1]) for s in sorted(per))
        print("   %-9s | %d/%d | %s | %.3f | %.2f" % (
            k, sum(m for m, *_ in v), sum(n for _, n, *_ in v), ps,
            sum(a for _, _, a, _, _ in v) / len(v), sum(p for *_, p, _ in v) / len(v)))


if __name__ == "__main__":
    S = [0, 1, 2]
    report("A Malimg restarts 1, 100 steps", ["out_s9m_r1"])
    report("A Malimg restarts 4, 100 steps", ["out_s9m_r4"])
    report("A Malimg restarts 8, 100 steps", ["out_s9m_r8"])
    report("A Malimg restarts 1, 250 steps", ["out_s9m_s250"])
    report("A Malimg control eps 0", ["out_s9m_r1"], eps=0.0)
    for ds in ("mal", "isic"):
        for arm in ("rand", "sel"):
            report("C %s %s" % (ds, arm), ["out_s9o_%s_%s_s%d" % (ds, arm, s) for s in S])
            report("C %s %s control eps 0" % (ds, arm), ["out_s9o_%s_%s_s%d" % (ds, arm, s) for s in S], eps=0.0)
    for d in sys.argv[1:]:
        report(d, [d])
