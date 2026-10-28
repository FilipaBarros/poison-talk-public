"""Mean-feature distance from each candidate Malimg base class to the VB.AT target class, using
the frozen pretrained ResNet-18 backbone on the training split. Also the collapsed Allaple.A set.
"""
import torch
import torch.nn.functional as F
from poison import Net, load_folder, split, get_features

ROOT, SIZE, CAP, SEED = "malimg_all", 64, 300, 0
TARGET = "VB.AT"
MIN_TRAIN = 150
FULLY_DISTINCT = ["Agent.FYI", "Allaple.A", "Allaple.L", "Alueron.gen!J", "C2LOP.P", "C2LOP.gen!g",
                  "Dialplatform.B", "Dontovo.A", "Instantaccess", "Lolyda.AA1", "Lolyda.AA2",
                  "Lolyda.AA3", "Malex.gen!J", "Obfuscator.AD", "Skintrim.N", "Swizzor.gen!E",
                  "Swizzor.gen!I", "VB.AT", "Wintrim.BX"]

X, y, classes = load_folder(ROOT, SIZE, CAP, SEED)
Xtr, ytr, _, _ = split(X, y, 0.25, SEED)
net = Net("resnet18", len(classes), True)
T = classes.index(TARGET)
ft = get_features(net, Xtr[(ytr == T).nonzero().flatten()], "cpu")
mt = ft.mean(0)

rows = []
for c in classes:
    ci = classes.index(c)
    idx = (ytr == ci).nonzero().flatten()
    f = get_features(net, Xtr[idx], "cpu")
    m = f.mean(0)
    rows.append((c, len(idx), (m - mt).norm().item(), F.cosine_similarity(m, mt, dim=0).item(),
                 c in FULLY_DISTINCT))

print("%-22s %6s %12s %10s %9s" % ("class", "train", "L2 to VB.AT", "cosine", "distinct"))
for r in sorted(rows, key=lambda r: r[2]):
    print("%-22s %6d %12.3f %10.4f %9s" % (r[0], r[1], r[2], r[3], "yes" if r[4] else "no"))

cand = [r for r in rows if r[4] and r[1] >= MIN_TRAIN and r[0] != TARGET]
cand.sort(key=lambda r: r[2])
print("\ncandidates: 100%% distinct, at least %d training images, not the target" % MIN_TRAIN)
for r in cand:
    print("  %-22s train %d  L2 %.3f  cosine %.4f" % (r[0], r[1], r[2], r[3]))
print("closest to VB.AT: %s (L2 %.3f)" % (cand[0][0], cand[0][2]))
print("farthest from VB.AT: %s (L2 %.3f)" % (cand[-1][0], cand[-1][2]))

print("\ncollapsed sets (every training image replaced by the class medoid):")
for c in ("Yuner.A", "Allaple.A"):
    ci = classes.index(c)
    idx = (ytr == ci).nonzero().flatten()
    f = get_features(net, Xtr[idx], "cpu")
    pick = idx[(f - f.mean(0)).norm(dim=1).argmin()].item()
    fc = get_features(net, Xtr[pick].unsqueeze(0), "cpu")[0]
    print("  %-22s medoid training index %5d  L2 to VB.AT %.3f  cosine %.4f"
          % (c + " collapsed", pick, (fc - mt).norm(), F.cosine_similarity(fc, mt, dim=0)))
