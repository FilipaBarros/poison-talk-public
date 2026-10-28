# Upper bound per target class (run from the project folder) (linear head on frozen pretrained resnet18 features):
# how many exact copies of a target test image's feature, labelled base, flip it?
import argparse, sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
p = argparse.ArgumentParser()
p.add_argument("--data", default="malimg_all")
p.add_argument("--base", default="Yuner.A")
p.add_argument("--targets", nargs="*", default=None, help="target classes to test (default: all but base)")
p.add_argument("--size", type=int, default=64)
p.add_argument("--max-per-class", type=int, default=300)
p.add_argument("--n-test-images", type=int, default=3)
p.add_argument("--copies", type=int, nargs="+", default=[5, 25, 50])
a = p.parse_args()
from poison import load_folder, split, Net, get_features
torch.manual_seed(0)
X, y, classes = load_folder(a.data, a.size, a.max_per_class, 0)
Xtr, ytr, Xte, yte = split(X, y, 0.25, 0)
B = classes.index(a.base)
net = Net("resnet18", len(classes), True)
ftr, fte = get_features(net, Xtr, "cpu"), get_features(net, Xte, "cpu")

def head(f, lab, epochs=5, lr=1e-3, bs=64, seed=1):
    torch.manual_seed(seed)
    h = torch.nn.Linear(f.shape[1], len(classes)); opt = torch.optim.Adam(h.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(f)).split(bs):
            loss = F.cross_entropy(h(f[idx]), lab[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return h

pred0 = head(ftr, ytr)(fte).argmax(1)
print("%-16s %5s %6s %7s  %s" % ("target", "ntrain", "spread", "recall", "flipped at m=%s (%d test images each)" % ("/".join(map(str, a.copies)), a.n_test_images)))
for T, name in enumerate(classes):
    if T == B or (a.targets and name not in a.targets):
        continue
    fT = ftr[ytr == T]
    spread = (fT - fT.mean(0)).norm(dim=1).mean().item()
    rec = (pred0[yte == T] == T).float().mean().item()
    tidx = ((yte == T) & (pred0 == T)).nonzero().flatten()[:a.n_test_images]
    res = []
    for m in a.copies:
        fl = 0
        for t in tidx:
            h = head(torch.cat([ftr, fte[t].repeat(m, 1)]), torch.cat([ytr, torch.full((m,), B)]))
            fl += int(h(fte[t:t+1]).argmax(1).item() != T)
        res.append("%d/%d" % (fl, len(tidx)))
    print("%-16s %5d %6.2f %7.3f  %s" % (name, len(fT), spread, rec, " ".join(res)), flush=True)
