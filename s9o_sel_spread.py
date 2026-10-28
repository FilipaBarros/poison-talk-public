import torch
from poison import Net, load_folder, split, train, predict, select_base

for data, mx, tgt, base in [("malimg_all", 300, "VB.AT", "Yuner.A"), ("isic_folder", 1500, "malignant", "benign")]:
    X, y, classes = load_folder(data, 64, mx, 0)
    Xtr, ytr, Xte, yte = split(X, y, 0.25, 0)
    T, B = classes.index(tgt), classes.index(base)
    sur = train(Net("resnet18", len(classes), True), Xtr, ytr, 5, 1e-3, 64, "cpu", "linear", seed=0)
    clean = train(Net("resnet18", len(classes), True), Xtr, ytr, 5, 1e-3, 64, "cpu", "linear", seed=1)
    pool = ((yte == T) & (predict(clean, Xte, "cpu") == T)).nonzero().flatten()
    g = torch.Generator().manual_seed(0)
    picks = pool[torch.randperm(len(pool), generator=g)][:25].view(5, 5)
    base_all = (ytr == B).nonzero().flatten()
    pick, cos = select_base(sur, Xtr[base_all], B, Xte[picks[0]], "cpu", 50)
    q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    print("%s: n_base=%d cosine min %.4f q25 %.4f median %.4f q75 %.4f max %.4f std %.5f | top50 mean %.4f all mean %.4f"
          % (data, len(cos), *torch.quantile(cos, q).tolist(), cos.std().item(), cos[pick].mean().item(), cos.mean().item()), flush=True)
