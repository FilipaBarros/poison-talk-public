import json, torch
from poison import Net, load_folder, split, train, predict, craft_gradmatch

CT = ["none", "jpeg90", "jpeg75", "jpeg50", "resize32"]
SEED = 0


def dct_mat():
    import math
    n = 8
    m = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            m[k, i] = math.cos(math.pi * (2 * i + 1) * k / (2 * n)) * (math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n))
    return m


D = dct_mat()
uv = torch.arange(8).view(8, 1) + torch.arange(8).view(1, 8)
BANDS = {"low (u+v<=3)": uv <= 3, "mid (4<=u+v<=7)": (uv >= 4) & (uv <= 7), "high (u+v>=8)": uv >= 8}


def band_energy(delta):
    x = delta * 255.0
    n, c, h, w = x.shape
    b = x.reshape(n, c, h // 8, 8, w // 8, 8).permute(0, 1, 2, 4, 3, 5)
    co = D @ b @ D.t()
    e = (co ** 2).sum(dim=(0, 1, 2, 3))
    return {k: (e[m].sum() / e.sum()).item() for k, m in BANDS.items()}


out = []
for data, mx, tgt, base, gray in [("malimg_all", 300, "VB.AT", "Yuner.A", True),
                                  ("isic_folder", 1500, "malignant", "benign", False)]:
    X, y, classes = load_folder(data, 64, mx, SEED)
    Xtr, ytr, Xte, yte = split(X, y, 0.25, SEED)
    T, B = classes.index(tgt), classes.index(base)
    sur = train(Net("resnet18", len(classes), True), Xtr, ytr, 5, 1e-3, 64, "cpu", "linear", seed=SEED)
    clean = train(Net("resnet18", len(classes), True), Xtr, ytr, 5, 1e-3, 64, "cpu", "linear", seed=SEED + 1)
    pool = ((yte == T) & (predict(clean, Xte, "cpu") == T)).nonzero().flatten()
    g = torch.Generator().manual_seed(SEED)
    picks = pool[torch.randperm(len(pool), generator=g)][:5 * 5].view(5, 5)
    base_all = (ytr == B).nonzero().flatten()
    bidx = base_all[torch.randperm(len(base_all), generator=g)][:50]
    Xb, Xt = Xtr[bidx], Xte[picks[0]]
    for name, ct in [("plain", None), ("compression-aware", CT)]:
        Xp, cos = craft_gradmatch(sur, Xb, B, Xt, 8 / 255, 100, 2 / 255, "cpu", "linear", gray=gray,
                                  craft_transforms=ct, quantize=bool(ct))
        d = Xp - Xb
        r = {"data": data, "arm": name, "cos": cos, "n": len(Xp),
             "mean_abs_gray": (d.abs().mean() * 255).item(),
             "linf_gray": (d.abs().amax() * 255).item()}
        r.update(band_energy(d))
        out.append(r)
        print(json.dumps(r), flush=True)
json.dump(out, open("results/dct_bands.json", "w"), indent=1)
