"""With --restarts 1 and --select-base off, craft_gradmatch must reproduce the loop exactly as it
stood before this change. The old loop is copied verbatim below and both are run in one process."""
import math, torch
import torch.nn.functional as F
torch.set_num_threads(1)
import poison
from poison import Net, load_folder, split, train, craft_gradmatch, diff_transform


def craft_old(surrogate, X_base, base_label, X_target, eps, steps, step_size, device, mode,
              schedule="decay", augment=True, gray=False, craft_transforms=None, lowres=0, quantize=False):
    surrogate.eval().to(device)
    params = list(surrogate.head.parameters()) if mode == "linear" else [p for p in surrogate.parameters()]
    xt = X_target.to(device)
    yt = torch.full((len(xt),), base_label, dtype=torch.long, device=device)
    g_adv = torch.autograd.grad(F.cross_entropy(surrogate(xt), yt), params)
    g_adv = torch.cat([g.flatten() for g in g_adv]).detach()
    xb = X_base.to(device)
    yb = torch.full((len(xb),), base_label, dtype=torch.long, device=device)
    shape = xb[:, :1] if gray else xb
    if lowres:
        shape = shape[:, :, :lowres, :lowres]
    delta = (torch.rand_like(shape) * 2 - 1) * eps * 0.1
    up = (lambda d: F.interpolate(d, size=xb.shape[-2:], mode="bilinear", align_corners=False)) if lowres else (lambda d: d)
    best, best_cos = delta.clone(), -2.0
    for t in range(steps):
        delta.requires_grad_(True)
        xp = (xb + up(delta)).clamp(0, 1)
        if augment:
            if torch.rand(1).item() < 0.5:
                xp = xp.flip(3)
            if torch.rand(1).item() < 0.5:
                xp = xp.flip(2)
        if craft_transforms:
            xp = diff_transform(xp, craft_transforms[torch.randint(len(craft_transforms), (1,)).item()], gray=gray)
        g_p = torch.autograd.grad(F.cross_entropy(surrogate(xp), yb), params, create_graph=True)
        g_p = torch.cat([g.flatten() for g in g_p])
        cos = F.cosine_similarity(g_p, g_adv, dim=0)
        g, = torch.autograd.grad(1 - cos, delta)
        if cos.item() > best_cos:
            best_cos, best = cos.item(), delta.detach().clone()
        a = step_size if schedule == "constant" else step_size * 0.5 * (1 + math.cos(math.pi * t / steps))
        delta = (delta.detach() - a * g.sign()).clamp(-eps, eps)
    xp = (xb + up(best)).clamp(0, 1).detach().cpu()
    if quantize:
        xp = torch.round(xp * 255).clamp(0, 255) / 255.0
    return xp, best_cos


CT = ["none", "jpeg90", "jpeg75", "jpeg50", "resize32"]
for data, mx, tgt, base, gray in [("malimg_all", 300, "VB.AT", "Yuner.A", True),
                                  ("isic_folder", 1500, "malignant", "benign", False)]:
    X, y, classes = load_folder(data, 64, mx, 0)
    Xtr, ytr, Xte, yte = split(X, y, 0.25, 0)
    T, B = classes.index(tgt), classes.index(base)
    s = train(Net("resnet18", len(classes), True), Xtr, ytr, 5, 1e-3, 64, "cpu", "linear", seed=0)
    Xb = Xtr[(ytr == B).nonzero().flatten()[:20]]
    Xt = Xte[yte == T][:3]
    for ct in (None, CT):
        a = {}
        for tag, fn in [("old", craft_old), ("new", craft_gradmatch)]:
            torch.manual_seed(7)
            a[tag] = fn(s, Xb, B, Xt, 8 / 255, 30, 2 / 255, "cpu", "linear", gray=gray,
                        craft_transforms=ct, quantize=bool(ct))
        print("%s ct=%s: max pixel diff %.1f gray levels, identical=%s, cos old %.10f new %.10f" % (
            data, bool(ct), (a["old"][0] - a["new"][0]).abs().max() * 255,
            torch.equal(a["old"][0], a["new"][0]), a["old"][1], a["new"][1]), flush=True)
