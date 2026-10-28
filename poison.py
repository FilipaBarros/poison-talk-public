"""
Clean-label feature-collision data poisoning, L-inf bounded.

Attack (class-level variant of Poison Frogs, Shafahi et al. 2018):
  The attacker wants test images of a TARGET class T to be classified as a BASE class B.
  Poisons are images of class B (label stays B, so they look correctly labelled).
  Each poison x_b is perturbed so its features move toward the mean feature of class T:

      delta* = argmin_{||delta||_inf <= eps}  || phi(x_b + delta) - mu_T ||_2^2
      mu_T   = mean over target-class training images of phi(x_t)
      x_p    = clip(x_b + delta, 0, 1)

  phi is the penultimate layer of a surrogate model the attacker trains on clean data.
  After the victim retrains on clean + poisoned data, T-like features sit inside the
  region labelled B, and recall on class T drops on a clean test set.

Victim training modes:
  linear   : backbone frozen (pretrained), only the final layer is retrained
  finetune : whole network retrained, starting from the same pretrained backbone
  scratch  : whole network retrained from random init (hardest for the attacker)
"""
import hashlib
import io
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image


# ----------------------------------------------------------------------------- data

def list_classes(root):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def load_folder(root, size, max_per_class=None, seed=0):
    """root/<class>/*.{png,jpg,jpeg}. Returns X float32 [N,3,size,size] in [0,1], y int64, class names."""
    rng = random.Random(seed)
    classes = list_classes(root)
    xs, ys = [], []
    for ci, c in enumerate(classes):
        files = sorted(f for f in os.listdir(os.path.join(root, c))
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            im = Image.open(os.path.join(root, c, f)).convert("RGB").resize((size, size), Image.BILINEAR)
            xs.append(np.asarray(im, dtype=np.uint8))
            ys.append(ci)
    X = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 255.0
    return X, torch.tensor(ys, dtype=torch.long), classes


def split(X, y, test_frac, seed):
    g = torch.Generator().manual_seed(seed)
    tr, te = [], []
    for c in y.unique():
        idx = (y == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        k = max(1, int(len(idx) * test_frac))
        te.append(idx[:k])
        tr.append(idx[k:])
    tr, te = torch.cat(tr), torch.cat(te)
    return X[tr], y[tr], X[te], y[te]


def cap_class(X, y, c, k, seed):
    """Keep only k training images of class c (simulates a new family with few known samples)."""
    idx = (y == c).nonzero().flatten()
    g = torch.Generator().manual_seed(seed + 123)
    drop = idx[torch.randperm(len(idx), generator=g)][k:]
    keep = torch.ones(len(y), dtype=torch.bool)
    keep[drop] = False
    return X[keep], y[keep]


# ----------------------------------------------------------------------------- model

_FEAT_CACHE = None


def enable_feature_cache(on=True):
    global _FEAT_CACHE
    _FEAT_CACHE = {} if on else None


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Net(nn.Module):
    """backbone -> feature vector (phi) -> linear head. Input in [0,1]; normalisation is inside."""

    def __init__(self, arch, n_classes, pretrained):
        super().__init__()
        if arch == "smallcnn":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            dim = 128
        elif arch in ("resnet18", "resnet50"):
            import torchvision
            if arch == "resnet18":
                m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
            else:
                m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
            dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            raise ValueError(arch)
        self.head = nn.Linear(dim, n_classes)
        self._cache_tag = ("%s:%d:%d" % (arch, dim, int(pretrained))).encode()
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def features(self, x):
        if _FEAT_CACHE is None or (torch.is_grad_enabled() and x.requires_grad):
            return self.backbone((x - self.mean) / self.std)
        # the tag keeps two architectures from sharing an entry; in linear mode the backbone is
        # frozen and identical across models, so one entry is valid for every victim of a run
        keys = [hashlib.blake2b(r.detach().cpu().numpy().tobytes() + self._cache_tag, digest_size=16).digest() for r in x]
        miss = [i for i, k in enumerate(keys) if k not in _FEAT_CACHE]
        if miss:
            with torch.no_grad():
                f = self.backbone((x[miss] - self.mean) / self.std).cpu()
            for i, fi in zip(miss, f):
                _FEAT_CACHE[keys[i]] = fi
        return torch.stack([_FEAT_CACHE[k] for k in keys]).to(x.device)

    def forward(self, x):
        return self.head(self.features(x))


def train(model, X, y, epochs, lr, bs, device, mode="finetune", augment=True, seed=0):
    torch.manual_seed(seed)
    if mode == "linear":
        for p in model.backbone.parameters():
            p.requires_grad = False
        params = model.head.parameters()
    else:
        params = model.parameters()
    opt = torch.optim.Adam(params, lr=lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True)
    model.to(device)
    for ep in range(epochs):
        model.train()
        if mode == "linear":
            model.backbone.eval()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if augment:
                if torch.rand(1).item() < 0.5:
                    xb = xb.flip(3)
                if torch.rand(1).item() < 0.5:
                    xb = xb.flip(2)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    for p in model.parameters():
        p.requires_grad = True
    return model


@torch.no_grad()
def predict(model, X, device, bs=128):
    model.eval().to(device)
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device)).argmax(1).cpu())
    return torch.cat(out)


@torch.no_grad()
def get_features(model, X, device, bs=128):
    model.eval().to(device)
    if len(X) == 0:
        return torch.empty(0, 1)
    out = []
    for i in range(0, len(X), bs):
        out.append(model.features(X[i:i + bs].to(device)).cpu())
    return torch.cat(out)


def evaluate(model, X, y, n_classes, device):
    pred = predict(model, X, device)
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    recall = (cm.diag().float() / cm.sum(1).clamp(min=1)).tolist()
    return {"acc": (pred == y).float().mean().item(), "recall": recall, "confusion": cm.tolist()}


# ----------------------------------------------------------------------------- attack

def craft_poisons(surrogate, X_base, feat_targets, eps, steps, step_size, device, bs=64, schedule="decay",
                  gray=False):
    """PGD on the feature-collision objective with an L-inf bound of eps (pixel scale 0..1).
    feat_targets: [D] (one shared target feature) or [N, D] (one target feature per poison).
    schedule: "constant" keeps step_size fixed and returns the last iterate; "decay" lowers the
    step with a cosine schedule from step_size to ~0 and returns the best iterate per poison."""
    surrogate.eval().to(device)
    if feat_targets.dim() == 1:
        feat_targets = feat_targets.view(1, -1).expand(len(X_base), -1)
    out = []
    for i in range(0, len(X_base), bs):
        xb = X_base[i:i + bs].to(device)
        mu = feat_targets[i:i + bs].to(device)
        # gray=True: one channel, broadcast to R, G and B, so grayscale images stay grayscale
        delta = (torch.rand_like(xb[:, :1] if gray else xb) * 2 - 1) * eps * 0.1
        best = delta.clone()
        best_loss = torch.full((len(xb),), float("inf"), device=device)
        for t in range(steps):
            delta.requires_grad_(True)
            xp = (xb + delta).clamp(0, 1)
            per = ((surrogate.features(xp) - mu) ** 2).sum(1)
            g, = torch.autograd.grad(per.mean(), delta)
            with torch.no_grad():
                better = per < best_loss
                best[better] = delta[better]
                best_loss = torch.where(better, per, best_loss)
            a = step_size if schedule == "constant" else step_size * 0.5 * (1 + math.cos(math.pi * t / steps))
            delta = (delta.detach() - a * g.sign()).clamp(-eps, eps)
        if schedule == "constant":
            best = delta  # previous behaviour: last iterate
        out.append((xb + best).clamp(0, 1).detach().cpu())
    return torch.cat(out)


def craft_gradmatch(surrogate, X_base, base_label, X_target, eps, steps, step_size, device, mode,
                    schedule="decay", augment=True, gray=False, craft_transforms=None, lowres=0,
                    quantize=False, restarts=1):
    """Gradient matching (Witches' Brew, Geiping et al. 2021), L-inf bounded, clean label.
    The attacker's goal is the training loss L_adv = CE(f(X_target), base): target-class images
    classified as base. Poisons keep their true label (base) and are perturbed so that their
    training gradient points the same way as the gradient of L_adv:

        delta* = argmin_{||delta||_inf <= eps}  1 - cos( grad_theta CE(f(x_b + delta), base),
                                                       grad_theta L_adv )

    theta = parameters the victim trains (only the head in linear mode, everything otherwise).
    All poisons are optimised jointly. Returns the poisons and the best cosine reached."""
    surrogate.eval().to(device)
    params = list(surrogate.head.parameters()) if mode == "linear" else [p for p in surrogate.parameters()]
    xt = X_target.to(device)
    yt = torch.full((len(xt),), base_label, dtype=torch.long, device=device)
    g_adv = torch.autograd.grad(F.cross_entropy(surrogate(xt), yt), params)
    g_adv = torch.cat([g.flatten() for g in g_adv]).detach()

    xb = X_base.to(device)
    yb = torch.full((len(xb),), base_label, dtype=torch.long, device=device)
    # gray=True: one channel, broadcast to R, G and B, so grayscale images stay grayscale
    # lowres>0: the variable is a lowres x lowres perturbation, upsampled to full size. Bilinear
    # upsampling is a weighted average, so clamping the variable to eps keeps the full-size one there.
    shape = xb[:, :1] if gray else xb
    if lowres:
        shape = shape[:, :, :lowres, :lowres]
    up = (lambda d: F.interpolate(d, size=xb.shape[-2:], mode="bilinear", align_corners=False)) if lowres else (lambda d: d)
    best, best_cos = None, -2.0
    # restarts>1: independent random inits, the best cosine over all of them is kept
    for r in range(max(restarts, 1)):
        delta = (torch.rand_like(shape) * 2 - 1) * eps * 0.1
        if best is None:
            best = delta.clone()
        for t in range(steps):
            delta.requires_grad_(True)
            xp = (xb + up(delta)).clamp(0, 1)
            if augment:  # same random flips as victim training, so the poisons survive them
                if torch.rand(1).item() < 0.5:
                    xp = xp.flip(3)
                if torch.rand(1).item() < 0.5:
                    xp = xp.flip(2)
            if craft_transforms:  # one transform per step, so the poisons are optimised against all of them
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
    if quantize:  # what the attacker actually ships: whole pixel values
        xp = torch.round(xp * 255).clamp(0, 255) / 255.0
    return xp, best_cos


def select_base(surrogate, X_base, base_label, X_target, device, k):
    """Pick the k base-class images whose head gradient grad_theta CE(f(x), base) has the highest
    cosine with the attacker's target gradient grad_theta CE(f(X_target), base). Head only, which
    is what the victim trains in linear mode. Returns the chosen indices into X_base."""
    surrogate.eval().to(device)
    with torch.no_grad():
        pb = F.softmax(surrogate(X_base.to(device)), 1)
        fb = get_features(surrogate, X_base, device)
        pt = F.softmax(surrogate(X_target.to(device)), 1)
        ft = get_features(surrogate, X_target, device)
    rb, rt = pb.clone(), pt.clone()
    rb[:, base_label] -= 1  # softmax minus the one-hot label, the residual the gradient scales
    rt[:, base_label] -= 1
    g_t = torch.cat([(rt.t() @ ft).flatten(), rt.sum(0)])
    g_b = torch.cat([torch.einsum("nc,nd->ncd", rb, fb).flatten(1), rb], 1)
    cos = F.cosine_similarity(g_b, g_t.unsqueeze(0), dim=1)
    return cos.topk(min(k, len(cos))).indices.cpu(), cos.cpu()


# ----------------------------------------------------------------------------- differentiable transforms
# Used inside the crafting loop only. The evaluation transforms stay PIL (transform_images).

_JPEG_Y = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
_JPEG_C = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]], dtype=torch.float32)


def _qtables(quality):
    s = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return (torch.floor((_JPEG_Y * s + 50) / 100).clamp(1, 255),
            torch.floor((_JPEG_C * s + 50) / 100).clamp(1, 255))


def _dct_matrix():
    x = torch.arange(8, dtype=torch.float32)
    D = torch.cos((2 * x + 1).view(1, -1) * x.view(-1, 1) * math.pi / 16) * math.sqrt(2.0 / 8)
    D[0] /= math.sqrt(2.0)
    return D


def _ste_round(x):
    """Rounding whose gradient is the identity (straight-through estimator)."""
    return x + (torch.round(x) - x).detach()


def _blockwise(x, q):
    """8x8 DCT, divide by the quantization table, round (straight through), invert. x in -128..127."""
    N, C, H, W = x.shape
    D = _dct_matrix().to(x.device)
    b = x.view(N, C, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 3, 5)
    b = D @ b @ D.t()
    b = _ste_round(b / q) * q
    b = D.t() @ b @ D
    return b.permute(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)


def diff_jpeg(x, quality, gray=False):
    """Differentiable JPEG round trip on images in [0,1]. No chroma subsampling."""
    qy, qc = _qtables(quality)
    qy, qc = qy.to(x.device), qc.to(x.device)
    x = _ste_round(x.clamp(0, 1) * 255) / 255.0  # the encoder sees whole pixel values
    if gray:
        y = _blockwise(x[:, :1] * 255 - 128, qy)
        return ((y + 128) / 255.0).clamp(0, 1).expand(-1, 3, -1, -1)
    r, g, b = x[:, :1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    y = _blockwise(y * 255 - 128, qy)
    c = _blockwise(torch.cat([cb, cr], 1) * 255 - 128, qc)
    y, cb, cr = (y + 128) / 255.0, (c[:, :1] + 128) / 255.0, (c[:, 1:] + 128) / 255.0
    return torch.cat([y + 1.402 * (cr - 0.5),
                      y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5),
                      y + 1.772 * (cb - 0.5)], 1).clamp(0, 1)


def diff_transform(x, name, gray=False):
    if name == "none":
        return x
    if name.startswith("jpeg"):
        return diff_jpeg(x, int(name[4:]), gray=gray)
    if name.startswith("resize"):
        s = int(name[6:])
        y = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True)
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
    raise ValueError(name)


def transform_images(X, name):
    out = []
    for x in X:
        im = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))
        if name.startswith("jpeg"):
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=int(name[4:]))
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
        elif name.startswith("resize"):
            s = int(name[6:])
            im = im.resize((s, s), Image.BILINEAR).resize(im.size, Image.BILINEAR)
        elif name.startswith("blur"):
            from PIL import ImageFilter
            im = im.filter(ImageFilter.GaussianBlur(radius=float(name[4:])))
        elif name not in ("uint8", "none"):
            raise ValueError(name)
        out.append(np.asarray(im, dtype=np.uint8))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0


def psnr(a, b):
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    return (10 * torch.log10(1.0 / mse)).mean().item()


# ----------------------------------------------------------------------------- defenses
# Both defenses see only the (poisoned) training set and the model trained on it.

def flag_deep_knn(feats, labels, k=15):
    """Deep k-NN (Peri et al. 2020): flag a sample if fewer than half of its k nearest
    neighbours in feature space share its label."""
    f = F.normalize(feats, dim=1)
    sim = f @ f.T
    sim.fill_diagonal_(-2)
    nn_idx = sim.topk(k, dim=1).indices
    agree = (labels[nn_idx] == labels.view(-1, 1)).float().mean(1)
    return agree < 0.5


def flag_spectral(feats, labels, remove_frac):
    """Spectral signatures (Tran et al. 2018), per class: score = squared projection of the
    centred features on the top singular vector; flag the top `remove_frac` of each class."""
    flag = torch.zeros(len(labels), dtype=torch.bool)
    for c in labels.unique():
        idx = (labels == c).nonzero().flatten()
        Z = feats[idx] - feats[idx].mean(0)
        _, _, Vt = torch.linalg.svd(Z, full_matrices=False)
        score = (Z @ Vt[0]) ** 2
        k = int(round(remove_frac * len(idx)))
        if k > 0:
            flag[idx[score.topk(k).indices]] = True
    return flag


def defense_report(name, flag, is_poison):
    tp = (flag & is_poison).sum().item()
    fp = (flag & ~is_poison).sum().item()
    return {"defense": name, "poisons_flagged": tp, "poisons_total": int(is_poison.sum()),
            "clean_flagged": fp, "clean_total": int((~is_poison).sum()),
            "tpr": tp / max(1, int(is_poison.sum())), "fpr": fp / max(1, int((~is_poison).sum()))}


# ----------------------------------------------------------------------------- experiment

def run(cfg):
    """cfg keys: data, size, max_per_class, test_frac, arch, pretrained, mode, epochs, lr, bs,
    target, base, eps_list (0..255 scale), n_poison_list, attack_steps, seed, out, device"""
    t0 = time.time()
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["out"], exist_ok=True)
    X, y, classes = load_folder(cfg["data"], cfg["size"], cfg.get("max_per_class"), cfg["seed"])
    Xtr, ytr, Xte, yte = split(X, y, cfg["test_frac"], cfg["seed"])
    nc = len(classes)
    T, B = classes.index(cfg["target"]), classes.index(cfg["base"])
    if cfg.get("target_train_cap"):
        Xtr, ytr = cap_class(Xtr, ytr, T, cfg["target_train_cap"], cfg["seed"])
        print("target class capped to %d training images" % int((ytr == T).sum()))
    print(f"classes={nc} train={len(Xtr)} test={len(Xte)} target={cfg['target']} base={cfg['base']} device={device}")

    def new_model():
        return Net(cfg["arch"], nc, cfg["pretrained"])

    # attacker's surrogate: trained on clean data (same architecture and starting point as the victim)
    surrogate = train(new_model(), Xtr, ytr, cfg["epochs"], cfg["lr"], cfg["bs"], device, cfg["mode"], augment=cfg.get("augment", True), seed=cfg["seed"])
    fT = get_features(surrogate, Xtr[ytr == T], device)
    mu_T = fT.mean(0)
    spread_T = (fT - mu_T).norm(dim=1).mean().item()  # mean distance of target training features to mu_T

    # victim trained on clean data only (baseline)
    clean = train(new_model(), Xtr, ytr, cfg["epochs"], cfg["lr"], cfg["bs"], device, cfg["mode"], augment=cfg.get("augment", True), seed=cfg["seed"] + 1)
    clean_eval = evaluate(clean, Xte, yte, nc, device)
    print("clean   acc=%.3f recall[target]=%.3f" % (clean_eval["acc"], clean_eval["recall"][T]))

    base_idx = (ytr == B).nonzero().flatten()
    g = torch.Generator().manual_seed(cfg["seed"])
    base_idx = base_idx[torch.randperm(len(base_idx), generator=g)]
    # what each poison is pushed toward: the target-class mean, or one target training image each
    pairing = cfg.get("pairing", "mean")
    if pairing == "mean":
        feat_targets = mu_T.view(1, -1).expand(len(base_idx), -1)
    elif pairing == "random":
        feat_targets = fT[torch.randint(len(fT), (len(base_idx),), generator=g)]
    else:
        raise ValueError(pairing)

    results = {"classes": classes, "target": cfg["target"], "base": cfg["base"], "clean": clean_eval,
               "cfg": {k: v for k, v in cfg.items()}, "runs": []}
    saved_examples = None
    for eps255 in cfg["eps_list"]:
        eps = eps255 / 255.0
        schedule = cfg.get("step_schedule", "decay")
        step = max(eps / (10 if schedule == "constant" else 4), 1e-4)
        max_n = min(max(cfg["n_poison_list"]), len(base_idx))
        Xb = Xtr[base_idx[:max_n]]
        if cfg.get("objective", "collision") == "gradmatch":
            Xp_all, cos = craft_gradmatch(surrogate, Xb, B, Xtr[ytr == T], eps, cfg["attack_steps"], step, device,
                                          cfg["mode"], schedule=schedule, augment=cfg.get("augment", True),
                                          gray=cfg.get("gray_delta", False))
            print("eps=%g gradient matching: best cosine %.3f" % (eps255, cos))
            results.setdefault("gradmatch_cosine", []).append({"eps255": eps255, "cosine": cos})
        else:
            Xp_all = craft_poisons(surrogate, Xb, feat_targets[:max_n], eps, cfg["attack_steps"], step, device,
                                   schedule=schedule, gray=cfg.get("gray_delta", False))
        linf = (Xp_all - Xb).abs().amax().item() * 255
        ps = psnr(Xp_all, Xb)
        # diagnostics: do the poisons reach the target region in the surrogate's feature space?
        fb, fp = get_features(surrogate, Xb, device), get_features(surrogate, Xp_all, device)
        ft = feat_targets[:max_n]
        diag = {"dist_to_goal_before": (fb - ft).norm(dim=1).mean().item(),
                "dist_to_goal_after": (fp - ft).norm(dim=1).mean().item(),
                "dist_to_muT_before": (fb - mu_T).norm(dim=1).mean().item(),
                "dist_to_muT_after": (fp - mu_T).norm(dim=1).mean().item(),
                "target_spread": spread_T,
                "surrogate_says_target": (predict(surrogate, Xp_all, device) == T).float().mean().item(),
                "victim_clean_says_target": (predict(clean, Xp_all, device) == T).float().mean().item()}
        print("eps=%g diag: dist to goal %.2f -> %.2f | dist to mu_T %.2f -> %.2f | target spread %.2f | "
              "poisons labelled target by surrogate %.2f, by clean victim %.2f" % (
                  eps255, diag["dist_to_goal_before"], diag["dist_to_goal_after"], diag["dist_to_muT_before"],
                  diag["dist_to_muT_after"], spread_T, diag["surrogate_says_target"], diag["victim_clean_says_target"]))
        results.setdefault("diagnostics", []).append({"eps255": eps255, **diag})
        for n in cfg["n_poison_list"]:
            n = min(n, max_n)
            keep = torch.ones(len(Xtr), dtype=torch.bool)
            keep[base_idx[:n]] = False  # replace the clean originals by their poisoned versions
            Xmix = torch.cat([Xtr[keep], Xp_all[:n]])
            ymix = torch.cat([ytr[keep], torch.full((n,), B)])
            # victim_reps > 1: several victims (different init and data order) per setting, recall averaged
            reps = []
            for r in range(cfg.get("victim_reps", 1)):
                victim = train(new_model(), Xmix, ymix, cfg["epochs"], cfg["lr"], cfg["bs"], device, cfg["mode"],
                               augment=cfg.get("augment", True), seed=cfg["seed"] + 1 + 1000 * r)
                reps.append(evaluate(victim, Xte, yte, nc, device))
            ev = reps[0]
            rec_mean = sum(e["recall"][T] for e in reps) / len(reps)
            acc_mean = sum(e["acc"] for e in reps) / len(reps)
            frac_base = n / int((ytr == B).sum())
            print("eps=%g n=%d (%.1f%% of base) acc=%.3f recall[target]=%.3f%s  Linf=%.1f/255 PSNR=%.1fdB" % (
                eps255, n, 100 * frac_base, acc_mean, rec_mean,
                " (mean of %d victims: %s)" % (len(reps), " ".join("%.3f" % e["recall"][T] for e in reps)) if len(reps) > 1 else "",
                linf, ps))
            results["runs"].append({"eps255": eps255, "n_poison": n, "frac_of_base_class": frac_base,
                                    "linf255": linf, "psnr_db": ps, **ev, "recall_target_mean": rec_mean,
                                    "acc_mean": acc_mean, "recall_target_reps": [e["recall"][T] for e in reps]})
            if (not cfg.get("skip_defense") and n > 0 and n == max(cfg["n_poison_list"])
                    and eps255 in (0, max(cfg["eps_list"]))):
                # defenses on the strongest configuration, then retrain on what they keep
                is_poison = torch.cat([torch.zeros(int(keep.sum()), dtype=torch.bool), torch.ones(n, dtype=torch.bool)])
                fv = get_features(victim, Xmix, device)
                flags = {"deep_knn": flag_deep_knn(fv, ymix)}
                for fr in cfg.get("spectral_fracs", [0.05, 0.1, 0.2]):
                    flags["spectral@%g" % fr] = flag_spectral(fv, ymix, remove_frac=fr)
                # eps 0: the same filters on unperturbed replacements, a control for what filtering alone does
                dkey = "defenses" if eps255 > 0 else "defenses_control"
                results[dkey] = []
                for name, fl in flags.items():
                    rep = defense_report(name, fl, is_poison)
                    kept = ~fl
                    # retrain as many victims as above on what the filter keeps, so before/after are comparable
                    reps2 = []
                    for r in range(cfg.get("victim_reps", 1)):
                        m2 = train(new_model(), Xmix[kept], ymix[kept], cfg["epochs"], cfg["lr"], cfg["bs"], device,
                                   cfg["mode"], augment=cfg.get("augment", True), seed=cfg["seed"] + 1 + 1000 * r)
                        reps2.append(evaluate(m2, Xte, yte, nc, device))
                    acc2 = sum(e["acc"] for e in reps2) / len(reps2)
                    rec2 = sum(e["recall"][T] for e in reps2) / len(reps2)
                    rep.update({"acc_after_filter": acc2, "recall_target_after_filter": rec2,
                                "recall_target_before_filter": rec_mean,
                                "target_train_kept": int(((ymix == T) & kept).sum()),
                                "target_train_total": int((ymix == T).sum())})
                    results[dkey].append(rep)
                    print(("control " if eps255 == 0 else "") + "defense %-20s flagged %d/%d poisons, %d/%d clean | after retrain: acc=%.3f recall[target]=%.3f (before %.3f)" % (
                        name, rep["poisons_flagged"], rep["poisons_total"], rep["clean_flagged"], rep["clean_total"],
                        acc2, rec2, rec_mean))
            if saved_examples is None or (eps255 == max(cfg["eps_list"]) and n == max(cfg["n_poison_list"])):
                saved_examples = (Xb[:8].clone(), Xp_all[:8].clone(), eps255)
                feats = {"clean_model_target": get_features(clean, Xte[yte == T], device),
                         "clean_model_base": get_features(clean, Xte[yte == B], device),
                         "victim_target": get_features(victim, Xte[yte == T], device),
                         "victim_base": get_features(victim, Xte[yte == B], device),
                         "victim_poison": get_features(victim, Xp_all[:n], device)}
    results["seconds"] = time.time() - t0
    with open(os.path.join(cfg["out"], "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    torch.save({"examples": saved_examples, "features": feats}, os.path.join(cfg["out"], "examples.pt"))
    return results


# ----------------------------------------------------------------------------- targeted-sample experiment

def run_targeted(cfg):
    """Targeted-sample variant (the setting of Poison Frogs and Witches' Brew): the attacker picks
    n_targets specific test images of the target class (for example their own malware samples) and
    wants those images classified as the base class after the victim retrains. Poisons are crafted
    against those images only. Repeated over `trials` random choices of target images.
    Success for a trial = share of its target images the poisoned victim no longer labels as target."""
    t0 = time.time()
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["out"], exist_ok=True)
    X, y, classes = load_folder(cfg["data"], cfg["size"], cfg.get("max_per_class"), cfg["seed"])
    Xtr, ytr, Xte, yte = split(X, y, cfg["test_frac"], cfg["seed"])
    nc = len(classes)
    T, B = classes.index(cfg["target"]), classes.index(cfg["base"])
    if cfg.get("target_train_cap"):
        Xtr, ytr = cap_class(Xtr, ytr, T, cfg["target_train_cap"], cfg["seed"])
        print("target class capped to %d training images" % int((ytr == T).sum()))
    if cfg.get("base_collapse"):
        # every base-class training image becomes a copy of the class medoid under the frozen
        # backbone, so the class holds one image repeated, as Yuner.A does naturally
        bi = (ytr == B).nonzero().flatten()
        fb = get_features(Net(cfg["arch"], nc, cfg["pretrained"]), Xtr[bi], device)
        pick = bi[(fb - fb.mean(0)).norm(dim=1).argmin()].item()
        Xtr = Xtr.clone()
        Xtr[bi] = Xtr[pick].clone()
        print("base class collapsed to %d copies of training image %d" % (len(bi), pick))
    aug = cfg.get("augment", True)
    fast = cfg.get("fast", False)
    if fast:
        assert cfg["mode"] == "linear" and cfg["pretrained"], "--fast needs linear mode with a pretrained backbone"
        enable_feature_cache()
    print(f"classes={nc} train={len(Xtr)} test={len(Xte)} target={cfg['target']} base={cfg['base']} device={device}")

    def new_model():
        return Net(cfg["arch"], nc, cfg["pretrained"])

    surrogate = train(new_model(), Xtr, ytr, cfg["epochs"], cfg["lr"], cfg["bs"], device, cfg["mode"], augment=aug, seed=cfg["seed"])
    clean = train(new_model(), Xtr, ytr, cfg["epochs"], cfg["lr"], cfg["bs"], device, cfg["mode"], augment=aug, seed=cfg["seed"] + 1)
    clean_eval = evaluate(clean, Xte, yte, nc, device)
    print("clean   acc=%.3f recall[target]=%.3f" % (clean_eval["acc"], clean_eval["recall"][T]))
    with torch.no_grad():
        clean_prob = torch.cat([F.softmax(clean(Xte[i:i + 128].to(device)), 1).cpu()
                                for i in range(0, len(Xte), 128)])

    # candidate targets: target-class test images the clean victim gets right
    pool = ((yte == T) & (predict(clean, Xte, device) == T)).nonzero().flatten()
    g = torch.Generator().manual_seed(cfg["seed"])
    k, trials = cfg.get("n_targets", 1), cfg.get("trials", 5)
    if len(pool) < k * trials:
        print("warning: only %d correctly classified target test images; drawing from all target test images" % len(pool))
        pool = (yte == T).nonzero().flatten()
    trials = min(trials, len(pool) // k)
    picks = pool[torch.randperm(len(pool), generator=g)][:k * trials].view(trials, k)
    base_all = (ytr == B).nonzero().flatten()
    others = torch.ones(len(Xte), dtype=torch.bool)
    others[picks.flatten()] = False  # accuracy is reported on the test images that are not targets

    results = {"classes": classes, "target": cfg["target"], "base": cfg["base"], "clean": clean_eval,
               "cfg": {kk: v for kk, v in cfg.items()}, "trials": []}
    saved = {}
    schedule = cfg.get("step_schedule", "decay")
    for eps255 in cfg["eps_list"]:
        eps = eps255 / 255.0
        step = max(eps / (10 if schedule == "constant" else 4), 1e-4)
        for tr in range(trials):
            tidx = picks[tr]
            Xt = Xte[tidx]
            max_n = min(max(cfg["n_poison_list"]), len(base_all))
            perm = base_all[torch.randperm(len(base_all), generator=g)]
            bidx = perm[:max_n]
            sel_cos = None
            if cfg.get("select_base"):
                # the random draw above is still made, so the generator state and the eps 0
                # controls stay comparable with a run that does not select
                pick, allcos = select_base(surrogate, Xtr[base_all], B, Xt, device, max_n)
                bidx = base_all[pick]
                sel_cos = [allcos[pick].mean().item(), allcos.mean().item()]
                print("  base selection: mean gradient cosine %.4f over the %d chosen, %.4f over all %d"
                      % (sel_cos[0], len(pick), sel_cos[1], len(allcos)))
            Xb = Xtr[bidx]
            if fast and eps255 == 0:
                Xp, cos = Xb.clone(), None
            elif cfg.get("objective", "collision") == "gradmatch":
                Xp, cos = craft_gradmatch(surrogate, Xb, B, Xt, eps, cfg["attack_steps"], step, device,
                                          cfg["mode"], schedule=schedule, augment=aug, gray=cfg.get("gray_delta", False),
                                          craft_transforms=cfg.get("craft_transforms"), lowres=cfg.get("lowres", 0),
                                          quantize=bool(cfg.get("craft_transforms") or cfg.get("lowres")),
                                          restarts=cfg.get("restarts", 1))
            else:
                ft = get_features(surrogate, Xt, device).mean(0)
                Xp, cos = craft_poisons(surrogate, Xb, ft, eps, cfg["attack_steps"], step, device, schedule=schedule, gray=cfg.get("gray_delta", False)), None
            for n in cfg["n_poison_list"]:
                if n == 0:
                    continue
                n = min(n, max_n)
                keep = torch.ones(len(Xtr), dtype=torch.bool)
                keep[bidx[:n]] = False
                Xmix = torch.cat([Xtr[keep], Xp[:n]])
                ymix = torch.cat([ytr[keep], torch.full((n,), B)])
                tf_recs = []
                for tf in cfg.get("transforms") or []:
                    Xq = transform_images(Xp[:n], tf)
                    v2 = train(new_model(), torch.cat([Xtr[keep], Xq]), ymix, cfg["epochs"], cfg["lr"], cfg["bs"],
                               device, cfg["mode"], augment=aug, seed=cfg["seed"] + 1)
                    p2, o2 = predict(v2, Xt, device), predict(v2, Xte[others], device)
                    tf_recs.append({"transform": tf, "success": (p2 != T).float().mean().item(),
                                    "pred": p2.tolist(), "to_base": (p2 == B).float().mean().item(),
                                    "acc_others": (o2 == yte[others]).float().mean().item(),
                                    "linf255": (Xq - Xb[:n]).abs().amax().item() * 255, "psnr_db": psnr(Xq, Xb[:n])})
                    print("  transform %-9s eps=%g trial=%d n=%d targets missed %.2f acc_others=%.3f PSNR=%.2fdB" % (
                        tf, eps255, tr, n, tf_recs[-1]["success"], tf_recs[-1]["acc_others"], tf_recs[-1]["psnr_db"]))
                victim = train(new_model(), Xmix, ymix, cfg["epochs"], cfg["lr"], cfg["bs"], device, cfg["mode"],
                               augment=aug, seed=cfg["seed"] + 1)
                pt = predict(victim, Xt, device)
                po = predict(victim, Xte[others], device)
                rec = {"eps255": eps255, "trial": tr, "n_poison": n, "frac_of_base_class": n / len(base_all),
                       "target_test_idx": tidx.tolist(), "pred": pt.tolist(), "cosine": cos,
                       "clean_conf_true": clean_prob[tidx, T].tolist(), "base_sel_cosine": sel_cos,
                       "success": (pt != T).float().mean().item(), "to_base": (pt == B).float().mean().item(),
                       "acc_others": (po == yte[others]).float().mean().item(),
                       "linf255": (Xp - Xb).abs().amax().item() * 255, "psnr_db": psnr(Xp, Xb),
                       "psnr_db_used": psnr(Xp[:n], Xb[:n]), "transforms": tf_recs,
                       "linf255_shipped": (Xp[:n] - Xb[:n]).abs().amax().item() * 255}
                results["trials"].append(rec)
                print("eps=%g trial=%d n=%d (%.1f%% of base) targets missed %.2f (as base %.2f) acc_others=%.3f%s" % (
                    eps255, tr, n, 100 * rec["frac_of_base_class"], rec["success"], rec["to_base"], rec["acc_others"],
                    "" if cos is None else " cos=%.3f" % cos))
                strongest = eps255 == max(cfg["eps_list"]) and n == min(max(cfg["n_poison_list"]), len(base_all))
                if strongest and "examples" not in saved:
                    saved["examples"] = (Xb[:8].clone(), Xp[:8].clone(), eps255)
                    saved["target_images"] = Xt.clone()
                    saved["target_pred"] = pt.tolist()
                if strongest and not cfg.get("skip_defense") and eps255 > 0:
                    # defenses see only the poisoned training set and the victim trained on it
                    is_poison = torch.cat([torch.zeros(int(keep.sum()), dtype=torch.bool), torch.ones(n, dtype=torch.bool)])
                    fv = get_features(victim, Xmix, device)
                    flags = {"deep_knn": flag_deep_knn(fv, ymix)}
                    for fr in cfg.get("spectral_fracs", [0.05, 0.1, 0.2]):
                        flags["spectral@%g" % fr] = flag_spectral(fv, ymix, remove_frac=fr)
                    rec["defenses"] = []
                    for name, fl in flags.items():
                        rep = defense_report(name, fl, is_poison)
                        m2 = train(new_model(), Xmix[~fl], ymix[~fl], cfg["epochs"], cfg["lr"], cfg["bs"], device,
                                   cfg["mode"], augment=aug, seed=cfg["seed"] + 1)
                        p2 = predict(m2, Xt, device)
                        rep.update({"target_missed_after_filter": (p2 != T).float().mean().item(),
                                    "acc_others_after_filter": (predict(m2, Xte[others], device) == yte[others]).float().mean().item()})
                        rec["defenses"].append(rep)
                        print("  defense %-14s caught %d/%d poisons, removed %d/%d clean | after retrain: target missed %.2f acc_others=%.3f" % (
                            name, rep["poisons_flagged"], rep["poisons_total"], rep["clean_flagged"], rep["clean_total"],
                            rep["target_missed_after_filter"], rep["acc_others_after_filter"]))
        for n in cfg["n_poison_list"]:
            rs = [r for r in results["trials"] if r["eps255"] == eps255 and r["n_poison"] == min(n, len(base_all))]
            if rs:
                print("SUMMARY eps=%g n=%d: targets missed %.2f over %d trials, acc_others %.3f" % (
                    eps255, n, sum(r["success"] for r in rs) / len(rs), len(rs),
                    sum(r["acc_others"] for r in rs) / len(rs)))
        ds = [r for r in results["trials"] if r["eps255"] == eps255 and "defenses" in r]
        for name in ([d["defense"] for d in ds[0]["defenses"]] if ds else []):
            reps = [d for r in ds for d in r["defenses"] if d["defense"] == name]
            print("DEFENSE SUMMARY eps=%g %-14s caught %.2f of poisons, removed %.2f of clean, target missed after retrain %.2f (before %.2f)" % (
                eps255, name, sum(d["tpr"] for d in reps) / len(reps), sum(d["fpr"] for d in reps) / len(reps),
                sum(d["target_missed_after_filter"] for d in reps) / len(reps), sum(r["success"] for r in ds) / len(ds)))
    results["seconds"] = time.time() - t0
    with open(os.path.join(cfg["out"], "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    if saved:
        torch.save(saved, os.path.join(cfg["out"], "examples.pt"))
    return results
