"""Why is the head-gradient cosine constant across the Malimg Yuner.A base class?
Prints the softmax residual, its norm, the feature norm, the split of the gradient between the
weight block and the bias block, and two alternative scores. ISIC is run alongside as the contrast.
"""
import torch
import torch.nn.functional as F
from poison import Net, load_folder, split, train, predict, get_features


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return ((ra - ra.mean()) * (rb - rb.mean())).sum() / (ra.std(unbiased=False) * rb.std(unbiased=False) * len(ra))


def grad_cos(logits_b, fb, logits_t, ft, B, temp=1.0):
    rb = F.softmax(logits_b / temp, 1).clone()
    rt = F.softmax(logits_t / temp, 1).clone()
    rb[:, B] -= 1
    rt[:, B] -= 1
    g_t = torch.cat([(rt.t() @ ft).flatten(), rt.sum(0)])
    g_b = torch.cat([torch.einsum("nc,nd->ncd", rb, fb).flatten(1), rb], 1)
    return F.cosine_similarity(g_b, g_t.unsqueeze(0), dim=1), rb


for data, mx, tgt, base in [("malimg_all", 300, "VB.AT", "Yuner.A"), ("isic_folder", 1500, "malignant", "benign")]:
    X, y, classes = load_folder(data, 64, mx, 0)
    Xtr, ytr, Xte, yte = split(X, y, 0.25, 0)
    T, B = classes.index(tgt), classes.index(base)
    sur = train(Net("resnet18", len(classes), True), Xtr, ytr, 5, 1e-3, 64, "cpu", "linear", seed=0)
    clean = train(Net("resnet18", len(classes), True), Xtr, ytr, 5, 1e-3, 64, "cpu", "linear", seed=1)
    pool = ((yte == T) & (predict(clean, Xte, "cpu") == T)).nonzero().flatten()
    g = torch.Generator().manual_seed(0)
    picks = pool[torch.randperm(len(pool), generator=g)][:25].view(5, 5)
    ba = (ytr == B).nonzero().flatten()
    Xb, Xt = Xtr[ba], Xte[picks[0]]
    with torch.no_grad():
        lb, lt = sur(Xb), sur(Xt)
        fb, ft = get_features(sur, Xb, "cpu"), get_features(sur, Xt, "cpu")

    cos, rb = grad_cos(lb, fb, lt, ft, B)
    pB = F.softmax(lb, 1)[:, B]
    rn, fn = rb.norm(dim=1), fb.norm(dim=1)
    print("\n==== %s, base class %s, %d training images" % (data, base, len(Xb)))
    print("softmax p(base): min %.6f median %.6f max %.6f" % (pB.min(), pB.median(), pB.max()))
    print("residual norm  : min %.3e median %.3e max %.3e" % (rn.min(), rn.median(), rn.max()))
    print("feature norm   : min %.3f median %.3f max %.3f" % (fn.min(), fn.median(), fn.max()))
    print("gradient block norms: weight ||r||*||phi|| median %.3e, bias ||r|| median %.3e, ratio %.1f"
          % ((rn * fn).median(), rn.median(), ((rn * fn) / rn).median()))
    print("head-gradient cosine: min %.6f max %.6f std %.3e" % (cos.min(), cos.max(), cos.std()))
    rdir = F.normalize(rb, dim=1)
    fdir = F.normalize(fb, dim=1)
    print("pairwise cosine of residual directions, off-diagonal: min %.6f mean %.6f"
          % ((rdir @ rdir.t()).fill_diagonal_(2).min(), ((rdir @ rdir.t()).fill_diagonal_(0).sum() / (len(Xb) ** 2 - len(Xb)))))
    print("pairwise cosine of feature directions, off-diagonal : min %.6f mean %.6f"
          % ((fdir @ fdir.t()).fill_diagonal_(2).min(), ((fdir @ fdir.t()).fill_diagonal_(0).sum() / (len(Xb) ** 2 - len(Xb)))))
    print("argmax of residual magnitude is the base class for %d/%d images" % ((rb.abs().argmax(1) == B).sum(), len(Xb)))

    alt = {"feature cosine to target mean": F.cosine_similarity(fb, ft.mean(0, keepdim=True), dim=1)}
    for tp in (2.0, 5.0, 10.0):
        alt["head gradient at temperature %g" % tp] = grad_cos(lb, fb, lt, ft, B, temp=tp)[0]
    for k, v in alt.items():
        top = v.topk(min(50, len(v))).indices
        base_top = cos.topk(min(50, len(cos))).indices
        print("  %-34s min %.6f max %.6f std %.3e | overlap with head-gradient top 50: %d/50 | spearman %.3f"
              % (k, v.min(), v.max(), v.std(), len(set(top.tolist()) & set(base_top.tolist())), spearman(v, cos)))
