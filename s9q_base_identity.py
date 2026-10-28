"""The Malimg selection score is constant because the frozen features are constant. This checks
whether the base-class images themselves are identical, at 64x64 and at native resolution."""
import os
import numpy as np
import torch
from PIL import Image
from poison import load_folder, split

for data, mx, base, size in [("malimg_all", 300, "Yuner.A", 64), ("isic_folder", 1500, "benign", 64)]:
    X, y, classes = load_folder(data, size, mx, 0)
    Xtr, ytr, _, _ = split(X, y, 0.25, 0)
    B = classes.index(base)
    Xb = Xtr[(ytr == B).nonzero().flatten()]
    flat = (Xb * 255).round().to(torch.uint8).reshape(len(Xb), -1)
    uniq = torch.unique(flat, dim=0)
    d = (Xb - Xb[0]).abs().amax(dim=(1, 2, 3)) * 255
    print("%s %s at %dx%d: %d images, %d distinct, max pixel distance to image 0: %.1f gray levels"
          % (data, base, size, size, len(Xb), len(uniq), d.max()))

root = os.path.join("malimg_all", "Yuner.A")
files = sorted(f for f in os.listdir(root) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
sizes, h0 = set(), None
for f in files[:60]:
    im = Image.open(os.path.join(root, f)).convert("L")
    sizes.add(im.size)
    a = np.asarray(im, dtype=np.int16)
    if h0 is None:
        h0, ref = a.shape, a
    print("  %s native %s" % (f, im.size)) if len(sizes) <= 3 and f in files[:3] else None
print("native sizes among the first 60 Yuner.A files: %s" % sorted(sizes)[:6])

ims = [np.asarray(Image.open(os.path.join(root, f)).convert("L").resize((64, 64), Image.BILINEAR), dtype=np.int16)
       for f in files[:60]]
st = np.stack(ims)
print("first 60 Yuner.A files resized to 64x64: %d distinct, max pixel spread %d gray levels, mean std per pixel %.3f"
      % (len(np.unique(st.reshape(60, -1), axis=0)), (st.max(0) - st.min(0)).max(), st.std(0).mean()))
