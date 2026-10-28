# Datasets

No images are committed to this repository. Both datasets must be obtained separately.

Every script expects an ImageFolder layout: `DATA_DIR/<class_name>/*.png`. Images are resized to
64x64 (PIL bilinear, aspect ratio not preserved) at load time, so the resolution of what you
download does not need to match.

## Malimg

Malware byte-plot images, 25 families, from Nataraj et al., "Malware Images: Visualization and
Automatic Classification". The dataset is distributed by its authors and is **not redistributable
here**, so it is not included and no copy is linked. Obtain it from the authors or a dataset host
you trust, and unpack it so that each family is its own directory:

```
malimg_all/Allaple.A/*.png
malimg_all/VB.AT/*.png
...
```

The runs in `RESULTS.md` use `--data malimg_all --max-per-class 300`, which samples at most 300
files per class with a fixed shuffle seed.

**Known problem with this dataset, read before using it.** Several Malimg classes are not sets of
distinct images. In the copy used here, all 800 files of Yuner.A decode to one identical pixel
array, and Autorun.K, Fakerean, Adialer.C and Rbot!gen are also substantially duplicated. Section
9t of `RESULTS.md` has the per-class audit and `s9t_class_audit.py` reproduces it on your copy. Run
it before drawing a conclusion from any Malimg experiment.

Its license was not checked as part of this work.

## ISIC 2020

Skin lesion photographs from the ISIC 2020 Challenge training set, fetched through the ISIC Archive
API v2. No account or API key is needed.

```
python download_isic.py --out isic_folder --malignant all --benign 1500
```

That writes `isic_folder/benign/*.jpg`, `isic_folder/malignant/*.jpg` and
`isic_folder/manifest.csv`, which records each image's id, label, **license** and attribution. The
runs use `--data isic_folder --max-per-class 1500`, giving 1500 benign and all 584 malignant
images at 256 px thumbnail size.

**Licensing matters here.** The downloaded set mixes licenses: in the copy used for this work,
1137 images are CC BY and 947 are CC BY-NC. Before publishing or presenting any ISIC image, check
its license in `manifest.csv` and carry the required attribution. CC BY-NC images cannot be used
in a commercial context, which includes some conference and video platforms. Section 7 of
`RESULTS.md` records the ids and credit line for the four CC BY images used in the example figure.

This is also why `out_*/` is gitignored: each run directory contains an `examples.pt` holding raw
image pixels from both datasets.
