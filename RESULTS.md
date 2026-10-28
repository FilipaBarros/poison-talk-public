# Mathematical Patterns of Poison: measured results

Clean-label data poisoning experiments on two image datasets, run for a 45 minute talk at BSides
Lisbon 2026. Speaker: Filipa Barros. See README.md for how to reproduce any number below.

Every number here comes from a run recorded in `results/raw/`. Weak and failed settings are
reported the same way as positive ones, and no cause is asserted for a result that was not
measured.

## Read this before quoting any Malimg number

The Malimg base class used by every Malimg experiment in sections 2 and 8 to 9s, Yuner.A, is a
single image repeated. All 800 files in that class folder decode to one identical pixel array at
native 683x768 resolution, and to one identical array at the 64x64 size the models see. The 133
distinct file checksums are PNG encoding differences only. So every Malimg poison in those
sections is a perturbation of the same image, and the 50 poisons in a run are 50 perturbations of
one picture rather than 50 different pictures.

Sections 9q to 9y take that apart. With a fully distinct base class the Malimg targeted attack
drops from 12 of 15 targets to 0 of 15, and raising the budget does not recover it. Collapsing a
distinct class to one repeated image recovers part of the effect, 3 to 4 of 15, which is how much
of the original result is attributable to duplication. The nearest and farthest distinct base
classes in feature space both score at most 1 of 15, so distance does not explain the remainder,
which is still open.

The ISIC results are not affected: the ISIC benign class is 1124 distinct images out of 1125.

## How this document is organised

Sections were written in the order the work was done and appended, never rewritten. **Later
sections correct earlier ones.** Where a later section revisits an earlier claim it says so
explicitly. Section 9t lists every earlier section whose base or victim class is affected by the
duplicate problem. If an early section and a late section disagree, the late one is the current
reading.

There is no section 9n: that experiment, a second victim architecture and an ensemble surrogate,
was planned but never run, and the gap is left visible rather than renumbered.

## 1. Setup (same for every experiment unless stated)

- Model: ResNet-18 pretrained on ImageNet. The feature layers are frozen and only the final classifier layer is retrained ("linear" mode). Images resized to 64x64. 5 training epochs.
- Attacker's knowledge: a surrogate model with the same architecture and starting weights, trained on the clean data.
- Threat model: clean-label. The poisons are real training images of the base class with their correct labels. Each one gets a small perturbation bounded in L-infinity norm by eps (on a 0 to 255 pixel scale). The defender retrains on the data and evaluates on a clean, untouched test set.
- Attack: gradient matching (Witches' Brew, Geiping et al.). The perturbation is optimized with projected sign-gradient steps (cosine step-size decay from eps/4, 100 steps) so that training on the poisons pushes the model's weights the same way as training directly on "this target should be labeled base". This is the "bounded norm constraint optimized against the target model's loss" in the abstract.
- Image quality: PSNR between clean and poisoned image is about 43.8 dB at eps 2, 38.7 dB at eps 4, 33.6 dB at eps 8 and 28.1 dB at eps 16.
- Controls: every result is compared to the same pipeline with the same images swapped in unperturbed (eps 0). That way any effect comes from the perturbation, not from the image swap.

Datasets:
- Malimg: malware binaries rendered as grayscale images, 25 families, up to 300 images per family, 75/25 train/test split. Clean model accuracy about 0.90.
- ISIC: ISIC 2020 Challenge training set via the ISIC Archive API. 1500 random benign images plus all 584 malignant ones (256 px thumbnails). Clean model accuracy about 0.79, malignant recall about 0.50. This is a weak baseline (tiny images, frozen features) and should be described as one.

## 2. Case study 1: malware (Malimg)

### 2a. Targeted samples: works

Goal: one specific VB.AT sample gets classified as Yuner.A after retraining. 5 targets per seed, 3 seeds (0, 1, 2), so 15 targets per cell. Accuracy on all other test images stays at about 0.90 in every setting.

Perturbation applied to all three color channels (RGB):

| eps | 25 poisons (11% of Yuner.A) | 50 poisons (22% of Yuner.A) |
|---|---|---|
| 0 (control) | 0/15 missed | 0/15 missed |
| 2 (seed 0 only) | 0/5 | 0/5 |
| 4 (seed 0 only) | 1/5 | 2/5 |
| 8 | 11/15 | 13/15 |
| 16 (seed 0 only) | 5/5 | 5/5 |

Caveat: Malimg images are grayscale. An RGB perturbation adds a faint color tint, which a defender could spot just by checking that the image is still gray. So the experiment was repeated with a grayscale perturbation (the same value on all three channels). This is the honest version for the talk:

| eps | 25 poisons | 50 poisons |
|---|---|---|
| 8 | 3/15 missed | 7/15 missed |
| 16 (seed 0 only) | 3/5 | 5/5 |

The grayscale attack is clearly weaker: about half the targets at eps 8 with 50 poisons.

Defenses, targeted attack (seed 0, 5 trials, 50 poisons). Both filters see only the poisoned training set and the model trained on it. "Missed after" means after removing what the filter flags and retraining.

| Filter | RGB eps 8: caught / clean removed / missed after | Gray eps 16: caught / clean removed / missed after |
|---|---|---|
| None | (missed: 5/5) | (missed: 5/5) |
| Deep k-NN | 23% / 9% / 4/5 | 26% / 9% / 3/5 |
| Spectral, remove 5% | 22% / 5% / 4/5 | 22% / 5% / 2/5 |
| Spectral, remove 10% | 44% / 9% / 2/5 | 44% / 9% / 1/5 |
| Spectral, remove 20% | 90% / 19% / 0/5 | 82% / 19% / 0/5 |

### 2b. A whole family: fails for established families, works for a new one

- Established families do not move. On Instantaccess with feature collision (Poison Frogs), recall stayed at 0.981 for every eps up to 8 and up to 200 poisons. With gradient matching up to eps 32 it stayed at 0.99 to 1.00.
- Why: an upper-bound check placed exact copies of a target's feature vector into training, labeled as the base class. This is stronger than any real poison. Families whose training images sit tightly together in feature space still did not flip. Spread is the mean distance of a family's training features to their center; the counts are targets flipped out of 3 with 5 / 25 / 50 copies.
  - Instantaccess: spread 5.90, flipped 0/3, 0/3, 0/3
  - Obfuscator.AD: spread 1.47, flipped 0/3, 0/3, 0/3
  - Skintrim.N: spread 4.22, flipped 0/3, 0/3, 0/3
  - VB.AT: spread 17.28, flipped 1/3, 3/3, 3/3
  - Lolyda.AA2: spread 17.69, flipped 1/3, 3/3, 3/3
  - Pattern: tight families resist, wide families flip.
- Data quirk: Autorun.K and Yuner.A are near-duplicates in Malimg (spread 0.00, recall 0 for Autorun.K even without an attack). Do not use that pair.
- New-family scenario (works): VB.AT with only 10 training images, as if the family had just appeared. Grayscale perturbation, eps 8, poisons replace Yuner.A images. VB.AT recall on the whole clean test set, each value the mean of 3 retrained victims:

| Seed | 50 unperturbed (control) | 25 poisoned | 50 poisoned |
|---|---|---|---|
| 0 | 0.964 | 0.778 | 0.738 |
| 1 | 0.822 | 0.609 | 0.302 |
| 2 | 0.964 | 0.907 | 0.844 |
| Mean | 0.917 | 0.765 | 0.628 |

  - Accuracy on everything else stays between 0.84 and 0.91.
  - A second independent run with defenses gave the same means: control 0.91, 50 poisoned 0.63.
  - Individual seeds vary a lot, so quote the means.

Defenses, new-family setting (3 seeds, 50 poisons, eps 8). "Recall after" is VB.AT recall after filtering and retraining. No filter: 0.91 with unperturbed images, 0.63 poisoned.

| Filter | Poisons caught | Clean removed | Recall after (poisoned set) | Recall after (unperturbed set) | VB.AT training images kept |
|---|---|---|---|---|---|
| Deep k-NN | 17% | 10% | 0.00 | 0.35 | 0 of 10 |
| Spectral 5% | 22% | 5% | 0.67 | 0.91 | 10 of 10 |
| Spectral 10% | 44% | 9% | 0.62 | 0.94 | 9 of 10 |
| Spectral 20% | 85% | 19% | 0.88 | 0.88 | 8 of 10 |

- Spectral at 20% undoes the attack, at the cost of a fifth of the training data.
- Deep k-NN removes the entire rare family: a family with 10 images never has a majority among 15 nearest neighbours. It is worse than no defense.

### 2c. Full fine-tuning: failed

With the whole network fine-tuned (not just the last layer), clean accuracy was 0.95. The attack got 0/3 targets at eps 8 with 50 poisons. The crafting could not match the full network's gradients (cosine similarity 0.47, against 0.98 in linear mode). Stronger crafting (restarts, model ensembles, a GPU) was not tried. Present this as a limitation.

## 3. Case study 2: medical imaging (ISIC)

### 3a. Targeted samples: works

Goal: one specific malignant lesion gets classified as benign. 5 targets per seed, 3 seeds, 15 targets per cell. Accuracy on other test images stays at about 0.78 to 0.79 (clean 0.79).

| eps | 25 poisons (2% of benign) | 50 poisons (4% of benign) |
|---|---|---|
| 0 (control) | 1/15 missed | 1/15 missed |
| 8 | 9/15 | 13/15 |
| 16 (seed 0 only) | 3/5 | 5/5 |

Defenses, targeted ISIC (seed 0, eps 8, 50 poisons, missed before filtering 5/5):

| Filter | Caught | Clean removed | Missed after |
|---|---|---|---|
| Deep k-NN | 1% | 22% | 5/5 |
| Spectral 5% | 24% | 4% | 5/5 |
| Spectral 10% | 31% | 9% | 5/5 |
| Spectral 20% | 39% | 19% | 4/5 |

No filter stops the targeted ISIC attack.

### 3b. Shift in diagnostic accuracy on the clean test set: works, modest

150 poisons (13% of benign), eps 8, no specific target: the poisons push malignant images in general toward benign. Malignant recall on the clean test set, each value the mean of 3 retrained victims:

| Seed | 150 unperturbed (control) | 150 poisoned |
|---|---|---|
| 0 | 0.516 | 0.370 |
| 1 | 0.493 | 0.365 |
| 2 | 0.546 | 0.443 |
| 3 | 0.621 | 0.530 |
| 4 | 0.443 | 0.352 |
| Mean | 0.524 | 0.412 |

- Recall drops in 5 of 5 seeds, by 0.09 to 0.15.
- Overall accuracy drops only about 0.01 to 0.03, so checking accuracy alone would barely notice.
- Honest caveat: swapping in unperturbed images alone moved recall by up to 0.12 in one seed. The shift is consistent in direction but similar in size to run-to-run noise.

Defenses, class-level ISIC (5 seeds). No filter: 0.53 unperturbed, 0.42 poisoned.

| Filter | Poisons caught | Clean removed | Recall after (poisoned set) | Recall after (unperturbed set) |
|---|---|---|---|---|
| Deep k-NN | 20% | 25% | 0.27 | 0.37 |
| Spectral 5% | 15% | 4% | 0.51 | 0.55 |
| Spectral 10% | 23% | 9% | 0.57 | 0.58 |
| Spectral 20% | 39% | 18% | 0.58 | 0.60 |

- Spectral filtering cancels the shift, but it catches only 39% of poisons at best. Part of the recovery comes from removing benign images, which raises malignant recall even with no attack (0.53 to 0.60). The accurate description is that filtering cancels the shift, not that filtering finds the poisons.
- Deep k-NN removes about 62% of the malignant training images and makes recall worse.

## 4. What did not work

- Feature collision, at the class level, on Malimg: no effect at any setting.
- Class-level attacks on established, tight malware families: no effect, and the upper-bound check shows they cannot be flipped this way.
- Full fine-tuning victim: 0 of 3 targets.
- Grayscale perturbation on Malimg is about half as effective as RGB at eps 8.
- No defense tested catches most poisons cheaply. The one that works (spectral at 20%) costs about 19% of clean data.
- Not tested: JPEG compression or resizing of poisons, training from scratch at scale, other architectures.

## 5. Figures

No figure images are committed. Each script reads a run directory and writes 200 dpi PNGs into
`figures/`.

| Script | Figures it writes |
|---|---|
| `make_figures.py DIR` | `fig_poison_examples.png` (clean, poisoned, difference x20), `fig_recall_vs_poison.png`, `fig_features.png` (2D PCA of target and base features, clean model against poisoned victim), `fig_confusion.png` |
| `make_figures_class.py DIR PREFIX` | `PREFIX_recall_by_seed.png`, `PREFIX_poison_examples.png`, `PREFIX_defenses.png` (poisons caught against clean removed, and target recall after filtering) |
| `make_figures_targeted.py DIR PREFIX` | `PREFIX_success_vs_eps.png`, `PREFIX_poison_examples.png`, `PREFIX_defenses.png` |

The example-image figures read pixel data from a run's `examples.pt`, which is not committed. They
need a local run directory, so regenerating them means rerunning the experiment. ISIC images carry
per-image licenses, some of them CC BY-NC, recorded in the `manifest.csv` that
`download_isic.py` writes; check them before publishing any ISIC figure.

## 6. Abstract claims and the evidence

| Abstract says | Evidence | Status |
|---|---|---|
| "small, often imperceptible" perturbations | eps 8/255, PSNR about 33.7 dB; example figures | Supported at eps 8. At eps 16 the texture is visible on Malimg. |
| "a bounded norm constraint optimized against the target model's loss" | L-infinity bound, gradient matching on the model's training loss | Supported, this is exactly the method. |
| "generalizes to any model trained on images" | Same method works on two very different image domains | Partly. Shown only on ResNet-18 with a frozen backbone. Fine-tuning failed. Frame it as "the math does not depend on the domain", not "tested on any model". |
| Case 1: "cause the model to miss a targeted malware family after retraining" | Targeted samples: 13/15 (RGB), 7/15 (gray). New family: recall 0.92 to 0.63 | Supported for specific samples and for a newly appearing family with few samples. Not for established families. |
| Case 2: "shifts diagnostic accuracy on a clean test set" | Malignant recall 0.52 to 0.41, 5/5 seeds | Supported, modest effect. |
| How attacks are caught, how far detection gets | Deep k-NN and spectral signatures, with controls | Supported. Main message: detection is weak, and the one filter that works costs a fifth of the data. |

## 7. Open items

- ISIC image licenses: the downloaded set is 1137 CC BY and 947 CC BY-NC. The talk is recorded for YouTube, so only CC BY images go on slides. The medical example figure uses these 4 CC BY images (license string in the ISIC manifest: "CC-BY"):
  - ISIC_5361778, Memorial Sloan Kettering Cancer Center
  - ISIC_2987630, Memorial Sloan Kettering Cancer Center
  - ISIC_7569912, The University of Queensland Diamantina Institute, The University of Queensland, Dermatology Research Centre
  - ISIC_1000648, Memorial Sloan Kettering Cancer Center
  - Credit line to use when showing these images: "Images: ISIC Archive (isic-archive.com), ISIC 2020 Challenge, CC BY. ISIC_5361778, ISIC_2987630, ISIC_1000648: Memorial Sloan Kettering Cancer Center. ISIC_7569912: The University of Queensland. Modified: perturbation added."
  - Do not publish isic_class_poison_examples.png, isic_class_def_poison_examples.png or isic_def_poison_examples.png: they contain CC BY-NC images.
- Malimg dataset license: not checked.

## 8. Do the poisons survive JPEG compression and resizing?

This section answers the "Not tested: JPEG compression or resizing of poisons" item in section 4. Sections 1 to 7 are unchanged.

### 8a. What the test covers

Where the resize happens: every image is resized to 64x64 (PIL bilinear, aspect ratio not kept) when it is loaded, before anything else. Surrogate training, poison crafting and victim training all work on these 64x64 images. So the perturbation is crafted at exactly the resolution the victim sees, and one resize variant (64 to 32 to 64) is enough. The test assumes the attacker hands over 64x64 poisons. Poisons crafted on the full-size originals (Malimg PNGs about 512 to 768 px wide, ISIC thumbnails 256 px wide) and then downscaled by the victim are not tested. In the reported runs the poisons are never rounded to whole pixel values (0 to 255), so every transform first rounds them. The "uint8" row shows the effect of that rounding alone.

- Same settings as the reported targeted runs: ResNet-18, frozen backbone, 5 epochs, same surrogate, eps 8, 25 and 50 poisons, 3 seeds x 5 targets. Malimg uses the grayscale perturbation.
- Only the poisoned images are transformed. The rest of the training set and the test set are untouched.
- JPEG: PIL, default settings, quality 90, 75 and 50, on the uint8 image. Resize: PIL bilinear, 64 to 32 to 64.
- Control: the same images swapped in unperturbed (eps 0), put through the same transforms.
- PSNR: transformed poison against the clean original image. For the eps 0 rows it shows how much the transform alone changes a clean image.
- The attacker does not know about the transform. Crafting poisons that are designed to survive JPEG or resizing was not tried.

### 8b. Malimg, targeted, grayscale perturbation

Targets missed out of 15 (3 seeds x 5 targets).

| Transform | eps 0, 25 | eps 0, 50 | eps 8, 25 poisons | eps 8, 50 poisons | PSNR eps 8 | PSNR eps 0 |
|---|---|---|---|---|---|---|
| None | 0/15 | 0/15 | 3/15 | 6/15 | 33.5 dB | n/a |
| uint8 rounding only | 0/15 | 0/15 | 3/15 | 6/15 | 33.5 dB | n/a |
| JPEG 90 | 0/15 | 0/15 | 0/15 | 3/15 | 32.4 dB | 38.4 dB |
| JPEG 75 | 0/15 | 0/15 | 0/15 | 0/15 | 32.0 dB | 34.9 dB |
| JPEG 50 | 0/15 | 0/15 | 0/15 | 0/15 | 31.2 dB | 32.8 dB |
| Resize 64-32-64 | 0/15 | 0/15 | 0/15 | 0/15 | 25.4 dB | 25.5 dB |

- Accuracy on the other test images: 0.902 to 0.907 in every cell.
- Per seed, eps 8, 50 poisons (seed 0 / 1 / 2, out of 5): none 3 / 3 / 0, JPEG 90 2 / 1 / 0, everything else 0 / 0 / 0.
- Result: the Malimg attack does not survive. JPEG 75, JPEG 50 and the resize remove it completely. JPEG 90 halves it at 50 poisons and removes it at 25.
- Resizing changes clean Malimg images a lot too (PSNR 25.5 dB with no perturbation): the byte-plot texture is fine-grained, and halving the resolution blurs it.

### 8c. ISIC, targeted

Targets missed out of 15 (3 seeds x 5 targets).

| Transform | eps 0, 25 | eps 0, 50 | eps 8, 25 poisons | eps 8, 50 poisons | PSNR eps 8 | PSNR eps 0 |
|---|---|---|---|---|---|---|
| None | 1/15 | 1/15 | 9/15 | 12/15 | 34.0 dB | n/a |
| uint8 rounding only | 1/15 | 1/15 | 9/15 | 12/15 | 33.9 dB | n/a |
| JPEG 90 | 1/15 | 1/15 | 7/15 | 9/15 | 34.3 dB | 40.2 dB |
| JPEG 75 | 1/15 | 2/15 | 5/15 | 7/15 | 34.5 dB | 37.9 dB |
| JPEG 50 | 2/15 | 2/15 | 4/15 | 5/15 | 34.0 dB | 35.9 dB |
| Resize 64-32-64 | 2/15 | 2/15 | 4/15 | 4/15 | 36.1 dB | 37.6 dB |

- Accuracy on the other test images: 0.785 to 0.790 in every cell.
- Per seed, eps 8, 50 poisons (seed 0 / 1 / 2, out of 5): none 5 / 5 / 2, JPEG 90 5 / 4 / 0, JPEG 75 3 / 4 / 0, JPEG 50 2 / 3 / 0, resize 2 / 2 / 0.
- Result: the ISIC attack is weakened but not removed. At 50 poisons it goes from 12/15 to 9/15 (JPEG 90), 7/15 (JPEG 75), 5/15 (JPEG 50) and 4/15 (resize). The matching controls miss 1 or 2 of 15, so JPEG 50 and resize leave only about 2 to 3 targets above the control level.
- JPEG on an ISIC poison does not lower PSNR against the clean image (34.0 to 34.5 dB, against 34.0 dB with no transform). Compression removes part of the perturbation about as much as it adds its own error.

### 8d. What survived and what did not

- Rounding to whole pixel values (uint8): no effect on either dataset.
- Malimg: does not survive JPEG at quality 75 or lower, or one resize. At JPEG 90, 3 of 15 targets are still missed at 50 poisons, 0 of 15 at 25 poisons.
- ISIC: partly survives every setting tested. The weakest settings (JPEG 50, resize) leave 4 to 5 of 15 targets missed at 50 poisons, against 1 to 2 of 15 in the matching controls.
- Neither transform costs accuracy on the other test images.
- For the talk: a defender who re-encodes or downscales incoming images cancels the malware attack and cuts the medical one from 12 of 15 targets to between 4 and 9 of 15, depending on the setting. This was tested only against an attacker who does not plan for the transform.

### 8e. Reproduction notes

- The "None" rows are a fresh rerun, not the stored runs from sections 2a and 3a. The rerun gives Malimg 3/15 and 6/15 (reported 3/15 and 7/15) and ISIC 9/15 and 12/15 (reported 9/15 and 13/15). On this CPU the crafted poisons are not bit-for-bit repeatable, so single targets can flip between runs. Within one trial, every row of the table uses the same poisons.
- The Malimg eps 0 controls for seeds 1 and 2 ran with `--fast`, which caches the frozen backbone's features and skips crafting at eps 0 (the perturbation is exactly zero there). A `--fast` rerun of the seed 0 control matched the original run's clean model, target predictions and targets missed in every row. The only differences were 5 accuracy values of the uint8 row at 25 poisons (0.9074 against 0.9048). At eps 0 that row's training data is bit-identical to the "None" row's. The original slow run gave the two rows different values, and `--fast` gave both 0.9074, so the difference is CPU floating-point noise in the slow run.

### 8f. Exact commands

Main runs (`robust_runs.sh`):

```
TF="uint8 jpeg90 jpeg75 jpeg50 resize32"
COMMON="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 25 50"
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
venv/bin/python run_experiment.py $ISIC $COMMON --eps 8 --seed 0 --transforms $TF --out out_robust_isic_s0
venv/bin/python run_experiment.py $ISIC $COMMON --eps 0 --skip-defense --seed 0 --transforms $TF --out out_robust_isic_ctrl_s0
for s in 1 2; do venv/bin/python run_experiment.py $ISIC $COMMON --eps 0 8 --skip-defense --seed $s --transforms $TF --out out_robust_isic_s$s; done
for s in 0 1 2; do venv/bin/python run_experiment.py $MAL $COMMON --eps 8 --skip-defense --seed $s --transforms $TF --out out_robust_malimg_s$s; done
venv/bin/python run_experiment.py $MAL $COMMON --eps 0 --skip-defense --seed 0 --transforms $TF --out out_robust_malimg_ctrl_s0
```

Malimg controls, seeds 1 and 2, plus the seed 0 check (`robust_runs_resume.sh`):

```
venv/bin/python -u run_experiment.py $MAL $COMMON --eps 0 --skip-defense --seed 0 --transforms $TF --fast --out out_robust_malimg_ctrl_s0_fastcheck
for s in 1 2; do venv/bin/python -u run_experiment.py $MAL $COMMON --eps 0 --skip-defense --seed $s --transforms $TF --fast --out out_robust_malimg_ctrl_s$s; done
```

The eps lists and `--skip-defense` choices match the stored runs, so the targets and base images drawn are the same. ISIC seed 0 runs the defenses, as the stored seed 0 run did. Tables: `venv/bin/python robust_summary.py`.

### 8g. Second frozen backbone: ResNet-50

Same as 8b and 8c, but the victim and the surrogate are both a frozen ImageNet ResNet-50 (the surrogate always has the victim's architecture, as in section 1). Only the strongest cell was run: eps 8, 50 poisons, 3 seeds x 5 targets, with the eps 0 control. Clean model accuracy: Malimg 0.92, ISIC 0.79 to 0.81. A ResNet-18 surrogate against a ResNet-50 victim (transfer) was not tested.

Targets missed out of 15:

| Transform | Malimg eps 0 | Malimg eps 8 | Malimg PSNR eps 8 | ISIC eps 0 | ISIC eps 8 | ISIC PSNR eps 8 |
|---|---|---|---|---|---|---|
| None | 0/15 | 15/15 | 34.1 dB | 1/15 | 15/15 | 34.4 dB |
| uint8 rounding only | 0/15 | 15/15 | 34.0 dB | 1/15 | 15/15 | 34.4 dB |
| JPEG 90 | 0/15 | 9/15 | 32.8 dB | 1/15 | 12/15 | 35.1 dB |
| JPEG 75 | 0/15 | 0/15 | 32.3 dB | 1/15 | 7/15 | 35.3 dB |
| JPEG 50 | 0/15 | 0/15 | 31.5 dB | 1/15 | 4/15 | 34.6 dB |
| Resize 64-32-64 | 0/15 | 0/15 | 25.4 dB | 1/15 | 7/15 | 36.6 dB |

- Accuracy on the other test images: Malimg 0.914 to 0.922, ISIC 0.783 to 0.802, in every cell.
- Per seed, eps 8 (seed 0 / 1 / 2, out of 5): Malimg none 5 / 5 / 5, JPEG 90 3 / 3 / 3, everything else 0 / 0 / 0. ISIC none 5 / 5 / 5, JPEG 90 5 / 4 / 3, JPEG 75 1 / 3 / 3, JPEG 50 1 / 1 / 2, resize 1 / 3 / 3.
- With no transform, the attack is stronger on ResNet-50 than on ResNet-18: 15/15 on both datasets, against 6/15 (Malimg, grayscale) and 12/15 (ISIC) for ResNet-18 in the same rerun.
- After the transforms the pattern is the same as on ResNet-18. Malimg: JPEG 75, JPEG 50 and resize remove the attack, and JPEG 90 leaves 9/15. ISIC: weakened but not removed, 4/15 to 12/15 against 1/15 in the control.
- Only a single run, so treat ResNet-50 as one data point for "other architectures", not as a general result. Full fine-tuning was not tested.
- These runs used `--fast`. On ResNet-50, training the same victim with and without the cache gave identical classifier weights (maximum difference 0) and identical predictions on all 521 ISIC test images.

Command (`robust_runs_r50.sh`):

```
TF="uint8 jpeg90 jpeg75 jpeg50 resize32"
COMMON="--arch resnet50 --size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --eps 0 8 --skip-defense --fast --transforms $TF"
for s in 0 1 2; do venv/bin/python -u run_experiment.py $ISIC $COMMON --seed $s --out out_robust_r50_isic_s$s; done
for s in 0 1 2; do venv/bin/python -u run_experiment.py $MAL $COMMON --seed $s --out out_robust_r50_malimg_s$s; done
```

ResNet-50 weights: torchvision 0.17.2 default (IMAGENET1K_V2, resnet50-11ad3fa6.pth). Tables: `venv/bin/python robust_summary.py out_robust_r50_isic_s0 ...` (one block per run).

## 9. Compression-aware crafting

Section 8 showed that JPEG and resizing cancel the Malimg attack and weaken the ISIC one. This section asks whether an attacker who knows about the transform can craft poisons that survive it. Short answer: not in these experiments. Sections 1 to 8 are unchanged.

### 9a. What was tried

Two changes to the crafting loop, both off by default:

- Crafting through the transforms (`--craft-transforms`). At each of the 100 crafting steps one transform is drawn at random from none, JPEG 90, JPEG 75, JPEG 50 and resize 64 to 32 to 64, and applied to the poisoned images before the gradient matching loss. The projected sign-gradient update and the cosine step schedule are unchanged. This needs transforms that can be differentiated: JPEG is the 8x8 DCT with the standard quantization tables scaled for the quality, and rounding replaced by a straight-through estimator. The resize is bilinear with antialiasing, matching the evaluation transform.
- A low-resolution perturbation (`--lowres N`). The variable optimized is an N x N perturbation (one channel for Malimg, three for ISIC) upsampled bilinearly to 64x64. Bilinear upsampling is a weighted average, so bounding the variable by eps keeps the full-size perturbation within eps. N 16 and N 32 were tried.

Assumption: the attacker knows the transform family and can put it in the crafting loop. The attacker does not know which transform the victim will use, or its exact implementation.

Shipped poisons from these runs are rounded to whole pixel values before anything else, which is what an attacker would hand over. The evaluation transform is then applied to that image, as in section 8.

Differentiable JPEG against PIL at the same quality, mean absolute difference over 64 real images, in gray levels out of 255:

| Transform | Malimg (grayscale) | ISIC (color) |
|---|---|---|
| JPEG 90 | 0.46 | 1.50 |
| JPEG 75 | 0.40 | 1.83 |
| JPEG 60 | 0.36 | 2.06 |
| JPEG 50 | 0.38 | 2.20 |
| Resize 64 to 32 to 64 | 0.45 | 0.44 |

The ISIC error is larger because the differentiable version works in YCbCr with no chroma subsampling, while PIL subsamples chroma at these qualities. Chroma subsampling was skipped. The resize needed antialiasing to match PIL: without it the Malimg difference is 1.99 gray levels instead of 0.45.

With both flags off, the crafting loop is bit-identical to the previous one. Checked by running the new function and a verbatim copy of the old one in the same process, single threaded, same seed: the crafted poisons match exactly (largest pixel difference 0) on both datasets. A rerun cannot show this, because the pipeline is not bit-reproducible on this machine: running the same command twice gave cosine 0.9764 and 0.9775 for the same trial. That is the same run-to-run spread noted in section 8e.

### 9b. Pilot, Malimg seed 0, eps 8, 50 poisons

Grayscale perturbation, 5 targets, one seed. "No transform" is the shipped poison with whole pixel values. Accuracy on the other test images is 0.906 to 0.909 everywhere.

| Variant | No transform | JPEG 75 | Resize 64-32-64 | Crafting cosine | PSNR | L-infinity |
|---|---|---|---|---|---|---|
| Plain crafting (section 8) | 3/5 | 0/5 | 0/5 | 0.977 | 33.6 dB | 8.0 |
| Craft through transforms | 1/5 | 0/5 | 0/5 | 0.951 | 33.4 dB | 8.0 |
| Low resolution 16 | 0/5 | 0/5 | 0/5 | 0.851 | 35.8 dB | 8.0 |
| Craft through transforms + low resolution 16 | 0/5 | 0/5 | 0/5 | 0.843 | 36.0 dB | 8.0 |
| Craft through transforms + low resolution 32 | 0/5 | 0/5 | 0/5 | 0.907 | 35.7 dB | 8.0 |

Pass bar was at least 2/5 under both JPEG 75 and resize. No variant reached 1/5 under either. Every variant is also weaker than plain crafting with no transform at all.

The cosine column is the reason. Gradient matching reaches 0.977 unconstrained, 0.951 when crafting through the transforms, and 0.84 to 0.91 with a low-resolution perturbation. Both changes cost alignment with the target gradient, and the Malimg grayscale attack is marginal to begin with at 3 of 5. The low-resolution runs also use less of the budget in practice, which shows as higher PSNR at the same L-infinity bound: a smooth perturbation cannot sit at plus or minus eps on every pixel.

### 9c. Larger budget and more poisons (step D)

Same seed and 5 targets. The setting is stated in each row because it is no longer the section 8 setting.

| Setting | Variant | No transform | JPEG 75 | Resize | Cosine | PSNR |
|---|---|---|---|---|---|---|
| eps 16, 50 poisons | Plain | 5/5 | 0/5 | 0/5 | 0.983 | 28.1 dB |
| eps 16, 50 poisons | Craft through transforms | 2/5 | 1/5 | 0/5 | 0.963 | 28.0 dB |
| eps 16, 50 poisons | Transforms + low resolution 32 | 0/5 | 0/5 | 0/5 | 0.947 | 30.6 dB |
| eps 8, 100 poisons | Plain | 5/5 | 0/5 | 0/5 | 0.976 | 33.6 dB |
| eps 8, 100 poisons | Craft through transforms | 1/5 | 0/5 | 0/5 | 0.957 | 33.4 dB |
| eps 8, 100 poisons | Transforms + low resolution 32 | 0/5 | 0/5 | 0/5 | 0.904 | 35.6 dB |
| eps 16, 100 poisons | Plain | 5/5 | 2/5 | 0/5 | 0.982 | 28.1 dB |
| eps 16, 100 poisons | Craft through transforms | 1/5 | 1/5 | 0/5 | 0.968 | 28.0 dB |
| eps 16, 100 poisons | Transforms + low resolution 32 | 0/5 | 0/5 | 0/5 | 0.946 | 30.4 dB |

Accuracy on the other test images is 0.905 to 0.908 everywhere. L-infinity is 16.0 in the eps 16 rows and 8.0 in the eps 8 row.

- The resize column is 0/5 in every row. Nothing tried here survives one 64 to 32 to 64 round trip.
- The best JPEG 75 number came from plain crafting at eps 16 with 100 poisons, 2 of 5, not from compression-aware crafting. Doubling the budget and doubling the poison count beat the method this section is about.
- eps 16 on Malimg is visible texture, as section 6 already says, so that cell is not a quiet attack.

### 9d. Held-out transforms

Same poisons, evaluated under transforms that were not in the crafting loop (JPEG 60, resize 64 to 48 to 64, Gaussian blur radius 0.5). This is a separate run at eps 16 with 100 poisons, seed 0, 5 targets, with its own eps 0 control. In the crafting loop: JPEG 90, 75, 50 and resize 32. Held out: JPEG 60, resize 48, blur.

| Transform | In crafting loop | Plain | Craft through transforms | eps 0 control (both) |
|---|---|---|---|---|
| No transform | n/a | 5/5 | 2/5 | 0/5 |
| uint8 rounding | n/a | 5/5 | 2/5 | 0/5 |
| JPEG 90 | yes | 4/5 | 2/5 | 0/5 |
| JPEG 75 | yes | 2/5 | 2/5 | 0/5 |
| JPEG 50 | yes | 0/5 | 2/5 | 0/5 |
| Resize 64-32-64 | yes | 0/5 | 0/5 | 0/5 |
| JPEG 60 | no | 0/5 | 1/5 | 0/5 |
| Resize 64-48-64 | no | 0/5 | 0/5 | 0/5 |
| Blur radius 0.5 | no | 3/5 | 1/5 | 0/5 |

PSNR 28.1 dB (plain) and 28.0 dB (compression-aware), L-infinity 16.0 in both. Accuracy on the other test images 0.905 to 0.906. Crafting cosine 0.982 and 0.969.

This is the one place where crafting through the transforms does something. It flattens the attack across the JPEG family: 2 of 5 at JPEG 90, 75 and 50 alike, where plain crafting falls from 4 to 2 to 0. At JPEG 50 the compression-aware poisons beat plain crafting, 2 of 5 against 0 of 5. The price is the ceiling: 2 of 5 instead of 5 of 5 with no transform. The flatness carries over weakly to the held-out JPEG 60 (1 of 5 against 0 of 5) but not to the held-out resize or the blur, where plain crafting is better.

Caution on these counts. The compression-aware JPEG 75 cell is 2 of 5 here and 1 of 5 in the step D run of the same configuration. That is run-to-run variation of a single target, not a difference in setting. At this sample size, 5 targets on one seed, a difference of one target means little.

### 9e. Detection (step G)

Does a perturbation crafted through compression stand out more to the filters? Malimg, seed 0, eps 16, 50 poisons, 5 trials. Both filters see only the poisoned training set and the model trained on it. The plain row is the section 2a grayscale eps 16 result, repeated here for comparison.

| Filter | Plain: caught / clean removed / missed after | Craft through transforms | Transforms + low resolution 32 |
|---|---|---|---|
| None | (missed 5/5) | (missed 2/5) | (missed 0/5) |
| Deep k-NN | 26% / 9% / 3/5 | 10% / 9% / 2/5 | 13% / 9% / 0/5 |
| Spectral, remove 5% | 22% / 5% / 2/5 | 22% / 5% / 1/5 | 22% / 5% / 0/5 |
| Spectral, remove 10% | 44% / 9% / 1/5 | 44% / 9% / 1/5 | 44% / 9% / 0/5 |
| Spectral, remove 20% | 82% / 19% / 0/5 | 88% / 19% / 0/5 | 90% / 19% / 0/5 |

The guess that a low-frequency perturbation is easier to catch is not supported. Spectral signatures catch slightly more of the compression-aware poisons at the 20% setting, 88 and 90% against 82%, and exactly the same share at 5 and 10%. Deep k-NN catches fewer, 10 to 13% against 26%. None of these gaps is large, and the compression-aware attack is weaker to begin with, so there is less to catch.

### 9f. What this means and what was not done

- Crafting through the transforms does not rescue the Malimg attack. It trades peak strength for an even response across the JPEG qualities it trained on, and at eps 8 with 50 poisons, the section 8 setting, that leaves nothing at all.
- A low-resolution perturbation makes things worse at every setting tried. The gradient matching cosine drops furthest there.
- Nothing survives resizing. Across every variant, budget and poison count in this section, the resize column is 0 of 5.
- The honest line for the talk stays the one from section 8: re-encoding or downscaling incoming images cancels the malware attack. An attacker who plans for the transform did not change that here.
- Not tested, because no variant passed the pilot: all three seeds, 25 poisons, ISIC, ResNet-50. Also not tested anywhere in this work: poisons crafted at the original image resolution and downscaled by the victim, architectures beyond ResNet-18 and ResNet-50, full fine-tuning, stronger crafting such as restarts or model ensembles, and attacks aimed at a specific known transform rather than a family.
- Scale caveat, same as the rest of this work: 64x64 images, frozen backbone, 5 epochs, one seed in this section.

### 9g. Figure

`figures/malimg_s9_ct_poison_examples.png`: target VB.AT sample, 4 clean Yuner.A images, the compression-aware poisoned versions at eps 16, and the difference x20. How visible: clearly visible. At eps 16 the perturbation changes pixels by up to 16 gray levels, 9.05 on average, and 56% of pixels move by more than 8 levels. Against the smooth banded texture of a Malimg byte plot the poisoned images look visibly mottled next to the clean ones. This is not a "small, often imperceptible" example and should not be used as one. The section 5 figures remain the ones for that.

### 9h. Exact commands

Unit checks: the differentiable JPEG against PIL, and the new default crafting path against a verbatim copy of the previous crafting loop (`s9_check_defaults_restarts.py`).

Pilot (`s9_pilot.sh`):

```
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
COMMON="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --eps 8 --skip-defense --seed 0 --fast --transforms none jpeg75 resize32"
CT="none jpeg90 jpeg75 jpeg50 resize32"
venv/bin/python -u run_experiment.py $MAL $COMMON --craft-transforms $CT --out out_s9_pilot_ct
venv/bin/python -u run_experiment.py $MAL $COMMON --lowres 16 --out out_s9_pilot_lr16
venv/bin/python -u run_experiment.py $MAL $COMMON --craft-transforms $CT --lowres 16 --out out_s9_pilot_ct_lr16
venv/bin/python -u run_experiment.py $MAL $COMMON --craft-transforms $CT --lowres 32 --out out_s9_pilot_ct_lr32
```

Larger budget and more poisons (`s9_stepd.sh`), with V empty, `--craft-transforms $CT`, or `--craft-transforms $CT --lowres 32`:

```
COMMON="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --skip-defense --seed 0 --fast --transforms none jpeg75 resize32"
venv/bin/python -u run_experiment.py $MAL $COMMON --eps 16 --n-poison 50  $V --out out_s9_d_eps16_$tag
venv/bin/python -u run_experiment.py $MAL $COMMON --eps 8  --n-poison 100 $V --out out_s9_d_n100_$tag
venv/bin/python -u run_experiment.py $MAL $COMMON --eps 16 --n-poison 100 $V --out out_s9_d_eps16n100_$tag
```

Detection and held-out transforms (`s9_gc.sh`):

```
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --seed 0 --fast"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"
venv/bin/python -u run_experiment.py $MAL $BASE --eps 16 --n-poison 50 --transforms none jpeg75 --craft-transforms $CT --out out_s9_g_ct
venv/bin/python -u run_experiment.py $MAL $BASE --eps 16 --n-poison 50 --transforms none jpeg75 --craft-transforms $CT --lowres 32 --out out_s9_g_ctlr32
venv/bin/python -u run_experiment.py $MAL $BASE --eps 0 16 --n-poison 100 --skip-defense --transforms $ALLTF --out out_s9_c_base
venv/bin/python -u run_experiment.py $MAL $BASE --eps 0 16 --n-poison 100 --skip-defense --transforms $ALLTF --craft-transforms $CT --out out_s9_c_ct
```

Figure: `venv/bin/python make_figures_targeted.py figures/malimg_s9_ct out_s9_c_ct`.

The defense runs leave out `--skip-defense`, so the filters run on the strongest cell. Every number in this section comes from one of these runs.

### 9i. ISIC with compression-aware crafting

Follow-up to 9b to 9f, which were Malimg only. ISIC, ResNet-18, frozen backbone, eps 8, 50 poisons, 3 seeds x 5 targets, crafting through none, JPEG 90, JPEG 75, JPEG 50 and resize 64 to 32 to 64. The eps 0 control is from the same runs, so poisoned and control see the same transforms. The section 8c controls agree with it: 1 to 2 of 15 in every cell.

Targets missed out of 15. The last three transforms were not in the crafting loop.

| Transform | In crafting loop | eps 8 | eps 0 control | PSNR eps 8 |
|---|---|---|---|---|
| No transform (shipped uint8) | n/a | 11/15 | 1/15 | 33.6 dB |
| uint8 rounding, repeated | n/a | 9/15 | 1/15 | 33.6 dB |
| JPEG 90 | yes | 8/15 | 1/15 | 34.0 dB |
| JPEG 75 | yes | 8/15 | 2/15 | 33.9 dB |
| JPEG 50 | yes | 8/15 | 2/15 | 33.4 dB |
| Resize 64-32-64 | yes | 8/15 | 2/15 | 35.4 dB |
| JPEG 60 | no | 8/15 | 2/15 | 33.5 dB |
| Resize 64-48-64 | no | 9/15 | 2/15 | 36.0 dB |
| Blur radius 0.5 | no | 10/15 | 1/15 | 34.8 dB |

Accuracy on the other test images is 0.778 to 0.785 at eps 8 and 0.788 to 0.790 at eps 0. L-infinity of the shipped poisons is 8.0. Crafting cosine 0.983, against 0.951 for the same change on Malimg.

Read against section 8c, which is plain crafting on the same dataset and setting:

| Transform | Plain (section 8c) | Compression-aware (here) |
|---|---|---|
| No transform | 12/15 | 11/15 |
| JPEG 90 | 9/15 | 8/15 |
| JPEG 75 | 7/15 | 8/15 |
| JPEG 50 | 5/15 | 8/15 |
| Resize 64-32-64 | 4/15 | 8/15 |

On ISIC the method does what it is supposed to do, which it did not on Malimg. The response is flat at 8 of 15 across every transform tried, in the crafting loop or not, and the untransformed cell barely moves (11 against 12). Under the two strongest transforms it roughly doubles the plain result: 8 of 15 against 5 at JPEG 50 and against 4 at resize. Holding at 8 of 15 under the held-out JPEG 60 and resize 48 matters more than the trained transforms, because those were never optimized for.

Why it works here and not on Malimg: the crafting cosine stays at 0.983 on ISIC with the transforms in the loop, against 0.977 for plain crafting in the same rerun. On Malimg the same change drops the cosine from 0.977 to 0.951. ISIC has 1137 benign training images against 225 Yuner.A images, and its features are smoother, so the optimizer has room to satisfy the transforms without giving up gradient alignment.

Honest limit on the size of this effect. The gap between plain and compression-aware at JPEG 90 and JPEG 75 (one target either way) is inside the noise floor measured below. The JPEG 50 and resize gaps, 3 and 4 targets, are larger than it.

Noise floor, measured here and not estimated. The "no transform" and "uint8 rounding" rows are the same images: `transform_images` builds both by rounding to whole pixel values, and the two tensors are bit-identical (checked with `torch.equal`). They still differ by 2 of 15, in seed 1 trial 0 and seed 2 trial 3. Two victims trained on bit-identical data with the same seed reached different verdicts on the target. So a single cell in any of these tables carries an uncertainty of roughly plus or minus 2 of 15 from victim training alone, before any effect of the transform. This applies to the section 8 tables as well. Differences of one or two targets should not be read as real.

### 9j. ResNet-50 with compression-aware crafting

Malimg, grayscale, eps 8, 50 poisons, seed 0 only, 5 targets, same crafting transforms as above. One trial takes about 2 minutes, so the run took 8.5 minutes. Clean model accuracy 0.920.

| Transform | Compression-aware, ResNet-50 | Plain ResNet-50 (section 8g, 3 seeds) | Plain ResNet-18 (section 9b, seed 0) |
|---|---|---|---|
| No transform | 2/5 | 15/15 | 3/5 |
| JPEG 75 | 0/5 | 0/15 | 0/5 |
| Resize 64-32-64 | 0/5 | 0/15 | 0/5 |

Accuracy on the other test images 0.918 to 0.922. PSNR 33.8 dB, L-infinity 8.0. Crafting cosine 0.938, against 0.975 for plain ResNet-50 crafting in section 8g.

The Malimg failure carries over to the larger backbone. Crafting through the transforms costs the same alignment it costs on ResNet-18 (cosine 0.975 down to 0.938), the untransformed attack collapses from 15 of 15 to 2 of 5, and neither JPEG 75 nor the resize leaves anything. Plain ResNet-50 crafting at this setting is far stronger untransformed and equally dead after a transform. Nothing here changes the section 8 conclusion for malware images.

One seed and 5 targets, so this is a single data point. The eps 0 control is the section 8g one, 0 of 15 under every transform. ISIC on ResNet-50 was not run.

Commands (`s9_followup.sh`):

```
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --skip-defense --fast"
CT="none jpeg90 jpeg75 jpeg50 resize32"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $ISIC $BASE --eps 0 8 --seed $s --transforms $ALLTF --craft-transforms $CT --out out_s9_isic_ct_s$s
done
venv/bin/python -u run_experiment.py $MAL $BASE --arch resnet50 --eps 8 --seed 0 --transforms none jpeg75 resize32 --craft-transforms $CT --out out_s9_r50_ct
```

Revision to 9f: the line "crafting through the transforms does not rescue the attack" holds for Malimg on both backbones. It does not hold for ISIC, where 9i shows a flat 8 of 15 across every transform tried. The claim that nothing survives resizing holds for Malimg only. On ISIC, compression-aware poisons survive the resize at 8 of 15 against a control of 2 of 15.

### 9k. ISIC on six seeds

9i used three seeds. Seeds 3, 4 and 5 were added for both plain and compression-aware crafting, so the comparison now rests on six seeds and 30 targets per arm. ISIC, ResNet-18, frozen backbone, eps 8, 50 poisons, 5 targets per seed. Seeds 0 to 2 of the plain arm are the section 8c runs and seeds 0 to 2 of the compression-aware arm are the 9i runs, both unchanged.

Targets missed out of 5, per seed:

| Transform | Plain s0 | s1 | s2 | s3 | s4 | s5 | Aware s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uint8, no transform | 5 | 5 | 2 | 5 | 4 | 5 | 4 | 4 | 1 | 5 | 4 | 5 |
| JPEG 90 | 5 | 4 | 0 | 5 | 3 | 3 | 4 | 4 | 0 | 5 | 4 | 4 |
| JPEG 75 | 3 | 4 | 0 | 5 | 2 | 3 | 4 | 4 | 0 | 5 | 3 | 4 |
| JPEG 50 | 2 | 3 | 0 | 5 | 2 | 0 | 4 | 4 | 0 | 5 | 3 | 3 |
| Resize 64-32-64 | 2 | 2 | 0 | 4 | 2 | 1 | 4 | 4 | 0 | 5 | 3 | 5 |
| JPEG 60 (held out) | - | - | - | 5 | 2 | 1 | 4 | 4 | 0 | 5 | 3 | 3 |
| Resize 64-48-64 (held out) | - | - | - | 5 | 3 | 3 | 5 | 4 | 0 | 5 | 4 | 5 |
| Blur 0.5 (held out) | - | - | - | 5 | 4 | 5 | 5 | 4 | 1 | 5 | 4 | 5 |

Dashes are not failures. The section 8c runs for plain seeds 0 to 2 were made before the held-out transforms existed, so those three cells have three seeds, not six. They were not rerun.

Pooled, with the eps 0 control from the same runs:

| Transform | Plain | Compression-aware | Plain control | Aware control |
|---|---|---|---|---|
| uint8, no transform | 26/30 | 23/30 | 6/30 | 6/30 |
| JPEG 90 | 20/30 | 21/30 | 6/30 | 6/30 |
| JPEG 75 | 17/30 | 20/30 | 7/30 | 7/30 |
| JPEG 50 | 12/30 | 19/30 | 6/30 | 6/30 |
| Resize 64-32-64 | 11/30 | 21/30 | 6/30 | 6/30 |
| JPEG 60 (held out, seeds 3-5) | 8/15 | 11/15 | 4/15 | 6/30 as above |
| Resize 64-48-64 (held out, seeds 3-5) | 11/15 | 14/15 | 5/15 | 7/30 as above |
| Blur 0.5 (held out, seeds 3-5) | 14/15 | 14/15 | 5/15 | 6/30 as above |

For the three held-out rows the compression-aware arm restricted to the same seeds 3 to 5 is JPEG 60 11/15, resize 48 14/15, blur 14/15. Those are the numbers to compare against the plain column, and they are what the table shows.

Accuracy on the other test images is 0.776 to 0.810 across every cell and both arms. PSNR of the transformed poisons against the clean image, pooled: 33.4 to 36.0 dB for compression-aware, 34.0 to 37.2 dB for plain. Crafting cosine 0.979 to 0.988 for compression-aware and 0.987 to 0.994 for plain.

What six seeds change:

- The effect reported in 9i holds and is larger where compression is strongest. At JPEG 50 the compression-aware arm reaches 19/30 against 12/30, and at resize 21/30 against 11/30. Against controls of 6/30 those are the two clear gaps.
- At JPEG 90 and JPEG 75 the two arms are within one to three targets of each other, and with no transform plain is ahead, 26/30 against 23/30. The trade seen on Malimg is present here too, just small enough that ISIC can absorb it.
- The held-out transforms support the result rather than undercut it. On matched seeds the compression-aware arm is ahead at JPEG 60 and resize 48 and level at blur, which was already near the ceiling for both.
- "Flat at 8 of 15" from 9i was partly luck of three seeds. Over six seeds the compression-aware arm runs from 19/30 to 24/30 across transforms. That is much flatter than plain, which runs from 11/30 to 26/30, but it is not flat.
- Seed 2 is an outlier in both arms: near zero under every transform, in plain and compression-aware alike. Dropping it would raise both columns and would not change which is higher. It is kept in.

Noise, measured again on a larger sample. The "no transform" and "uint8" evaluations build bit-identical training data, so every trial that runs both is a paired measurement of victim-training noise. Over the 90 trials in these runs that evaluated both, 3 disagreed, which is 3%. On a 30-target cell that is about 1 target. This revises the estimate in 9i, which came from 15 trials and suggested about 2 of 15. Use 3% and read differences of 1 to 2 targets in a 30-target cell as noise. The JPEG 50 and resize gaps, 7 and 10 targets, are well outside it.

Command (`s9k_seeds345.sh`):

```
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --skip-defense --fast --eps 0 8"
CT="none jpeg90 jpeg75 jpeg50 resize32"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"
for s in 3 4 5; do
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --out out_s9k_plain_s$s
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --craft-transforms $CT --out out_s9k_ct_s$s
done
```

`uint8` is in the evaluation list on top of `none` so that each trial yields a paired noise measurement. It costs one extra victim per trial and produces the 3% figure above.

### 9l. Where the perturbation energy sits (no training)

Seed 0, trial 0, 50 poisons, eps 8, ResNet-18 surrogate, linear mode. Only 8 poisons per run were
stored in `examples.pt`, so one seed was regenerated with the same base and target selection the
real runs make. Each perturbation is cut into 8x8 blocks, the 2D DCT is taken per block and per
channel, and the 64 coefficients are split by `u+v`. Energy is the sum of squared coefficients in
a band divided by the total. The bands hold 10, 26 and 28 coefficients, so a flat spectrum is not
an equal split and is given as a reference row.

| Dataset | Arm | Low (u+v<=3) | Mid (4 to 7) | High (u+v>=8) | Mean abs perturbation, gray levels | L-inf | Crafting cosine |
|---|---|---|---|---|---|---|---|
| Malimg gray | plain | 0.247 | 0.442 | 0.311 | 4.73 | 8.0 | 0.985 |
| Malimg gray | compression-aware | 0.353 | 0.405 | 0.242 | 4.82 | 8.0 | 0.970 |
| ISIC | plain | 0.285 | 0.412 | 0.303 | 4.25 | 8.0 | 0.995 |
| ISIC | compression-aware | 0.413 | 0.355 | 0.232 | 4.50 | 8.0 | 0.988 |
| reference | flat spectrum | 0.156 | 0.406 | 0.438 | | | |

Both plain arms are already biased toward low frequencies relative to a flat spectrum.
Compression-aware crafting moves a further 0.11 (Malimg) to 0.13 (ISIC) of the total energy into
the low band. That gain comes from both other bands, not from the high band alone: the high band
loses 0.069 on Malimg and 0.071 on ISIC, and the mid band loses 0.037 on Malimg and 0.057 on ISIC.
The mean absolute perturbation barely changes, so the
compression-aware poisons are not using more of the eps budget.

The shift happens on both datasets by a similar amount, but the survival benefit in sections 9b to
9k appeared only on ISIC and not on Malimg. The spectral shift and the survival are both measured;
no link between them was measured, and none is claimed here.

Command:

```
venv/bin/python -u s9l_part_d.py     # output in part_d.log and out_part_d.json
```

### 9m. Restarts and step count (Malimg grayscale, null result)

Seed 0 pilot, plain crafting, eps 8, 50 poisons, 5 targets, ResNet-18 frozen, `--fast` victims.
Pass bar set before the runs: beat restarts 1 at JPEG 75 or at resize 64-32-64 by at least
2 targets out of 5. Nothing met it, so seeds 1 and 2 and the `--craft-transforms` repeat were not
run.

| Setting | Crafting cosine | No transform | uint8 | JPEG 75 | Resize 64-32-64 |
|---|---|---|---|---|---|
| restarts 1, 100 steps (baseline) | 0.9747 to 0.9789 | 3/5 | 3/5 | 0/5 | 0/5 |
| restarts 4, 100 steps | 0.9800 to 0.9814 | 3/5 | 4/5 | 0/5 | 0/5 |
| restarts 8, 100 steps | 0.9796 to 0.9814 | 2/5 | 3/5 | 0/5 | 0/5 |
| restarts 1, 250 steps | 0.9834 to 0.9857 | 5/5 | 5/5 | 0/5 | 0/5 |
| eps 0 control | n/a | 0/5 | 0/5 | 0/5 | 0/5 |

Accuracy on the other test images is 0.902 to 0.908 in every cell, including the controls.
PSNR of the shipped poison against the clean image is 33.2 to 33.6 dB with no transform,
31.8 to 32.1 dB after JPEG 75 and 25.4 dB after the resize.

More restarts and more steps both raise the crafting cosine, and 250 steps raised the
untransformed attack from 3 of 5 to 5 of 5. Neither moved JPEG 75 or the resize off zero. The
Malimg result in section 8b therefore does not look like a crafting-budget problem at this eps,
though all that is established is that these two budget knobs do not fix it.

Commands:

```
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --skip-defense --fast"
TF="none uint8 jpeg75 resize32"
venv/bin/python -u run_experiment.py $MAL $BASE --seed 0 --eps 0 8 --transforms $TF --restarts 1 --out out_s9m_r1
venv/bin/python -u run_experiment.py $MAL $BASE --seed 0 --eps 8 --transforms $TF --restarts 4 --out out_s9m_r4
venv/bin/python -u run_experiment.py $MAL $BASE --seed 0 --eps 8 --transforms $TF --restarts 8 --out out_s9m_r8
venv/bin/python -u run_experiment.py $MAL $BASE --seed 0 --eps 8 --transforms $TF --attack-steps 250 --out out_s9m_s250
```

### 9o. Poison selection by gradient cosine

`--select-base` replaces the random draw of 50 base-class images with the 50 whose head gradient
`grad_theta CE(f(x), base)` has the highest cosine with the attacker's target gradient
`grad_theta CE(f(X_target), base)`. Head only, which is what the victim trains in linear mode.
The selection is recomputed per trial because it depends on the target images. The flag still
performs the random draw it replaces, so the generator state and the eps 0 controls stay
comparable between the two arms. Crafting after selection is unchanged.

Seeds 0 to 2, 5 targets per seed, eps 8, 50 poisons, plain crafting. Targets missed out of 15,
control in the same row.

| Dataset | Arm | No transform | uint8 | JPEG 75 | Resize 64-32-64 |
|---|---|---|---|---|---|
| Malimg gray | random | 7/15 | 7/15 | 0/15 | 0/15 |
| Malimg gray | selected | 7/15 | 6/15 | 0/15 | 0/15 |
| Malimg gray | control, random | 0/15 | 0/15 | 0/15 | 0/15 |
| Malimg gray | control, selected | 0/15 | 0/15 | 0/15 | 0/15 |
| ISIC | random | 11/15 | 13/15 | 6/15 | 4/15 |
| ISIC | selected | 14/15 | 14/15 | 11/15 | 7/15 |
| ISIC | control, random | 1/15 | 1/15 | 2/15 | 2/15 |
| ISIC | control, selected | 1/15 | 1/15 | 2/15 | 2/15 |

Per seed, out of 5:

| Dataset | Arm | Transform | s0 | s1 | s2 |
|---|---|---|---|---|---|
| Malimg gray | random | uint8 | 4 | 2 | 1 |
| Malimg gray | selected | uint8 | 3 | 3 | 0 |
| ISIC | random | uint8 | 5 | 5 | 3 |
| ISIC | selected | uint8 | 5 | 5 | 4 |
| ISIC | random | JPEG 75 | 2 | 4 | 0 |
| ISIC | selected | JPEG 75 | 5 | 4 | 2 |
| ISIC | random | resize32 | 2 | 2 | 0 |
| ISIC | selected | resize32 | 5 | 2 | 0 |

On ISIC selection helps, and it helps most under transforms: JPEG 75 goes from 6 of 15 to 11 of 15
and the resize from 4 of 15 to 7 of 15, against identical controls. Crafting cosine also rises,
0.9885 to 0.9960 selected against 0.9818 to 0.9943 random. The gain is carried by seeds 0 and 2;
seed 1 is unchanged under both transforms.

On Malimg selection does nothing, and the reason was measured. The per-image gradient cosine is
constant across all 225 Yuner.A training images: minimum, all three quartiles and maximum are all
0.3290, with standard deviation below 1e-5. The reason was found afterwards and is in section 9q:
every Yuner.A image is the same image, so there is nothing for the selection to choose between.
The sentence in the first version of this section calling the result a second random draw was
wrong and 9q corrects it.

Accuracy on the other test images: 0.902 to 0.907 on Malimg, 0.783 to 0.790 on ISIC, in both arms
and both controls. PSNR of the shipped poison against the clean image: 33.5 dB on Malimg and
33.9 to 34.1 dB on ISIC with no transform.

Not tested: selection combined with compression-aware crafting, selection at other eps or poison
counts, selection on anything but the head gradient, and more than three seeds.

Commands:

```
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MAL $BASE --seed $s --eps 0 8 --transforms $TF --out out_s9o_mal_rand_s$s
  venv/bin/python -u run_experiment.py $MAL $BASE --seed $s --eps 0 8 --transforms $TF --select-base --out out_s9o_mal_sel_s$s
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --eps 0 8 --transforms $TF --out out_s9o_isic_rand_s$s
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --eps 0 8 --transforms $TF --select-base --out out_s9o_isic_sel_s$s
done
venv/bin/python -u s9o_sel_spread.py     # the gradient cosine spread quoted above
```

The flags-off equivalence check for `--restarts` and `--select-base` is
`s9_check_defaults_restarts.py`, which runs the new crafting function and a verbatim copy of the
previous loop in one single-threaded process. With `--restarts 1` and `--select-base` off the
poisons are bit-identical on both datasets, plain and compression-aware: maximum pixel difference
0.0 gray levels and crafting cosines equal to 10 decimal places.

### 9p. Per-target outcomes

Per-target recording was added for these runs: each trial now stores the victim predictions for
every evaluation transform and the clean model's softmax confidence on each target's true class.
Sections 8 and 9 up to 9k stored per-target predictions for the untransformed victim only; their
per-transform records hold just the mean over the targets, and clean confidence was never stored.
Those runs were not regenerated, so no per-target or per-confidence figures exist for them.

The full table is 145 rows, one per target per run across 9m and 9o at both eps values, written to
`s9p_per_target.csv` with columns for part, setting, run, seed, eps, trial, target test index,
clean confidence on the true class, and whether the target was missed under each of no transform,
uint8, JPEG 75 and resize 64-32-64. It is one target per trial and five trials per run, which is
where 145 comes from rather than a larger number.

Pooled over both datasets, miss rate rises sharply as clean confidence falls:

| Clean confidence on true class | Targets | No transform | uint8 | JPEG 75 | Resize 64-32-64 |
|---|---|---|---|---|---|
| 0.50 to 0.90 | 24 | 20/24 | 22/24 | 14/24 | 10/24 |
| 0.90 to 0.99 | 34 | 20/34 | 21/34 | 3/34 | 1/34 |
| 0.99 and above | 22 | 12/22 | 12/22 | 0/22 | 0/22 |

That pooled table is confounded and should not be used. Split by dataset, at eps 8:

| Dataset | Clean confidence | Targets | No transform | uint8 | JPEG 75 | Resize 64-32-64 |
|---|---|---|---|---|---|---|
| Malimg gray | below 0.90 | 0 | n/a | n/a | n/a | n/a |
| Malimg gray | 0.90 to 0.99 | 28 | 15/28 | 16/28 | 0/28 | 0/28 |
| Malimg gray | 0.99 and above | 22 | 12/22 | 12/22 | 0/22 | 0/22 |
| ISIC | below 0.90 | 24 | 20/24 | 22/24 | 14/24 | 10/24 |
| ISIC | 0.90 to 0.99 | 6 | 5/6 | 5/6 | 3/6 | 1/6 |
| ISIC | 0.99 and above | 0 | n/a | n/a | n/a | n/a |

Almost every Malimg target sits above 0.90 confidence and almost every ISIC target sits below it,
so the pooled trend is mostly the Malimg against ISIC difference already reported in sections 8b
and 8c, not a confidence effect. Within Malimg, confidence makes no clear difference: 15 of 28
against 12 of 22 untransformed, and zero under both transforms in both bins. Within ISIC the low
bin does worse than the high bin under JPEG 75 (14 of 24 against 3 of 6), but the high bin holds
6 targets, which is too few to rest anything on.

The eps 0 controls show the same split: at ISIC confidence below 0.90, 2 of 24 targets are missed
with no transform and 4 of 24 under each transform, so a few low-confidence ISIC targets flip
without any poisoning at all. Every Malimg control cell is 0.

Noise note, carried over from 9k: `none` and `uint8` produce bit-identical tensors, so every trial
that evaluates both is a paired measurement of victim-training noise. Over the 90 trials of 9k,
3 disagreed, about 3%, or roughly 1 target in 30. Differences at or below that size in 9m and 9o
should not be read as effects. This applies to the Malimg selected against random comparison in
9o, where the gap is 1 target in 15.

Commands:

```
venv/bin/python s9_ac_summary.py
venv/bin/python s9p_per_target.py
```

Not tested across 9l to 9p: a second victim architecture or an ensemble surrogate (Part B was not
run, the only pretrained weights available offline here are ResNet-18 and ResNet-50, which cannot
make an ensemble that excludes both victims), restarts or step count on ISIC, restarts combined
with selection, and full fine-tuning instead of linear mode.

### 9q. Why selection does nothing on Malimg: the base class is one image

Section 9o reported that the head-gradient selection score is constant across all 225 Yuner.A
training images and did not explain it. The cause was measured and it is not softmax saturation.

First the score itself, surrogate at seed 0, 50-poison setting, 225 Yuner.A training images:

| Quantity | Malimg, Yuner.A | ISIC, benign |
|---|---|---|
| Images in the base class | 225 | 1125 |
| softmax p(base), min / median / max | 0.580523 / 0.580523 / 0.580524 | 0.075 / 0.881 / 0.997 |
| Residual norm, min / median / max | 0.5751 / 0.5751 / 0.5751 | 0.0048 / 0.168 / 1.308 |
| Feature norm, min / median / max | 35.074 / 35.074 / 35.074 | 25.251 / 45.449 / 94.720 |
| Weight block over bias block, median ratio | 35.1 | 45.4 |
| Head-gradient cosine, min / max / std | 0.330172 / 0.330172 / 6.0e-09 | 0.365 / 0.825 / 7.8e-02 |
| Pairwise cosine of feature directions, minimum over pairs | 1.000000 | 0.167 |

The cosine reads 0.330172 here and 0.3290 in section 9o. The surrogate is retrained in each
process and CPU training is nondeterministic, which is enough to account for that difference. The
constancy across images is the point, not the value.

Not saturation: p(base) is 0.58, nowhere near 1, and the residual norm is a healthy 0.575. Not the
bias term either: the weight block of the gradient is 35 times larger than the bias block, the same
ratio as on ISIC where the score works fine.

The score is constant because its input is constant. On Malimg the feature norm is identical to
three decimals across all 225 images and the pairwise cosine of feature directions is exactly
1.000000 for every pair, so the frozen backbone maps every Yuner.A training image to the same
feature vector. Checking the pixels explains that:

| Set | Images | Distinct at 64x64 | Max pixel spread |
|---|---|---|---|
| Yuner.A training split | 225 | 1 | 0 gray levels |
| All Yuner.A files in the folder | 800 | 1 | 0 gray levels |
| Yuner.A at native resolution, spread sample of 20 | 20 | 1 | 0 gray levels |
| ISIC benign training split | 1125 | 1124 | 246 gray levels |

Every Yuner.A file is the same image, pixel for pixel, at the native 768x683 resolution as well as
at 64x64. The 800 files carry 133 distinct md5 sums, so they differ as files, but only in PNG
encoding or metadata, not in pixels. This is a property of the dataset, not of the 64x64 resize.

Across the whole of Malimg, sampling up to 300 files per class and counting distinct 64x64 images:

| Class | Files | Sampled | Distinct |
|---|---|---|---|
| Yuner.A | 800 | 300 | 1 |
| Autorun.K | 106 | 106 | 1 |
| Fakerean | 381 | 300 | 30 |
| Adialer.C | 122 | 122 | 44 |
| Rbot!gen | 158 | 158 | 135 |
| Lolyda.AT | 159 | 159 | 153 |
| VB.AT (the target class) | 408 | 300 | 300 |
| the other 18 classes | | | all distinct |
| total | | 4579 | 3798 |

Two of the 25 classes collapse to a single image and two more are heavily duplicated. The target
class VB.AT is fully distinct, so the target side of the attack is unaffected.

Consequences, stated as narrowly as the measurements allow:

- The Malimg selection result in 9o is not evidence about selection. Any 50 of 225 identical
  images are the same 50 images, so `--select-base` cannot change the base set on this class. With
  exact ties `topk` returns the lowest 50 indices, so it is not even a second random draw as 9o
  says. The 1-target difference in 9o is the victim-training noise floor from 9k.
- Every Malimg targeted run in sections 8b, 9m, 9o and 9s uses 50 copies of one base image,
  each carrying its own perturbation. The attack is still clean label and the poisons are still
   50 distinct files, but the base diversity is nil, unlike ISIC.
- This was not measured as the cause of anything else. In particular, whether the degenerate base
  class is why Malimg poisons die under JPEG 75 and resize (sections 8b, 9m) was not tested. It is
  the first thing to test next, by rerunning the Malimg attack with a base class that has
  distinct images.

Alternative scores were tried as the task asked, and none of them varies on Malimg either, which
follows from the features being identical:

| Score | Malimg std | Malimg top-50 overlap | ISIC std | ISIC spearman with head gradient |
|---|---|---|---|---|
| head gradient, temperature 1 | 6.0e-09 | reference | 7.8e-02 | reference |
| head gradient, temperature 2 | 1.2e-08 | 50/50 | 7.8e-02 | 1.000 |
| head gradient, temperature 5 | 0.0 | 41/50 | 7.8e-02 | 0.999 |
| head gradient, temperature 10 | 6.0e-08 | 41/50 | 7.7e-02 | 0.999 |
| feature cosine to the target mean | 6.0e-08 | 41/50 | 7.7e-02 | 0.998 |

The Malimg overlap figures are meaningless, they are tie-breaking order among identical images.
On ISIC all four scores rank the base images almost identically, spearman 0.998 or better, so the
choice of score is not what makes selection work there.

Because the ranking does not vary on Malimg, the Malimg poisoning run with selection was not
performed. Selecting among 225 copies of one image cannot pick anything.

Commands:

```
venv/bin/python -u s9q_sel_diag.py
venv/bin/python -u s9q_base_identity.py
```

The per-class distinct-image count was produced by a one-off loop over `malimg_all`, counting
unique 64x64 arrays per class with the same sampling the loader uses.

### 9r. ISIC: selection on top of compression-aware crafting

ISIC, ResNet-18 frozen, eps 8, 50 poisons, 5 targets per seed, seeds 0 to 5,
`--craft-transforms none jpeg90 jpeg75 jpeg50 resize32`, same protocol and evaluation transforms as
9k including the three held out from crafting. Two new arms, both with `--select-base`, one at the
default 100 crafting steps and one at 250. The compression-aware arm on randomly drawn base images
is the 9k aware column and was not rerun; it is cited below.

Targets missed out of 30, with the eps 0 control from the same runs:

| Transform | 9k plain, random | 9k aware, random | 9r (a) aware, selected | 9r (b) aware, selected, 250 steps | 9r control |
|---|---|---|---|---|---|
| uint8, no transform | 26/30 | 23/30 | 26/30 | 26/30 | 4/30 |
| JPEG 90 | 20/30 | 21/30 | 25/30 | 26/30 | 4/30 |
| JPEG 75 | 17/30 | 20/30 | 25/30 | 25/30 | 4/30 |
| JPEG 50 | 12/30 | 19/30 | 25/30 | 25/30 | 3/30 |
| Resize 64-32-64 | 11/30 | 21/30 | 25/30 | 25/30 | 3/30 |
| JPEG 60 (held out) | 8/15, s3-5 | 11/15, s3-5 | 25/30 | 25/30 | 4/30 |
| Resize 64-48-64 (held out) | 11/15, s3-5 | 14/15, s3-5 | 26/30 | 26/30 | 4/30 |
| Blur 0.5 (held out) | 14/15, s3-5 | 14/15, s3-5 | 26/30 | 26/30 | 4/30 |

For the held-out rows the 9k columns cover seeds 3 to 5 only. On those same three seeds, 9r (a) is
JPEG 60 13/15, resize 48 14/15, blur 14/15, and 9r (b) is identical. The 9k aware figures to
compare against are 11/15, 14/15 and 14/15.

Targets missed out of 5, per seed:

| Transform | (a) s0 | s1 | s2 | s3 | s4 | s5 | (b) s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uint8, no transform | 5 | 5 | 2 | 5 | 4 | 5 | 5 | 5 | 2 | 5 | 4 | 5 |
| JPEG 90 | 5 | 5 | 2 | 5 | 3 | 5 | 5 | 5 | 2 | 5 | 4 | 5 |
| JPEG 75 | 5 | 5 | 2 | 5 | 3 | 5 | 5 | 5 | 2 | 5 | 3 | 5 |
| JPEG 50 | 5 | 5 | 2 | 5 | 3 | 5 | 5 | 5 | 2 | 5 | 3 | 5 |
| Resize 64-32-64 | 5 | 5 | 2 | 5 | 3 | 5 | 5 | 5 | 2 | 5 | 3 | 5 |
| JPEG 60 (held out) | 5 | 5 | 2 | 5 | 3 | 5 | 5 | 5 | 2 | 5 | 3 | 5 |
| Resize 64-48-64 (held out) | 5 | 5 | 2 | 5 | 4 | 5 | 5 | 5 | 2 | 5 | 4 | 5 |
| Blur 0.5 (held out) | 5 | 5 | 2 | 5 | 4 | 5 | 5 | 5 | 2 | 5 | 4 | 5 |

Crafting cosine 0.9811 to 0.9939 for arm (a) and 0.9878 to 0.9963 for arm (b). Base selection
cosine, averaged over trials: 0.7287 over the 50 chosen against 0.5080 over all 1125 benign
training images. Accuracy on the other test images 0.783 to 0.796 in every cell including controls.
PSNR of the transformed poisons against the clean image 33.1 to 36.3 dB.

What this shows:

- Selection on top of compression-aware crafting is close to flat across all eight transforms,
  25 to 26 of 30, against 19 to 23 of 30 for compression-aware crafting on random base images.
  The three transforms held out of crafting behave like the three used in it.
- Against the 9k aware column the gain is largest where 9k was weakest: JPEG 50 19/30 to 25/30,
  JPEG 75 20/30 to 25/30, resize 21/30 to 25/30, JPEG 90 21/30 to 25/30, no transform 23/30 to
  26/30. Selection also removes the one place compression-aware crafting lost to plain crafting:
  9k plain reached 26/30 with no transform against the aware arm's 23/30, and 9r is back at 26/30
  while staying at 25/30 under every transform.
- 250 crafting steps raise the cosine and change the outcome by at most one target in any row.
  On ISIC the step count is not a useful knob, the same conclusion 9m reached on Malimg.
- Seed 2 is again the weak seed, 2 of 5 in nearly every cell of both arms, as in 9i and 9k. Seeds
  0, 1, 3 and 5 are at 5 of 5 almost everywhere, so most of the remaining headroom is seed 2 and
  seed 4.

A caution on comparing across runs. The eps 0 control here is 3 to 4 of 30 where the 9k control was
6 to 7 of 30. At eps 0 with `--fast` the poisons are the unmodified base images, so the victim's
training set is set-identical to the clean one and differs only in the order of the rows. The
control spread between runs is therefore pure victim-training noise, and it is about 2 to 3 targets
in 30, larger than the 1 in 30 that the paired `none` against `uint8` probe in 9k measures within a
run. Read against that slack, the JPEG 50, JPEG 75 and resize gains over 9k (6, 5 and 4 targets)
are larger than the noise, while the JPEG 90 gain (4) and the no-transform difference (3) are
closer to it.

Not tested: selection with compression-aware crafting on Malimg (impossible on the Yuner.A base
class, see 9q), any eps or poison count other than eps 8 and 50, and whether the defenses in
section 3 catch selected poisons any better or worse.

Commands:

```
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --skip-defense --fast --eps 0 8"
CT="none jpeg90 jpeg75 jpeg50 resize32"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"
for s in 0 1 2 3 4 5; do
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --craft-transforms $CT \
    --select-base --out out_s9r_sel_s$s
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --craft-transforms $CT \
    --select-base --attack-steps 250 --out out_s9r_sel250_s$s
done
```

### 9s. Malimg: does the 250-step result hold on three seeds?

9m found 5 of 5 at 250 steps on seed 0 alone, against 3 of 5 at 100 steps. Malimg grayscale, plain
crafting, eps 8, 50 poisons, seeds 0 to 2, no transform and uint8 only.

| Setting | Crafting cosine | No transform | uint8 | Per seed, uint8 |
|---|---|---|---|---|
| 100 steps, random base (the 9o arm) | 0.9608 to 0.9790 | 7/15 | 7/15 | 4, 2, 1 |
| 250 steps | 0.9710 to 0.9865 | 14/15 | 12/15 | 5, 4, 3 |
| eps 0 control, 250 steps | n/a | 0/15 | 0/15 | 0, 0, 0 |

It holds, and more than seed 0 suggested. Seed 0 repeats at 5 of 5, and the weak seeds improve the
most: seed 1 goes 2 to 4 and seed 2 goes 1 to 3. Pooled on uint8 that is 7 of 15 against 12 of 15,
well outside the 1 in 30 noise floor. Accuracy on the other test images is 0.903 to 0.907, and the
eps 0 control stays at zero.

Note that the cosine ranges overlap between the two settings, 0.9710 to 0.9865 at 250 steps against
0.9608 to 0.9790 at 100 steps, so cosine alone does not order these runs. Note also that 9m's seed 0
cosine range at 250 steps, 0.9834 to 0.9857, sits at the top of the range found here across three
seeds.

This does not change the compression picture. 9m measured 0 of 5 at JPEG 75 and 0 of 5 at resize for
the 250-step setting on seed 0, and this section did not evaluate those transforms. So the step
count moves the untransformed Malimg attack a long way and, on the one seed where it was measured,
not at all under compression.

Also unchanged by this section: all 50 poisons are still perturbations of a single base image, see
9q.

Commands:

```
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MAL $BASE --seed $s --transforms none uint8 --attack-steps 250 \
    --out out_s9s_mal250_s$s
done
```

### 9t. Duplicate audit of every Malimg class, and which sections it touches

Same loader sampling as the runs: shuffle each class folder with seed 0, keep at most 300 files,
resize to 64x64 bilinear, then the usual 75/25 split. "Distinct" counts unique 64x64 uint8 arrays
among the sampled files.

| Class | Files | Sampled | Distinct | Share | Train | Test |
|---|---|---|---|---|---|---|
| Adialer.C | 122 | 122 | 44 | 36.1% | 92 | 30 |
| Agent.FYI | 116 | 116 | 116 | 100.0% | 87 | 29 |
| Allaple.A | 2949 | 300 | 300 | 100.0% | 225 | 75 |
| Allaple.L | 1591 | 300 | 300 | 100.0% | 225 | 75 |
| Alueron.gen!J | 198 | 198 | 198 | 100.0% | 149 | 49 |
| Autorun.K | 106 | 106 | 1 | 0.9% | 80 | 26 |
| C2LOP.P | 146 | 146 | 146 | 100.0% | 110 | 36 |
| C2LOP.gen!g | 200 | 200 | 200 | 100.0% | 150 | 50 |
| Dialplatform.B | 177 | 177 | 177 | 100.0% | 133 | 44 |
| Dontovo.A | 162 | 162 | 162 | 100.0% | 122 | 40 |
| Fakerean | 381 | 300 | 30 | 10.0% | 225 | 75 |
| Instantaccess | 431 | 300 | 300 | 100.0% | 225 | 75 |
| Lolyda.AA1 | 213 | 213 | 213 | 100.0% | 160 | 53 |
| Lolyda.AA2 | 184 | 184 | 184 | 100.0% | 138 | 46 |
| Lolyda.AA3 | 123 | 123 | 123 | 100.0% | 93 | 30 |
| Lolyda.AT | 159 | 159 | 153 | 96.2% | 120 | 39 |
| Malex.gen!J | 136 | 136 | 136 | 100.0% | 102 | 34 |
| Obfuscator.AD | 142 | 142 | 142 | 100.0% | 107 | 35 |
| Rbot!gen | 158 | 158 | 135 | 85.4% | 119 | 39 |
| Skintrim.N | 80 | 80 | 80 | 100.0% | 60 | 20 |
| Swizzor.gen!E | 128 | 128 | 128 | 100.0% | 96 | 32 |
| Swizzor.gen!I | 132 | 132 | 132 | 100.0% | 99 | 33 |
| VB.AT | 408 | 300 | 300 | 100.0% | 225 | 75 |
| Wintrim.BX | 97 | 97 | 97 | 100.0% | 73 | 24 |
| Yuner.A | 800 | 300 | 1 | 0.3% | 225 | 75 |

Five classes fall below 90% distinct: Yuner.A 0.3%, Autorun.K 0.9%, Fakerean 10.0%, Adialer.C
36.1%, Rbot!gen 85.4%. The other twenty are at 96% or above, eighteen of them fully distinct.

Which earlier work this touches. Enumerated from the stored `cfg` of every `out_*/results.json`
rather than from the prose, so this is the set of runs that still exist on disk:

| Target | Base | Stored runs | Status |
|---|---|---|---|
| VB.AT | Yuner.A | 62 | base 0.3% distinct |
| Instantaccess | Yuner.A | 7 | base 0.3% distinct |

Every Malimg run in this repo uses Yuner.A as the base class, so every Malimg result rests on a
base class that is one image repeated. Both target classes used, VB.AT and Instantaccess, are
fully distinct, so the target side is unaffected everywhere. Sections carrying affected Malimg
results: 2a, 2b, 2c, the Malimg bullets of 4, the Malimg figures in 5, 8b, 8d, 8e, 8f, 8g, 9a to
9h, 9j, the Malimg rows of 9l, 9m, 9o and 9p, and 9s. No stored run uses Autorun.K, Fakerean,
Adialer.C or Rbot!gen as base or target.

Autorun.K appears once in this work, in the feasibility screen in `feasibility_malimg.log`, as
`Autorun.K 80 0.00 0.000`, which section 2b records as a near-duplicate quirk with "spread 0.00".
That screen measured the symptom. 9q and this section measure what it was: 1 distinct image out of
106 files.

Two statements in earlier sections are affected and cannot be edited there, so they are corrected
here:

- Section 2b calls Autorun.K and Yuner.A "near-duplicates". Within each of those two classes every
  image is identical, which is stronger than near-duplicate. Whether the two classes are identical
  to each other was not measured.
- Section 9i explains the Malimg against ISIC difference partly by "1137 benign training images
  against 225 Yuner.A images". The count is right but the comparison is not between 1137 and 225
  images, it is between 1125 distinct images and one image repeated 225 times. 9i's reasoning
  about optimizer room is not supported by any measurement in this work and should not be used.

Base class chosen for 9u: **Allaple.A**. It has 225 training images, the largest training split
among fully distinct classes, tied with Allaple.L, Instantaccess and VB.AT, and the tie is broken
on total files, 2949. VB.AT is kept as the target. A degeneracy check on the frozen features of the
training split, the same measurement as 9q:

| Class | Images | Feature norm min / max | Mean pairwise cosine of feature directions |
|---|---|---|---|
| Allaple.A | 225 | 18.738 / 36.544 | 0.8496 |
| Allaple.L | 225 | 19.803 / 31.685 | 0.9070 |
| Yuner.A | 225 | 35.074 / 35.074 | 1.0000 |

Allaple.A features vary, so the selection score has something to rank.

Command:

```
venv/bin/python -u s9t_class_audit.py
```

### 9u. Malimg with a non-degenerate base class: the attack stops working

Base class Allaple.A (225 training images, all distinct, chosen in 9t), target VB.AT, grayscale
perturbation, eps 8, 50 poisons (22.2% of the base class, the same fraction as the Yuner.A runs),
seeds 0 to 2, 5 targets per seed, 250 crafting steps throughout. Three arms as specified.

Targets missed out of 15:

| Arm | Crafting cosine | No transform | uint8 | JPEG 75 | Resize 64-32-64 | Per seed, uint8 |
|---|---|---|---|---|---|---|
| (a) random base, plain | 0.9310 to 0.9832 | 0/15 | 0/15 | 0/15 | 0/15 | 0, 0, 0 |
| (b) selected base, plain | 0.9642 to 0.9876 | 0/15 | 0/15 | 0/15 | 0/15 | 0, 0, 0 |
| (c) selected base, compression-aware | 0.9336 to 0.9742 | 0/15 | 0/15 | 0/15 | 0/15 | 0, 0, 0 |
| eps 0 control, all three arms | n/a | 0/15 | 0/15 | 0/15 | 0/15 | 0, 0, 0 |

Every cell is zero. Not one target moved, in any arm, under any transform.

Does the selection score vary now? Yes. Averaged over trials the base selection cosine is 0.3334
over the 50 chosen against 0.2651 over all 225, and 9t measured the mean pairwise cosine of the
frozen feature directions at 0.8496 rather than Yuner.A's exact 1.0000. So the score ranks real
differences here, and arms (b) and (c) are meaningful tests rather than tie-breaking. They still
produce nothing.

The null is not a broken setup. Compared against the Yuner.A run at the same 250 steps in 9s,
seed 0:

| | Yuner.A base (9s) | Allaple.A base (9u) |
|---|---|---|
| Clean victim accuracy | 0.901 | 0.901 |
| Clean victim recall on the target class | 1.000 | 1.000 |
| Clean confidence on the five targets | 0.985, 0.998, 0.998, 0.983, 0.977 | the same five values |
| Poisons as a share of the base class | 22.2% | 22.2% |
| Crafting cosine over the five trials | 0.9832 to 0.9865 | 0.9616 to 0.9814 |
| Target predictions after poisoning | base class, all 5 | target class, all 5 |

The target images are literally the same five images, since the seed and the target pool are
unchanged. Gradient matching reaches a comparable cosine. The poisons simply have no effect.

What this does and does not establish:

- The Malimg targeted result is specific to the base class. Swapping Yuner.A for Allaple.A, with
  everything else held fixed, takes it from 12 of 15 (9s) to 0 of 15.
- It does not establish that duplication is the cause. Allaple.A and Yuner.A differ in more than
  duplicate structure: different malware family, different feature geometry, different distance
  from VB.AT in feature space. None of those was isolated.
- The decisive test, not run here, is Allaple.A with its 225 training images replaced by 225
  copies of a single Allaple.A image. That changes only the duplication and holds the family and
  the feature geometry fixed. It is three runs, about 30 minutes. Until it is done, the honest
  statement is that the base class matters enormously and the reason is unmeasured.
- Crafting cosine is lower and wider on the distinct base class, 0.931 to 0.983 against 0.983 to
  0.987 on Yuner.A, so the optimizer does have a harder time. Whether that accounts for any of the
  gap was not measured, and the arm that reached the highest cosine here, (b) at 0.9642 to 0.9876,
  still scored zero.

For the talk, the safe version of this is that the Malimg case study rests on one base class, a
different and fully distinct base class produces nothing, and the reason is still open. Section 3's
ISIC results are untouched by this, since ISIC benign is 1124 distinct images out of 1125.

Commands:

```
MAL="--data malimg_all --target VB.AT --base Allaple.A --max-per-class 300 --gray-delta"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --fast --eps 0 8"
CT="none jpeg90 jpeg75 jpeg50 resize32"
TF="none uint8 jpeg75 resize32"
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MAL $BASE --skip-defense --seed $s --transforms $TF \
    --attack-steps 250 --out out_s9u_a_rand_s$s
  venv/bin/python -u run_experiment.py $MAL $BASE --skip-defense --seed $s --transforms $TF \
    --attack-steps 250 --select-base --out out_s9u_b_sel_s$s
  venv/bin/python -u run_experiment.py $MAL $BASE --skip-defense --seed $s --transforms $TF \
    --attack-steps 250 --select-base --craft-transforms $CT --out out_s9u_c_selct_s$s
done
```

### 9v. Defenses against the strongest ISIC attack

9r arm (a) rerun on seeds 0 to 2 with the defenses on, same filters and settings as sections 2 and
3: Deep k-NN at k=15, and spectral signatures removing 5, 10 and 20% per class. The
filters see only the poisoned training set and the victim trained on it, and the poisons they see
are the untransformed ones, as in section 3a. Targets missed before filtering, 12 of 15, which
reproduces the seeds 0 to 2 part of 9r.

| Filter | Caught, selected and aware | Caught, plain (3a) | Clean cut, aware | Clean cut, plain (3a) | Missed after, aware | Missed after, plain (3a) |
|---|---|---|---|---|---|---|
| Deep k-NN | 2% | 1% | 22% | 22% | 14/15 | 5/5 |
| Spectral 5% | 7% | 24% | 5% | 4% | 15/15 | 5/5 |
| Spectral 10% | 11% | 31% | 10% | 9% | 13/15 | 5/5 |
| Spectral 20% | 18% | 39% | 20% | 19% | 12/15 | 4/5 |

Section 3a is seed 0 only with 5 targets, so its "missed after" column is out of 5 and its catch
rates rest on one seed. The aware columns are three seeds and 15 targets.

Missed after filtering, per seed, out of 5: Deep k-NN 5, 5, 4. Spectral 5% 5, 5, 5. Spectral 10%
4, 5, 4. Spectral 20% 3, 5, 4. Accuracy on the other test images after filtering: 0.784 for Deep
k-NN, then 0.778, 0.762 and 0.759 as the spectral fraction rises.

What this shows:

- Spectral signatures catch roughly half as many of these poisons as of plain ones, 18% against
  39% at the 20% setting and 7% against 24% at 5%, while cutting the same share of clean data.
  This is the opposite direction from 9e, where spectral caught slightly more of the
  compression-aware Malimg poisons than of plain ones.
- No filter stops the attack. The strongest, spectral at 20%, leaves 12 of 15 targets missed,
  which is the same as the 12 of 15 before filtering.
- Spectral at 5% leaves 15 of 15, higher than the 12 of 15 before filtering. Removing 5% of each
  class made the attack look stronger. That is 3 targets on a noise floor of about 1 in 30 per the
  9k probe, so it is more than noise, but what it is was not measured. Do not present it as the
  filter helping the attacker without testing it directly.
- Deep k-NN remains useless here and expensive, 2% caught for 22% of clean data cut, matching its
  behaviour in sections 2 and 3.

So the detection finding holds and gets stronger: against the best attack in this work the
one filter that did something now catches 18% of poisons for a fifth of the data and does not stop
a single additional target.

Not tested: the filters against the transformed poisons rather than the untransformed ones, any
filter beyond the two tested here, and whether selection is what reduced the spectral catch rate,
since 9v changes both selection and compression-aware crafting against 3a at once.

Commands:

```
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --craft-transforms $CT \
    --select-base --spectral-frac 0.05 0.1 0.2 --out out_s9v_def_s$s
done
venv/bin/python s9v_summary.py
```

### 9w. Isolating duplication: Allaple.A collapsed to one repeated image

9u left two candidate explanations for why Yuner.A works and Allaple.A does not: Yuner.A is one
image repeated 225 times, and the two classes sit at different places in feature space. This
section changes only the first.

Setup is arm (a) of 9u exactly: Malimg, target VB.AT, base Allaple.A, grayscale perturbation,
eps 8, 50 poisons, 250 crafting steps, random base draw, seeds 0 to 2, 5 targets per seed. The one
change is that all 225 Allaple.A training images are replaced by 225 copies of a single Allaple.A
image before anything else runs.

Rule for the single image: the base-class training image whose frozen-backbone feature is closest
to the class mean feature, that is the medoid of the class under the same ResNet-18 features the
victim trains on. For Allaple.A at seed 0 that is training index 372. Verified after the
substitution: the base class holds 1 distinct 64x64 array in 225 slots, pixel spread 0.0 gray
levels, and the other 3215 training images are bit-identical to the unmodified run.

Targets missed out of 15:

| Setting | Crafting cosine | No transform | uint8 | JPEG 75 | Resize 64-32-64 | Per seed, uint8 |
|---|---|---|---|---|---|---|
| Allaple.A, original 225 distinct images (9u arm a) | 0.9310 to 0.9832 | 0/15 | 0/15 | 0/15 | 0/15 | 0, 0, 0 |
| Allaple.A, collapsed to 225 copies of one image | 0.9407 to 0.9803 | 4/15 | 3/15 | 0/15 | 0/15 | 2, 1, 0 |
| Yuner.A, naturally one repeated image (9s) | 0.9710 to 0.9865 | 14/15 | 12/15 | not tested | not tested | 5, 4, 3 |
| eps 0 control, collapsed run | n/a | 0/15 | 0/15 | 0/15 | 0/15 | 0, 0, 0 |

Collapsing the base class moves the untransformed attack from 0 of 15 to 3 or 4 of 15 against a
control of 0 of 15. That is above the roughly 1 in 30 noise floor, so duplication is doing
something. It recovers a fraction of Yuner.A's 12 of 15, not the whole gap, and the per-seed
pattern is uneven at 2, 1, 0. JPEG 75 and resize stay at 0 of 15, unchanged from every other
Malimg arm in this document.

Accuracy on the other test images is 0.889 to 0.891 here against 0.903 to 0.907 in the Yuner.A
runs. The collapsed run deletes 224 distinct training images from the dataset, so some drop is
expected; this was not isolated further.

Mean-feature distance from each base class to the VB.AT class mean, frozen ResNet-18 backbone,
training split, class means over the images actually used:

| Base class | L2 to VB.AT mean | Cosine to VB.AT mean |
|---|---|---|
| Yuner.A, all 225 training images | 29.037 | 0.6893 |
| Allaple.A, all 225 training images | 32.342 | 0.5336 |
| Allaple.A collapsed to the medoid | 32.858 | 0.5143 |

Yuner.A is closer to the target class than Allaple.A, so the class that works is the nearer one and
distance remains a live confound rather than being ruled out. Collapsing Allaple.A moves its mean
feature only slightly, 32.342 to 32.858, which is what makes this a reasonably clean duplication
test: the geometry barely moved and the result changed. Yuner.A's collapsed and full means are
identical to six figures, as they must be, since every image in the class is already the same.

What this establishes: duplication of the base class accounts for part of the Malimg effect and not
for all of it. What it does not establish: the size of the remaining part, or whether distance
explains it. 9x is the section that varies distance.

Commands:

```
MALBASE="--data malimg_all --target VB.AT --max-per-class 300 --gray-delta"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --skip-defense --fast --attack-steps 250"
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MALBASE --base "Allaple.A" $BASE --eps 0 8 --n-poison 50 \
    --seed $s --transforms none uint8 jpeg75 resize32 --base-collapse --out out_s9w_collapse_s$s
done
venv/bin/python s9w_feature_distances.py
```

### 9x. Isolating distance: the nearest and the farthest distinct base class

Candidate rule, applied to the 9t audit and the distances above: among Malimg classes with 100%
distinct images, at least 150 training images, and not the target class itself. That leaves five
candidates, ordered by L2 from the VB.AT class mean: C2LOP.gen!g 25.509, Lolyda.AA1 26.216,
Instantaccess 30.506, Allaple.A 32.342, Allaple.L 33.819. So the nearest is C2LOP.gen!g and the
farthest is Allaple.L. Allaple.A, already run in 9u, sits between them.

Same setup as 9w part one but with each class's original images: eps 8, 50 poisons, grayscale,
250 steps, random base, seeds 0 to 2, 5 targets per seed.

Targets missed out of 15:

| Base class | L2 to VB.AT | Crafting cosine | No transform | uint8 | JPEG 75 | Resize | Per seed, uint8 |
|---|---|---|---|---|---|---|---|
| C2LOP.gen!g, nearest | 25.509 | 0.9790 to 0.9930 | 3/15 | 1/15 | 0/15 | 0/15 | 1, 0, 0 |
| Allaple.A, middle (9u arm a) | 32.342 | 0.9310 to 0.9832 | 0/15 | 0/15 | 0/15 | 0/15 | 0, 0, 0 |
| Allaple.L, farthest | 33.819 | 0.9289 to 0.9772 | 1/15 | 1/15 | 0/15 | 0/15 | 1, 0, 0 |
| eps 0 control, nearest | 25.509 | n/a | 0/15 | 0/15 | 0/15 | 0/15 | 0, 0, 0 |
| eps 0 control, farthest | 33.819 | n/a | 0/15 | 0/15 | 0/15 | 0/15 | 0, 0, 0 |

This is a null. The nearest class scores 1 of 15 on uint8 and the farthest also 1 of 15, and both
are single trials on seed 0. 3 of 15 untransformed on C2LOP.gen!g is the largest cell and still sits
within a couple of targets of the noise floor. Nothing here orders the three distinct base classes
by distance, and none of them reaches the 3 of 15 that collapsing Allaple.A produced, let alone
Yuner.A's 12 of 15.

Crafting cosine is highest of any Malimg arm in this document on the nearest class, 0.9790 to 0.9930,
and it still produces almost nothing, which is one more case of cosine not predicting whether a run
lands.

Accuracy on the other test images is 0.905 to 0.909 across both classes and the controls, in line
with the other distinct-base runs.

Read with 9w: varying the base class over its full available distance range, with all images
distinct, moves the result by at most 1 target out of 15, while collapsing one of those same
classes to a single repeated image moves it by 3. Duplication is the factor that measurably
changed the outcome in this pair of sections; distance is not, over the range tested. This does not
account for the remaining gap to Yuner.A, which stays unexplained.

Commands:

```
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MALBASE --base "C2LOP.gen!g" $BASE --eps 0 8 --n-poison 50 \
    --seed $s --transforms none uint8 jpeg75 resize32 --out out_s9x_near_s$s
  venv/bin/python -u run_experiment.py $MALBASE --base "Allaple.L" $BASE --eps 0 8 --n-poison 50 \
    --seed $s --transforms none uint8 jpeg75 resize32 --out out_s9x_far_s$s
done
```

### 9y. More budget on the distinct base class

Arm (a) of 9u again, Allaple.A with its original distinct images, 250 steps, seeds 0 to 2, no
transform and uint8 only. Two larger budgets.

Targets missed out of 15:

| Setting | Crafting cosine | No transform | uint8 | Per seed, uint8 |
|---|---|---|---|---|
| eps 8, 50 poisons (9u arm a) | 0.9310 to 0.9832 | 0/15 | 0/15 | 0, 0, 0 |
| eps 16, 50 poisons | 0.9557 to 0.9876 | 0/15 | 0/15 | 0, 0, 0 |
| eps 8, 100 poisons | 0.9414 to 0.9854 | 0/15 | 0/15 | 0, 0, 0 |
| eps 0 control, eps 16 run | n/a | 0/15 | 0/15 | 0, 0, 0 |
| eps 0 control, 100-poison run | n/a | 0/15 | 0/15 | 0, 0, 0 |

Another null, and a flat one. Doubling the perturbation bound and doubling the poison count each
leave the attack at zero on all 30 victim trainings. 100 poisons is 44.4% of the Allaple.A training
class, so this is not a small-budget failure.

Accuracy on the other test images is 0.903 at eps 16 and 0.893 at 100 poisons, against 0.902 and
0.906 for the matching controls.

Budget is therefore not the limiting factor on a distinct Malimg base class, over the range tested.

Commands:

```
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MALBASE --base "Allaple.A" $BASE --eps 0 16 --n-poison 50 \
    --seed $s --transforms none uint8 --out out_s9y_eps16_s$s
  venv/bin/python -u run_experiment.py $MALBASE --base "Allaple.A" $BASE --eps 0 8 --n-poison 100 \
    --seed $s --transforms none uint8 --out out_s9y_n100_s$s
done
```

Noise floor for 9w to 9y: `transform_images(X, 'none')` returns the tensor unchanged, so every trial
that evaluates both the untransformed poisons and the "none" transform retrains the victim twice on
bit-identical data. Across the 150 such paired trials in these three sections, 3 disagreed, which is
2% and matches the roughly 1 target in 30 measured in 9k.
