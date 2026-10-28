# Mathematical Patterns of Poison

Clean-label data poisoning experiments on two image datasets, written for a 45 minute talk at
BSides Lisbon 2026 by Filipa Barros.

The attack is gradient matching in the style of Witches' Brew (Geiping et al. 2021): perturb a
handful of training images inside an L-infinity budget, leave their labels correct, and have the
victim misclassify one specific test image after training. The repository also contains a feature
collision baseline (Poison Frogs), a JPEG and resize robustness study, expectation-over-
transformation crafting, gradient-cosine poison selection, and two published detection filters.

Everything is CPU-only and small: 64x64 images, a frozen ImageNet backbone with only the final
linear layer retrained.

## Read this first

**The headline malware result rests on a degenerate base class.** Every Malimg experiment in
sections 2 and 8 to 9s of `RESULTS.md` draws its poisons from the Yuner.A family, and all 800
files in that family decode to one identical image. Sections 9q to 9y take it apart: with a fully
distinct base class the targeted attack drops from 12 of 15 targets to 0 of 15, raising the budget
does not recover it, and collapsing a distinct class to one repeated image recovers only 3 to 4 of
15. The ISIC results are unaffected, since the ISIC benign class is 1124 distinct images out of
1125.

That finding is kept in the repository rather than removed, because the audit that found it
(`s9t_class_audit.py`) is probably the most reusable thing here. If you work with Malimg, run it on
your copy before trusting anything.

`RESULTS.md` is the full record, written in the order the work was done. Later sections correct
earlier ones.

## Install

```
python -m venv venv
venv/bin/pip install -r requirements.txt
```

Python 3.9 and CPU PyTorch are enough. Pretrained ResNet-18 and ResNet-50 weights download from
torchvision on first use.

Datasets are not included. See `DATA.md`: Malimg must be obtained from its authors, ISIC 2020 is
fetched with `download_isic.py` and needs no API key. Check the ISIC per-image licenses in the
manifest before publishing any figure.

## Running an experiment

```
venv/bin/python run_experiment.py \
  --data malimg_all --target VB.AT --base Allaple.A --max-per-class 300 --gray-delta \
  --size 64 --epochs 5 --objective gradmatch \
  --n-targets 1 --trials 5 --n-poison 50 --eps 0 8 --attack-steps 250 \
  --transforms none uint8 jpeg75 resize32 --skip-defense --fast \
  --seed 0 --out out_demo
```

`--n-targets 1 --trials 5` attacks five separate target images, one per trial. `--eps 0 8` runs an
eps 0 control alongside the real attack. `--transforms` retrains an extra victim on poisons passed
through each named transform. `--fast` caches frozen-backbone features and is exact, not an
approximation, because the backbone never changes. `run_experiment.py --help` lists the rest.

The runner scripts in the repository root are the exact commands behind the reported numbers.

## Reproducing a number

Each row gives the slide, the section of `RESULTS.md` that holds the table, how to regenerate it,
and the run directories the numbers come from. The committed copies live in `results/raw/`; to
point the analysis scripts at them, restore the directory names they expect:

```
cp -R results/raw/out_* .
```

| Slide | Section | Regenerate with | Run output |
|---|---|---|---|
| 18 targeted malware | 2a | `robust_runs.sh`, or the command above with `--base Yuner.A` | `out_gm_targeted_vbat*`, `out_vbat_gray*` |
| 19 budget (eps and poison count) | 2a, 9y | `s9_stepd.sh`, `s9wxy_runs.sh` | `out_vbat_gray*`, `out_s9_d_*`, `out_s9y_eps16_s*`, `out_s9y_n100_s*` |
| 20 established families | 2b | `feasibility.py` | `out_gm_targeted`, `out_malware_linear` |
| 21 new family | 2b | `run_experiment.py --n-targets 0 --target-train-cap 10` | `out_newfam_s{0,1,2}`, `out_newfam_def_s{0,1,2}` |
| 22 fine-tuning | 2c | `run_experiment.py --mode finetune` | `out_vbat_finetune` |
| 23 duplicate audit | 9t | `s9t_class_audit.py` | printed to stdout |
| 24 base classes | 9u, 9w, 9x, 9y | `s9uv_runs.sh`, `s9wxy_runs.sh`, `s9w_feature_distances.py` | `out_s9u_*`, `out_s9w_collapse_s*`, `out_s9x_near_s*`, `out_s9x_far_s*` |
| 28 ISIC targeted | 3a | `robust_runs.sh` | `out_isic_gm_targeted`, `out_isic_targeted_s{1,2}`, `out_robust_isic_s*` |
| 29 ISIC class level | 3b | `run_experiment.py --n-targets 0 --victim-reps 3` | `out_isic_class_def_s{0..4}`, `out_isic_class_reps_s{0..4}` |
| 30 compression | 8b, 8c | `robust_runs.sh` | `out_robust_malimg_s*`, `out_robust_isic_s*` |
| 31 compression-aware | 9i, 9k, 9r | `s9_followup.sh`, `s9k_seeds345.sh`, `s9rs_runs.sh` | `out_s9_isic_ct_s*`, `out_s9k_*`, `out_s9r_sel_s*` |
| 34 to 37 detection | 2a, 3a, 9v | `s9uv_runs.sh`, then `s9v_summary.py` | `out_isic_def`, `out_vbat_def`, `out_s9v_def_s{0,1,2}` |

Analysis helpers that summarise several runs at once: `s9_ac_summary.py`, `robust_summary.py`,
`s9v_summary.py` (detection table), `s9w_feature_distances.py` (class mean-feature distances),
`s9l_part_d.py` (DCT band energy of the perturbation), `s9p_per_target.py` (per-target table),
`s9q_sel_diag.py` and `s9q_base_identity.py` (why selection does nothing on Malimg),
`s9o_sel_spread.py`, `s9_check_defaults_restarts.py` (checks that new flags default to the old
behaviour exactly).

## Results in this repository

```
results/per_target.csv      one row per target: run, seed, eps, trial, test index,
                            the clean model's confidence on the true class, and whether
                            the target was missed under each transform
results/dct_bands.json      low, mid and high 8x8 DCT band energy shares of the perturbation
results/raw/<run>/results.json
                            149 run records: config, clean model accuracy, per-trial
                            predictions, crafting cosine, transform arms, defense reports
```

`out_*/` is not committed. Each run directory also holds an `examples.pt` containing raw dataset
image pixels, and the ISIC images are not all redistributable.

## Cost

Measured on 8 CPU cores. A Malimg load is about 27 seconds, a surrogate about 42 seconds, and
crafting 50 poisons about 26 seconds at 100 steps. One three-seed five-target Malimg arm at 250
steps is roughly 30 minutes. The six-seed ISIC arms in section 9r are a few hours each.

## References

- Geiping et al., "Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching", 2021
- Shafahi et al., "Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks", 2018
- Peri et al., "Deep k-NN Defense Against Clean-Label Data Poisoning Attacks", 2020
- Tran et al., "Spectral Signatures in Backdoor Attacks", 2018
- Nataraj et al., "Malware Images: Visualization and Automatic Classification", 2011
- ISIC 2020 Challenge, International Skin Imaging Collaboration

## License

Dual licensed under MIT (`LICENSE-MIT`) or Apache License 2.0 (`LICENSE-APACHE`), at your option.
This covers the code and the result files only. The datasets carry their own licenses and are not
distributed here.
