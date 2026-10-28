set -x
cd "$(dirname "$0")"
TF="uint8 jpeg90 jpeg75 jpeg50 resize32"
COMMON="--arch resnet50 --size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --eps 0 8 --skip-defense --fast --transforms $TF"
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
for s in 0 1 2; do venv/bin/python -u run_experiment.py $ISIC $COMMON --seed $s --out out_robust_r50_isic_s$s > robust_r50_isic_s$s.log 2>&1; done
for s in 0 1 2; do venv/bin/python -u run_experiment.py $MAL $COMMON --seed $s --out out_robust_r50_malimg_s$s > robust_r50_malimg_s$s.log 2>&1; done
echo ALLDONE
