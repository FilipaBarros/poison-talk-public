set -x
cd "$(dirname "$0")"
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --skip-defense --fast"
CT="none jpeg90 jpeg75 jpeg50 resize32"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $ISIC $BASE --eps 0 8 --seed $s --transforms $ALLTF --craft-transforms $CT --out out_s9_isic_ct_s$s > s9_isic_ct_s$s.log 2>&1
done
venv/bin/python -u run_experiment.py $MAL $BASE --arch resnet50 --eps 8 --seed 0 --transforms none jpeg75 resize32 --craft-transforms $CT --out out_s9_r50_ct > s9_r50_ct.log 2>&1
echo ALLDONE
