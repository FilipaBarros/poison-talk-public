set -x
cd "$(dirname "$0")"
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --skip-defense --fast --eps 0 8"
CT="none jpeg90 jpeg75 jpeg50 resize32"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"
for s in 3 4 5; do
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --out out_s9k_plain_s$s > s9k_plain_s$s.log 2>&1
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --craft-transforms $CT --out out_s9k_ct_s$s > s9k_ct_s$s.log 2>&1
done
echo ALLDONE
