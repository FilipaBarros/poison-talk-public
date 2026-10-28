set -x
cd "$(dirname "$0")"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --skip-defense --fast --eps 0 8"
CT="none jpeg90 jpeg75 jpeg50 resize32"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"

# 9r: ISIC, compression-aware crafting on selected base images, with and without 250 steps
for s in 0 1 2 3 4 5; do
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --craft-transforms $CT \
    --select-base --out out_s9r_sel_s$s > s9r_sel_s$s.log 2>&1
done
echo ARM_A_DONE
for s in 0 1 2 3 4 5; do
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --craft-transforms $CT \
    --select-base --attack-steps 250 --out out_s9r_sel250_s$s > s9r_sel250_s$s.log 2>&1
done
echo ARM_B_DONE

# 9s: Malimg, plain crafting at 250 steps, three seeds, no transform and uint8 only
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MAL $BASE --seed $s --transforms none uint8 --attack-steps 250 \
    --out out_s9s_mal250_s$s > s9s_mal250_s$s.log 2>&1
done
echo ALLDONE
