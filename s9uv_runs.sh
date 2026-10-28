set -x
cd "$(dirname "$0")"
MAL="--data malimg_all --target VB.AT --base Allaple.A --max-per-class 300 --gray-delta"
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --fast --eps 0 8"
CT="none jpeg90 jpeg75 jpeg50 resize32"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"
TF="none uint8 jpeg75 resize32"

# 9u: Malimg with a fully distinct base class, 250 crafting steps
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MAL $BASE --skip-defense --seed $s --transforms $TF \
    --attack-steps 250 --out out_s9u_a_rand_s$s > s9u_a_rand_s$s.log 2>&1
  venv/bin/python -u run_experiment.py $MAL $BASE --skip-defense --seed $s --transforms $TF \
    --attack-steps 250 --select-base --out out_s9u_b_sel_s$s > s9u_b_sel_s$s.log 2>&1
  venv/bin/python -u run_experiment.py $MAL $BASE --skip-defense --seed $s --transforms $TF \
    --attack-steps 250 --select-base --craft-transforms $CT --out out_s9u_c_selct_s$s > s9u_c_selct_s$s.log 2>&1
done
echo PART9U_DONE

# 9v: defenses on 9r arm (a), same filters and settings as the detection slides
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --transforms $ALLTF --craft-transforms $CT \
    --select-base --spectral-frac 0.05 0.1 0.2 --out out_s9v_def_s$s > s9v_def_s$s.log 2>&1
done
echo ALLDONE
