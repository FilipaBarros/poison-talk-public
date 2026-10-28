set -x
cd "$(dirname "$0")"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --skip-defense --fast"
TF="none uint8 jpeg75 resize32"

# Part A: restarts and step count, Malimg grayscale, seed 0 pilot, plain crafting
venv/bin/python -u run_experiment.py $MAL $BASE --seed 0 --eps 0 8 --transforms $TF --restarts 1 --out out_s9m_r1 > s9m_r1.log 2>&1
venv/bin/python -u run_experiment.py $MAL $BASE --seed 0 --eps 8 --transforms $TF --restarts 4 --out out_s9m_r4 > s9m_r4.log 2>&1
venv/bin/python -u run_experiment.py $MAL $BASE --seed 0 --eps 8 --transforms $TF --restarts 8 --out out_s9m_r8 > s9m_r8.log 2>&1
venv/bin/python -u run_experiment.py $MAL $BASE --seed 0 --eps 8 --transforms $TF --attack-steps 250 --out out_s9m_s250 > s9m_s250.log 2>&1
echo PARTA_DONE

# Part C: poison selection, both datasets, seeds 0 to 2, random draw against gradient-cosine selection
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MAL $BASE --seed $s --eps 0 8 --transforms $TF --out out_s9o_mal_rand_s$s > s9o_mal_rand_s$s.log 2>&1
  venv/bin/python -u run_experiment.py $MAL $BASE --seed $s --eps 0 8 --transforms $TF --select-base --out out_s9o_mal_sel_s$s > s9o_mal_sel_s$s.log 2>&1
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --eps 0 8 --transforms $TF --out out_s9o_isic_rand_s$s > s9o_isic_rand_s$s.log 2>&1
  venv/bin/python -u run_experiment.py $ISIC $BASE --seed $s --eps 0 8 --transforms $TF --select-base --out out_s9o_isic_sel_s$s > s9o_isic_sel_s$s.log 2>&1
done
echo ALLDONE
