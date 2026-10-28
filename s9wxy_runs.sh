set -x
cd "$(dirname "$0")"
MALBASE="--data malimg_all --target VB.AT --max-per-class 300 --gray-delta"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --skip-defense --fast --attack-steps 250"
TF="none uint8 jpeg75 resize32"

# 9w: duplication isolated. Allaple.A with all 225 training images replaced by the class medoid.
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MALBASE --base "Allaple.A" $BASE --eps 0 8 --n-poison 50 \
    --seed $s --transforms $TF --base-collapse --out out_s9w_collapse_s$s > s9w_collapse_s$s.log 2>&1
done
echo PART9W_DONE

# 9x: distance isolated. Closest and farthest fully distinct base classes with 150+ training images.
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MALBASE --base "C2LOP.gen!g" $BASE --eps 0 8 --n-poison 50 \
    --seed $s --transforms $TF --out out_s9x_near_s$s > s9x_near_s$s.log 2>&1
  venv/bin/python -u run_experiment.py $MALBASE --base "Allaple.L" $BASE --eps 0 8 --n-poison 50 \
    --seed $s --transforms $TF --out out_s9x_far_s$s > s9x_far_s$s.log 2>&1
done
echo PART9X_DONE

# 9y: budget on Allaple.A, arm (a)
for s in 0 1 2; do
  venv/bin/python -u run_experiment.py $MALBASE --base "Allaple.A" $BASE --eps 0 16 --n-poison 50 \
    --seed $s --transforms none uint8 --out out_s9y_eps16_s$s > s9y_eps16_s$s.log 2>&1
  venv/bin/python -u run_experiment.py $MALBASE --base "Allaple.A" $BASE --eps 0 8 --n-poison 100 \
    --seed $s --transforms none uint8 --out out_s9y_n100_s$s > s9y_n100_s$s.log 2>&1
done
echo ALLDONE
