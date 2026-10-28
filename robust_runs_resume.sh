set -x
cd "$(dirname "$0")"
TF="uint8 jpeg90 jpeg75 jpeg50 resize32"
COMMON="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 25 50"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
venv/bin/python -u run_experiment.py $MAL $COMMON --eps 0 --skip-defense --seed 0 --transforms $TF --fast --out out_robust_malimg_ctrl_s0_fastcheck > robust_malimg_ctrl_s0_fastcheck.log 2>&1
for s in 1 2; do venv/bin/python -u run_experiment.py $MAL $COMMON --eps 0 --skip-defense --seed $s --transforms $TF --fast --out out_robust_malimg_ctrl_s$s > robust_malimg_ctrl_s$s.log 2>&1; done
echo ALLDONE
