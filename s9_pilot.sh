set -x
cd "$(dirname "$0")"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
COMMON="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 50 --eps 8 --skip-defense --seed 0 --fast --transforms none jpeg75 resize32"
CT="none jpeg90 jpeg75 jpeg50 resize32"
venv/bin/python -u run_experiment.py $MAL $COMMON --craft-transforms $CT --out out_s9_pilot_ct > s9_pilot_ct.log 2>&1
venv/bin/python -u run_experiment.py $MAL $COMMON --lowres 16 --out out_s9_pilot_lr16 > s9_pilot_lr16.log 2>&1
venv/bin/python -u run_experiment.py $MAL $COMMON --craft-transforms $CT --lowres 16 --out out_s9_pilot_ct_lr16 > s9_pilot_ct_lr16.log 2>&1
venv/bin/python -u run_experiment.py $MAL $COMMON --craft-transforms $CT --lowres 32 --out out_s9_pilot_ct_lr32 > s9_pilot_ct_lr32.log 2>&1
echo ALLDONE
