set -x
cd "$(dirname "$0")"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
BASE="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --seed 0 --fast"
CT="none jpeg90 jpeg75 jpeg50 resize32"
ALLTF="none uint8 jpeg90 jpeg75 jpeg50 resize32 jpeg60 resize48 blur0.5"
# G: detection check on the compression-aware poisons (defenses on, strongest cell)
venv/bin/python -u run_experiment.py $MAL $BASE --eps 16 --n-poison 50 --transforms none jpeg75 --craft-transforms $CT --out out_s9_g_ct > s9_g_ct.log 2>&1
venv/bin/python -u run_experiment.py $MAL $BASE --eps 16 --n-poison 50 --transforms none jpeg75 --craft-transforms $CT --lowres 32 --out out_s9_g_ctlr32 > s9_g_ctlr32.log 2>&1
# C: held-out transforms on the only cell that showed any jpeg75 survival, plus eps 0 control
venv/bin/python -u run_experiment.py $MAL $BASE --eps 0 16 --n-poison 100 --skip-defense --transforms $ALLTF --out out_s9_c_base > s9_c_base.log 2>&1
venv/bin/python -u run_experiment.py $MAL $BASE --eps 0 16 --n-poison 100 --skip-defense --transforms $ALLTF --craft-transforms $CT --out out_s9_c_ct > s9_c_ct.log 2>&1
echo ALLDONE
