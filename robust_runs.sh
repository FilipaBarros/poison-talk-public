set -x
cd "$(dirname "$0")"
TF="uint8 jpeg90 jpeg75 jpeg50 resize32"
COMMON="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --n-poison 25 50"
ISIC="--data isic_folder --target malignant --base benign --max-per-class 1500"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
venv/bin/python run_experiment.py $ISIC $COMMON --eps 8 --seed 0 --transforms $TF --out out_robust_isic_s0 > robust_isic_s0.log 2>&1
venv/bin/python run_experiment.py $ISIC $COMMON --eps 0 --skip-defense --seed 0 --transforms $TF --out out_robust_isic_ctrl_s0 > robust_isic_ctrl_s0.log 2>&1
for s in 1 2; do venv/bin/python run_experiment.py $ISIC $COMMON --eps 0 8 --skip-defense --seed $s --transforms $TF --out out_robust_isic_s$s > robust_isic_s$s.log 2>&1; done
for s in 0 1 2; do venv/bin/python run_experiment.py $MAL $COMMON --eps 8 --skip-defense --seed $s --transforms $TF --out out_robust_malimg_s$s > robust_malimg_s$s.log 2>&1; done
for s in 0 1 2; do venv/bin/python run_experiment.py $MAL $COMMON --eps 0 --skip-defense --seed $s --transforms $TF --out out_robust_malimg_ctrl_s$s > robust_malimg_ctrl_s$s.log 2>&1; done
echo ALLDONE
