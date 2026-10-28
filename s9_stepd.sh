set -x
cd "$(dirname "$0")"
MAL="--data malimg_all --target VB.AT --base Yuner.A --max-per-class 300 --gray-delta"
COMMON="--size 64 --epochs 5 --objective gradmatch --n-targets 1 --trials 5 --skip-defense --seed 0 --fast --transforms none jpeg75 resize32"
CT="none jpeg90 jpeg75 jpeg50 resize32"
for tag in base ct ctlr32; do
  case $tag in
    base)   V="" ;;
    ct)     V="--craft-transforms $CT" ;;
    ctlr32) V="--craft-transforms $CT --lowres 32" ;;
  esac
  venv/bin/python -u run_experiment.py $MAL $COMMON --eps 16 --n-poison 50  $V --out out_s9_d_eps16_$tag    > s9_d_eps16_$tag.log 2>&1
  venv/bin/python -u run_experiment.py $MAL $COMMON --eps 8  --n-poison 100 $V --out out_s9_d_n100_$tag     > s9_d_n100_$tag.log 2>&1
  venv/bin/python -u run_experiment.py $MAL $COMMON --eps 16 --n-poison 100 $V --out out_s9_d_eps16n100_$tag > s9_d_eps16n100_$tag.log 2>&1
done
echo ALLDONE
