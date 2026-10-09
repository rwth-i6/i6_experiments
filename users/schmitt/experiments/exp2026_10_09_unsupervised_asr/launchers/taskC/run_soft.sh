#!/bin/bash
# Task C, variant 3 (soft / Adam), n <= 2: collapse-repeats pushforward on the cheat-seg data.
# One arm per (setting, init); ~4 s per run, so this runs locally.
set -u
PY=/work/asr4/schmitt/venvs/returnn_torch/bin/python3
CACHE=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/cheat_seg_cache.pkl
OUT=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskC
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts

declare -A ARM=(
  [seg]=""                                          # control: no collapse (the segmented n<=2 criterion)
  [col_lz0]="--collapse --lam-z 0"
  [col_lz1]="--collapse --lam-z 1"
  [col_lz10]="--collapse --lam-z 10"
  [col_lz100]="--collapse --lam-z 100"
  [col_lz1000]="--collapse --lam-z 1000"
  [col_lz100_zt965]="--collapse --lam-z 100 --z-target 0.9652"   # oracle-informed Z, sensitivity only
)
export OMP_NUM_THREADS=4
for name in "${!ARM[@]}"; do
  for run in oracle s1 s2 s3 s4; do
    case $run in
      oracle) init="--init oracle --seed 1" ;;
      s*)     init="--init uniform --seed ${run#s}" ;;
    esac
    log=$OUT/${name}_${run}.log
    $PY -u calc_soft_map_search.py ${ARM[$name]} $init --steps 3000 --log-every 1000 \
        --cache $CACHE > $log 2>&1
    echo "$name $run  $(grep -o 'hard [0-9.]* (.*) | acc [0-9.]*' $log | head -1)"
  done
done
