#!/bin/bash
# Task C, variant 2b (hard / coordinate hill-climb), n <= 2.
set -u
PY=/work/asr4/schmitt/venvs/returnn_torch/bin/python3
CACHE=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/cheat_seg_cache.pkl
OUT=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskC
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts
export OMP_NUM_THREADS=4
run () {  # name, extra args
  $PY -u calc_cheat_seg_identifiability.py --objective kl --restarts 4 --seed 1 \
      --cache $CACHE $2 > $OUT/hard_$1.log 2>&1
  echo "=== $1"; grep -E "oracle map:|best random restart|climb from the oracle|diagonal" $OUT/hard_$1.log
}
run seg      ""
run col_lz0  "--collapse --lam-z 0"
run col_lz10 "--collapse --lam-z 10"
run col_lz100 "--collapse --lam-z 100"
