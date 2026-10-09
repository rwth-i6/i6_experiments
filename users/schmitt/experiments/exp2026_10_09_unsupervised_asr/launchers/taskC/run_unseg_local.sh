#!/bin/bash
set -u
PY=/work/asr4/schmitt/venvs/returnn_torch/bin/python3
OUT=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskC
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts
COMMON="--clus128-shards 4 --num-audio-utts 20000 --num-text-utts 60000 --num-eval-utts 1000 \
        --num-ceiling-utts 3000 --seeds 4 --steps 3000 --log-every 1500"
export OMP_NUM_THREADS=3
for lz in 0 1 100; do
  $PY -u calc_unsegmented_map.py $COMMON --lam-z $lz --em-iters 8 --per-cd-sweeps 0 \
      > $OUT/unseg_lz${lz}.log 2>&1 &
done
$PY -u calc_unsegmented_map.py $COMMON --no-collapse --lam-z 0 --em-iters 8 --per-cd-sweeps 0 \
    > $OUT/unseg_nocollapse.log 2>&1 &
$PY -u calc_unsegmented_map.py $COMMON --lam-z 10 --em-iters 40 --per-cd-sweeps 2 \
    > $OUT/unseg_lz10.log 2>&1 &
wait
echo ALL DONE
