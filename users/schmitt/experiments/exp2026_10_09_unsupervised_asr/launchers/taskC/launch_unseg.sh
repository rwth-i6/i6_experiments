#!/bin/bash
# Task C on the REAL unsegmented data (clus128, R = 2.63). One job per lam_z; the supervised
# ceiling (expensive, lam_z-independent) is computed only in the lam_z=10 job.
set -u
PY=/work/asr4/schmitt/venvs/returnn_torch/bin/python3
OUT=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskC
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts
COMMON="--clus128-shards 4 --num-audio-utts 20000 --num-text-utts 60000 --num-eval-utts 1000 \
        --num-ceiling-utts 3000 --seeds 4 --steps 3000 --log-every 1000"
for lz in 0 1 10 100; do
  if [ "$lz" = 10 ]; then extra="--em-iters 40 --per-cd-sweeps 2"; else extra="--em-iters 8 --per-cd-sweeps 0"; fi
  sbatch -p cpu_modern -c 8 --mem 24G -t 10:00:00 -J tC_lz$lz -o $OUT/unseg_lz${lz}.log \
    --wrap "$PY -u calc_unsegmented_map.py $COMMON --lam-z $lz $extra"
done
# ablation: no collapse at all on the same data (B removed -> the criterion sees the frame-level
# statistics directly, which the 2.6x length mismatch should wreck)
sbatch -p cpu_modern -c 8 --mem 24G -t 10:00:00 -J tC_nocol -o $OUT/unseg_nocollapse.log \
  --wrap "$PY -u calc_unsegmented_map.py $COMMON --no-collapse --lam-z 0 --em-iters 8 --per-cd-sweeps 0"
