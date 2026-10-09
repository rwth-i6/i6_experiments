#!/bin/bash
# Task B sweep: 4 effective-weight settings x (oracle start + 4 uniform seeds), one SLURM job each.
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts
OUT=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskB
PY=/work/asr4/schmitt/venvs/returnn_torch/bin/python3
CACHE=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/cheat_seg_cache.pkl
declare -A W=( [cur]="26,21,20,10" [flat]="10,10,10,10" [hi1]="5,5,10,20" [hi2]="2,4,8,16" )
for name in "${!W[@]}"; do
  for run in oracle s1 s2 s3 s4; do
    if [ $run = oracle ]; then init="--init oracle --seed 1"; else init="--init uniform --seed ${run#s}"; fi
    log=$OUT/${name}_${run}.log
    [ -f $log ] && { echo "skip $log"; continue; }
    sbatch -p cpu_modern -c 8 --mem 14G -t 10:00:00 -J tB_${name}_${run} -o $log --wrap \
      "$PY -u calc_soft_map_search.py $init --w ${W[$name]} --four-backoff 0.2 --steps 3000 --log-every 250 --cache $CACHE"
  done
done
