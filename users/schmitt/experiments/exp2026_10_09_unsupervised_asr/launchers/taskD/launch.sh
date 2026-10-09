#!/bin/bash
# Task D: frozen-transition HMM on the cheat-seg data.
set -u
PY=/work/asr4/schmitt/venvs/returnn_torch/bin/python3
CACHE=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/cheat_seg_cache.pkl
OUT=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskD
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts
C="--num-text-utts 3000 --cache $CACHE"
sub () { sbatch -p cpu_modern -c 8 --mem 16G -t $1 -J $2 -o $OUT/$2.log --wrap "$PY -u $3"; }

# required: both search modes, oracle start + 4 uniform cold seeds, hardened-objective selection
sub 4:00:00  D_em   "calc_hmm_map_search.py --mode em   $C --init uniform --seeds 4 --steps 80  --log-every 10 --save-npz $OUT/em_best.npz"
sub 12:00:00 D_grad "calc_hmm_map_search.py --mode grad $C --init uniform --seeds 4 --steps 300 --log-every 25 --save-npz $OUT/grad_best.npz"
# basin: how far from the truth can EM start and still recover? (the n-gram arm's --basin sweep)
for c in 0.2 0.4 0.6 0.8; do
  sub 3:00:00 D_basin$c "calc_hmm_map_search.py --mode em $C --init oracle-corrupt --corrupt $c --seeds 2 --steps 80 --log-every 20"
done
