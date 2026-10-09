#!/bin/bash
# Task E.1 -- the supervised-ceiling ladder. CPU only; /var/tmp is node-local, so logs go here.
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts
PY=/work/asr4/schmitt/venvs/returnn_torch/bin/python3
OUT=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE

sub () {  # name, args...
  local name=$1; shift
  sbatch -p cpu_modern -c 16 --mem 96G -J "E1_$name" -o "$OUT/$name.log" \
    --wrap "OMP_NUM_THREADS=16 $PY -u calc_emission_ladder.py $*"
}

# one job per feature stage (they are independent and the big one needs the memory to itself),
# plus the stage-independent clus128 control in the first.
sub pooled   --rungs a,b,c,d,e,f --stages pooled   --ks 128,256,512,1024,2048 --oversegment 1.0,0.8
sub cls_mean --rungs b,c,d,e,f   --stages cls_mean --ks 128,256,512,1024,2048 --oversegment 1.25,1.0
sub pca512   --rungs b,c,d,f     --stages pca512   --ks 128,512,2048          --oversegment 1.25,1.0
