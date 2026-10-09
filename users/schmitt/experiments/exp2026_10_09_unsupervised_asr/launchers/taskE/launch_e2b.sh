#!/bin/bash
# E.2 stage 2 + the two probes the plan requires (identifiability vs basin).
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts
P=/work/asr4/schmitt/venvs/returnn_torch/bin/python3
O=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE
R () { sbatch -p cpu_modern -c 8 --mem 48G -J "$1" -o "$O/$1.log" \
         --wrap "OMP_NUM_THREADS=8 $P -u calc_e2_unsup_map.py $2"; }

for ov in 1.00 1.25; do
  # the planned chain: order-4 refinement of the loss-selected EM map
  R "E2r_em_$ov"   "refine --overseg $ov --k 512 --init em"
  # identifiability probe: does the criterion's optimum sit near the SUPERVISED map, or walk away?
  R "E2r_ceil_$ov" "refine --overseg $ov --k 512 --init ceiling"
  R "E2em_ceil_$ov" "em    --overseg $ov --k 512 --init ceiling --em-npz '$O/../taskE/e2/none'"
done
# basin probe: the II.B recipe (order-4 from uniform, no EM), on the R~1 arm
for sd in 1 2 3 4; do
  R "E2r_uni_$sd" "refine --overseg 1.00 --k 512 --init uniform --seed $sd"
done
