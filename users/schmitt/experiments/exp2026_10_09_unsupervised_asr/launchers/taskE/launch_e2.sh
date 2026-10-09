#!/bin/bash
# E.2 stage 1: supervised reference + HMM-EM over 4 uniform seeds, for both segmentation arms.
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts
P=/work/asr4/schmitt/venvs/returnn_torch/bin/python3
O=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE
for ov in 1.00 1.25; do
  sbatch -p cpu_modern -c 8 --mem 32G -J "E2c$ov" -o "$O/e2_ceiling_$ov.log" \
    --wrap "OMP_NUM_THREADS=8 $P -u calc_e2_unsup_map.py ceiling --overseg $ov --k 512"
  for sd in 1 2 3 4; do
    sbatch -p cpu_modern -c 8 --mem 32G -J "E2em${ov}_$sd" -o "$O/e2_em_${ov}_s$sd.log" \
      --wrap "OMP_NUM_THREADS=8 $P -u calc_e2_unsup_map.py em --overseg $ov --k 512 --seed $sd"
  done
done
