#!/bin/bash
# E.3: supervised generator references (labels) + label-free runs, both arms. GPU (gpu_11gb).
#   ./launch_e3.sh ceiling      -> ceiling_k{1,4}.pt per arm (k1 must reproduce E.1: 41.67 / 32.27)
#   ./launch_e3.sh smoke        -> 300-step uniform run per arm (timing + sanity)
#   ./launch_e3.sh train        -> 4 uniform seeds + the ceiling-start probe per arm
cd /u/schmitt/experiments/2026_04_09_unsupervised_asr/recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/scripts
P=/work/asr4/schmitt/venvs/torch-2.11/bin/python3
O=/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE
S="sbatch -p gpu_11gb --gres=gpu:1 -c 4 --mem 24G"
arm_flags() { [ "$1" = 1.25 ] && echo "--no-repeat" || echo ""; }
case "$1" in
ceiling)
  for ov in 1.00 1.25; do for k in 1 4; do
    $S -J "E3c${ov}k$k" -o "$O/e3_ceiling_${ov}_k$k.log" \
      --wrap "OMP_NUM_THREADS=4 $P -u calc_e3_generator.py ceiling --overseg $ov --kernel $k"
  done; done ;;
smoke)
  for ov in 1.00 1.25; do
    $S -J "E3s$ov" -o "$O/e3_smoke_$ov.log" \
      --wrap "OMP_NUM_THREADS=4 $P -u calc_e3_generator.py train --overseg $ov --kernel 1 $(arm_flags $ov) \
              --steps 300 --eval-every 100 --log-every 25 --score-every-eval --out $O/e3/smoke_$ov.pt"
  done ;;
train)
  for ov in 1.00 1.25; do
    for sd in 1 2 3 4; do
      $S -J "E3u${ov}_$sd" -o "$O/e3_train_${ov}_uniform_s$sd.log" \
        --wrap "OMP_NUM_THREADS=4 $P -u calc_e3_generator.py train --overseg $ov --kernel 1 $(arm_flags $ov) \
                --init uniform --seed $sd --score-every-eval"
    done
    $S -J "E3p$ov" -o "$O/e3_train_${ov}_ceiling_s1.log" \
      --wrap "OMP_NUM_THREADS=4 $P -u calc_e3_generator.py train --overseg $ov --kernel 1 $(arm_flags $ov) \
              --init ceiling --seed 1 --score-every-eval"
  done ;;
esac
# added after the first probe: x1.00 with --no-repeat (the plain criterion charges the collapse
# decode's repeats; the supervised generator scored 33.50 there vs ~22 for cold runs)
# basin sweep (2026-09-29): x1.25 no-repeat, start = supervised generator damaged
#   bash launch_e3.sh basin
if [ "$1" = basin ]; then
  run() { $S -J "E3b$1$2_$3" -o "$O/e3_basin_$1$2_s$3.log" \
    --wrap "OMP_NUM_THREADS=4 $P -u calc_e3_generator.py train --overseg 1.25 --kernel 1 --no-repeat \
            --init corrupt --corrupt-kind $1 --corrupt-frac $2 --seed $3 --score-every-eval"; }
  for f in 0.2 0.4 0.6 0.8 1.0; do run perm $f 1; done
  run perm 0.6 2; run perm 1.0 2
  for f in 0.5 0.8 0.95; do run mix $f 1; done
fi
