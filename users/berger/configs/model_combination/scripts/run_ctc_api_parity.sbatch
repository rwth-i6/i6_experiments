#!/bin/bash
#SBATCH --job-name=ctc-api-parity
#SBATCH --partition=gpu_48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/u/berger/asr-exps/librispeech/20241106_seq2seq_template_setups
#SBATCH --output=/work/asr4/berger/refactor/asr-20261104/ctc-api-parity-%j.log
set -euo pipefail
exec env -u APPTAINER_BIND -u APPTAINER_BINDPATH -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
  apptainer exec --nv -B /work/asr4 -B /work/common \
  --env PYTHONNOUSERSITE=1 --env PREPEND_LD_LIBRARY_PATH=/.singularity.d/libs \
  /work/asr4/berger/apptainer/images/torch-2.8_onnx-1.26.v3.sif \
  python3 /work/asr4/berger/refactor/asr-20261104/research/i6_experiments/users/berger/configs/model_combination/scripts/run_ctc_api_parity.py \
  --output-dir /work/asr4/berger/refactor/asr-20261104/ctc-api-parity-dev-other
