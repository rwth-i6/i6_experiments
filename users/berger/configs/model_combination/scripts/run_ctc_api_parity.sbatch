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
exec env -u APPTAINER_BIND -u APPTAINER_BINDPATH -u APPTAINER_MOUNT -u SINGULARITY_BIND -u SINGULARITY_BINDPATH -u SINGULARITY_MOUNT \
  apptainer exec --nv -B /work/asr4 -B /work/common \
  --env PYTHONNOUSERSITE=1 --env PREPEND_LD_LIBRARY_PATH=/.singularity.d/libs \
  --env HF_HOME=/work/asr4/berger/huggingface \
  --env HF_HUB_CACHE=/work/asr4/berger/huggingface/hub \
  --env HF_DATASETS_CACHE=/work/asr4/berger/huggingface/datasets \
  --env HF_XET_CACHE=/work/asr4/berger/huggingface/xet \
  --env HF_ASSETS_CACHE=/work/asr4/berger/huggingface/assets \
  --env HF_TOKEN_PATH=/u/berger/.cache/huggingface/token \
  --env PYTORCH_KERNEL_CACHE_PATH=/var/tmp/torch_kernel_cache_berger \
  /work/asr4/berger/apptainer/images/torch-2.8_onnx-1.26.v3.sif \
  python3 /work/asr4/berger/refactor/asr-20261104/research/i6_experiments/users/berger/configs/model_combination/scripts/run_ctc_api_parity.py \
  --output-dir /work/asr4/berger/refactor/asr-20261104/ctc-api-parity-dev-other
