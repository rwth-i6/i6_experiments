#!/bin/bash
# Rebuild the `w2vu` conda env: fairseq 0.12.2 + CUDA torch, for the SAE §1c wav2vec-U 2.0 GAN.
#
# Why a second env at all: the speech_llm env's torch is a CPU-only wheel, and fairseq 0.12.2 pins a
# py<=3.10 stack, so it cannot be installed there. Jobs declaring `requires_env = "w2vu"` get this
# env's LD_LIBRARY_PATH/PYTHONPATH from settings.py::worker_wrapper. NOTE settings.py is not under
# version control -- if it is lost, that gate must be rewritten or every fairseq job dies on
# `undefined symbol: PyObject_CallOneArg` (py3.9 dlopening speech_llm's py3.11 libtorch).
#
# Verified on GH200 (sm_90): real fairseq generator+discriminator forward/backward.
#
#   bash build_w2vu_env.sh [/path/to/conda/envs/w2vu]
set -euo pipefail

PREFIX="${1:-/e/project1/spell/wu24/env/conda/envs/w2vu}"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -p "$PREFIX" python=3.9
conda activate "$PREFIX"

# fairseq 0.12.2's deps list `PyYAML>=5.1.*`, which is not valid PEP 440. Modern pip rejects the
# whole resolve with a misleading "No matching distribution for omegaconf==2.0.6"; pip 23.x still
# tolerates it. Do not "fix" this by relaxing the omegaconf pin -- fairseq 0.12.2 needs exactly 2.0.6
# (hydra-core 1.0.7 is likewise not interchangeable with 1.1+).
python -m pip install -U 'pip==23.3.2'

# aarch64 has no torchaudio==2.6.0+cu126 wheel. It is not needed: the w2v-U path never imports it
# (our features are dumped by a separate speech_llm job), so it is deliberately omitted.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# --ignore-installed is load-bearing: jobs run with PYTHONNOUSERSITE=1, but pip still *sees*
# ~/.local/lib/python3.9/site-packages and silently skips anything already there. Without this,
# typing_extensions/cffi/soundfile resolve on the login node and then ImportError inside the job.
python -m pip install --ignore-installed \
  fairseq==0.12.2 omegaconf==2.0.6 hydra-core==1.0.7 numpy==1.23.5 \
  typing_extensions==4.15.0 cffi==2.0.0 soundfile==0.13.1 bitarray==3.7.2 \
  editdistance==0.8.1 sacrebleu==2.5.1 regex==2026.1.15 Cython==3.1.5 \
  scipy==1.13.1 PyYAML==6.0.3 npy-append-array==0.9.19 kenlm==0.3.0

# Gate: every import must resolve with user-site suppressed, i.e. as the job will see it.
# Checking on the login node without PYTHONNOUSERSITE=1 is NOT representative and will pass falsely.
PYTHONNOUSERSITE=1 python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible -- run this on a GPU node"
assert "sm_90" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()  # GH200
import fairseq, importlib
importlib.import_module("fairseq.examples.wav2vec.unsupervised.models.wav2vec_u")
print("OK", torch.__version__, fairseq.__version__)
EOF
