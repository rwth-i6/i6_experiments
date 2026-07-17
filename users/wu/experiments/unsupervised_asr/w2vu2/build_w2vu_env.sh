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
# numpy is deliberately NOT in this list: --ignore-installed overwrites without uninstalling, and for
# numpy that leaves a Frankenstein (numpy-1.23.5.dist-info AND numpy-2.0.2.dist-info side by side,
# pip reporting 1.23.5 while `import numpy` says 2.0.2). It is pinned separately below.
python -m pip install --ignore-installed \
  fairseq==0.12.2 omegaconf==2.0.6 hydra-core==1.0.7 \
  typing_extensions==4.15.0 cffi==2.0.0 soundfile==0.13.1 bitarray==3.7.2 \
  editdistance==0.8.1 sacrebleu==2.5.1 regex==2026.1.15 Cython==3.1.5 \
  scipy==1.13.1 PyYAML==6.0.3 npy-append-array==0.9.19 kenlm==0.3.0

# fairseq 0.12.2 needs numpy<1.24 -- fairseq/data/data_utils.py:488 still uses `np.int`.
# Uninstall in a loop: a previous --ignore-installed run can leave several dist-infos stacked, and
# each `pip uninstall` removes only one.
while python -m pip uninstall -y numpy >/dev/null 2>&1; do :; done
rm -rf "$PREFIX"/lib/python3.*/site-packages/numpy "$PREFIX"/lib/python3.*/site-packages/numpy-*.dist-info \
       "$PREFIX"/lib/python3.*/site-packages/numpy.libs
python -m pip install --no-cache-dir "numpy==1.23.5"

# Rebuild fairseq's two Cython extensions against *that* numpy. Neither pip mode can do this itself:
#   with build isolation    setup.py's `from torch.utils import cpp_extension` raises ImportError in
#                           the isolated env, so the libbase extension is skipped and the wheel builds
#                           -- but the isolated env holds the newest numpy, so the .so targets the
#                           numpy-2 ABI and every run dies in batch_by_size with
#                           "numpy.dtype size changed ... Expected 96 from C header, got 88".
#   without build isolation torch is importable, so setup.py adds libbase, which needs
#                           fairseq/clib/libbase/balanced_assignment.cpp -- a file the 0.12.2 sdist
#                           does not ship. The build cannot succeed at all.
# So build exactly the two extensions w2v-U needs, and drop them into the installed package.
SP="$(PYTHONNOUSERSITE=1 python -c 'import fairseq, os; print(os.path.dirname(fairseq.__file__))')"
BUILD="$(mktemp -d)"
( cd "$BUILD"
  python -m pip download fairseq==0.12.2 --no-deps --no-binary :all: --no-cache-dir -d . >/dev/null
  tar xf fairseq-0.12.2.tar.gz
  cd fairseq-0.12.2
  cat > build_ext_only.py <<'PY'
from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize
exts = [Extension(f"fairseq.data.{n}", sources=[f"fairseq/data/{n}.pyx"],
                  include_dirs=[numpy.get_include()], language="c++")
        for n in ("data_utils_fast", "token_block_utils_fast")]
setup(name="fairseq_ext_only", ext_modules=cythonize(exts, language_level="3"))
PY
  PYTHONNOUSERSITE=1 python build_ext_only.py build_ext --inplace
  cp fairseq/data/*.so "$SP/data/"
)
rm -rf "$BUILD"

# The unsupervised task imports `from examples.speech_recognition...`, so `examples` must be a
# top-level package. Upstream's repo root has fairseq/ and examples/ as siblings; the wheel nests
# examples *inside* fairseq/, next to logging/, data/ and tasks/ -- so putting site-packages/fairseq
# on PYTHONPATH would shadow the **stdlib** `logging` for every process that inherits it. This shim
# dir exposes `examples` and nothing else. settings.py::W2VU_SHIM_DIR points here.
SP="$(PYTHONNOUSERSITE=1 python -c 'import fairseq, os; print(os.path.dirname(fairseq.__file__))')"
mkdir -p "$PREFIX/fairseq_shim"
ln -sfn "$SP/examples" "$PREFIX/fairseq_shim/examples"

# Gate: every import must resolve with user-site suppressed, i.e. as the job will see it.
# Checking on the login node without PYTHONNOUSERSITE=1 is NOT representative and will pass falsely.
PYTHONNOUSERSITE=1 PYTHONPATH="$PREFIX/fairseq_shim" python - <<'EOF'
import logging
assert hasattr(logging, "getLogger"), f"stdlib logging is shadowed: {logging.__file__}"

import numpy as np
assert np.__version__ == "1.23.5", f"numpy is {np.__version__}; fairseq 0.12.2 needs <1.24 (np.int)"

import torch
assert "sm_90" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()  # GH200

import fairseq
from examples.speech_recognition.kaldi.kaldi_decoder import KaldiDecoderConfig  # noqa: F401

# The numpy-ABI check: importing the extension and actually calling it are different failures.
from fairseq.data import data_utils
b = data_utils.batch_by_size(np.arange(20), num_tokens_fn=lambda i: 10, max_tokens=50, max_sentences=4)
assert len(b) == 5, b
print("OK", torch.__version__, fairseq.__version__, np.__version__, "| cuda:", torch.cuda.is_available())
EOF
