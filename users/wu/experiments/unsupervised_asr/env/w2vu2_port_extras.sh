#!/usr/bin/env bash
# Add the two packages the wav2vec-U 2.0 reproduction (config/w2vu2.py) needs in the port's MAIN env
# (env/environment.yml, created by install_env.sh), not in the fairseq env of build_w2vu_env.sh:
#   torchaudio    data/w2vu2_features.py, the MFCC k-means and the feature-data job
#                 (torchaudio.compliance.kaldi.mfcc, torchaudio.functional.compute_deltas).
#                 environment.yml leaves it out.  Installed from pip with --no-deps at the version of
#                 the env's torch (2.7.1 from environment.yml), so pip cannot replace the conda-forge
#                 torch; this is how the reference env has it (JUPITER speech_llm: torchaudio 2.7.1,
#                 a pip wheel on top of conda-forge pytorch 2.7.1).
#   scikit-learn  data/w2vu2_features.py, MiniBatchKMeans.  1.8.0, the reference env's version
#                 (implementer report A, 2026-09-25); environment.yml already pins it, so this only
#                 installs it (conda-forge) when the env holds another version or none.
#
# Usage: w2vu2_port_extras.sh <main_env_prefix>
#   CONDA_BIN   conda or mamba executable, used only if scikit-learn must be installed.
#               Default: mamba, else conda, from PATH.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <main_env_prefix>" >&2
    exit 2
fi
PREFIX="$1"
PY="$PREFIX/bin/python"
SKLEARN_VERSION="1.8.0"
die() { echo "w2vu2_port_extras.sh: $*" >&2; exit 1; }
[[ -x "$PY" ]] || die "$PY is not an executable python"

# the torch version without its local tag (2.7.1+cu126 -> 2.7.1); torchaudio is released in lockstep
TORCH_VERSION="$(PYTHONNOUSERSITE=1 "$PY" -c 'import torch; print(torch.__version__.split("+")[0])')"

# ---- 1. torchaudio ------------------------------------------------------------------------------
PYTHONNOUSERSITE=1 "$PY" -m pip install --no-deps "torchaudio==$TORCH_VERSION"

# ---- 2. scikit-learn ----------------------------------------------------------------------------
HAVE_SKLEARN="$(PYTHONNOUSERSITE=1 "$PY" -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null || true)"
if [[ "$HAVE_SKLEARN" != "$SKLEARN_VERSION" ]]; then
    CONDA_BIN="${CONDA_BIN:-$(command -v mamba || command -v conda || true)}"
    [[ -n "$CONDA_BIN" ]] || die "scikit-learn is '${HAVE_SKLEARN:-absent}'; need conda/mamba (CONDA_BIN) to install $SKLEARN_VERSION"
    "$CONDA_BIN" install -y -p "$PREFIX" -c conda-forge --override-channels "scikit-learn=$SKLEARN_VERSION"
fi

# ---- 3. gate: the versions, torch untouched, and the calls the MFCC k-means makes ----------------
TORCH_VERSION="$TORCH_VERSION" SKLEARN_VERSION="$SKLEARN_VERSION" PYTHONNOUSERSITE=1 "$PY" - <<'EOF'
import os

import torch
import torchaudio
from torchaudio.compliance import kaldi
import sklearn
from sklearn.cluster import MiniBatchKMeans

assert torch.__version__.split("+")[0] == os.environ["TORCH_VERSION"], ("torch changed", torch.__version__)
assert torchaudio.__version__.split("+")[0] == os.environ["TORCH_VERSION"], torchaudio.__version__
assert sklearn.__version__ == os.environ["SKLEARN_VERSION"], sklearn.__version__

wav = torch.randn(1, 16000)
m = kaldi.mfcc(wav, sample_frequency=16000, use_energy=False)
d = torchaudio.functional.compute_deltas(m.T[None])[0].T
assert m.shape == d.shape and m.shape[1] == 13, (m.shape, d.shape)
MiniBatchKMeans(n_clusters=2, n_init=1, random_state=0).fit(m.numpy())
print("OK torch", torch.__version__, "torchaudio", torchaudio.__version__, "sklearn", sklearn.__version__)
EOF
