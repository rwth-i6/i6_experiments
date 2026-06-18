"""
External tools / paths for the SSL experiments on the JUPITER cluster.

The cluster uses a conda env (not apptainer); RETURNN + i6_models run from the in-workspace
checkouts. Paths are absolute with stable ``hash_overwrite`` so they do not leak into job hashes.
"""

from sisyphus import tk

_WORKSPACE = "/e/project1/spell/wu24/2026-06-17_ssl"

# conda interpreter that has torch 2.7.1 (GH200/aarch64), librosa, datasets, ...
RETURNN_EXE = tk.Path(
    "/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)

# in-workspace RETURNN checkout (the `returnn` symlink -> recipe/returnn)
RETURNN_ROOT = tk.Path(f"{_WORKSPACE}/recipe/returnn", hash_overwrite="SSL_RETURNN_ROOT")

# in-workspace i6_models (reviewed as clean); added to sys.path by the serializer's ExternalImport
I6_MODELS_REPO_PATH = tk.Path(f"{_WORKSPACE}/recipe/i6_models", hash_overwrite="SSL_I6_MODELS")

# recipe root, added to sys.path for forward/recog jobs (decoder imports i6_core etc.)
RECIPE_ROOT = tk.Path(f"{_WORKSPACE}/recipe", hash_overwrite="SSL_RECIPE_ROOT")

# SCTK bin dir (sclite) from the conda env -> offline WER scoring (no compile_sctk / internet).
SCTK_BINARY_PATH = tk.Path(
    "/e/project1/spell/wu24/env/conda/envs/speech_llm/bin", hash_overwrite="SSL_SCTK_BINARY_PATH"
)
