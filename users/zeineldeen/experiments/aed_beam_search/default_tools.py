from sisyphus import tk
from i6_experiments.common.tools.sctk import compile_sctk


PYTHON_EXE = tk.Path(
    "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2024-02-12--aed-beam-search/espnet_venv/bin/python3",
    hash_overwrite="espnet_env_python",
)

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "SWITCHBOARD_DEFAULT_SCTK_BINARY_PATH"
