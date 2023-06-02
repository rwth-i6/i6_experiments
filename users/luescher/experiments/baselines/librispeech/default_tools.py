"""
List of default tools and software to be defined as default independent of hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well

"""
from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob

from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_experiments.common.tools.sctk import compile_sctk

RASR_BINARY_PATH = compile_rasr_binaries_i6mode(
    branch="apptainer_tf_2_8", configure_options=["--apptainer-patch=2023-05-08_tensorflow-2.8_v1"]
)
RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_APPTAINER_RASR_BINARY_PATH"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

RETURNN_EXE_PATH = tk.Path("/usr/bin/python3")  # use apptainer system python 3
RETURNN_EXE_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_APPTAINER_PYTHON3_PATH"

RETURNN_ROOT_PATH = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="d7689b945b2fe781b3c79fbef9d82f018c7b11e8", checkout_folder_name="returnn").out_repository
RETURNN_ROOT_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN"

RETURNN_COMMON_PATH = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn_common", commit="04ed372893e81abb3c8f0d4f08e67fbcd4aed510", checkout_folder_name="returnn_common").out_repository
RETURNN_COMMON_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_COMMON"
