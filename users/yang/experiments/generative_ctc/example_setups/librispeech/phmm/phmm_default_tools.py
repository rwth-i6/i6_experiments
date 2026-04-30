"""
Defines the external software to be used for the Experiments
"""
from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob

from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.tools.sctk import compile_sctk

# python from apptainer/singularity/docker
RETURNN_EXE = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

MINI_RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/JackTemaki/MiniReturnn", commit="f9b9be691351f0edd60f2a3d0955c25f15cc2ccb"
).out_repository.copy()
MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

I6_MODELS_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_models",
    commit="1a5fe8302587d24237e2291f05b9c3ab5f01415c",
    checkout_folder_name="i6_models",
).out_repository.copy()
I6_MODELS_REPO_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_MODELS"

I6_NATIVE_OPS_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_native_ops",
    commit="422d579e0ea04b9c12ca0119478f4ea187528528",
    checkout_folder_name="i6_native_ops",
).out_repository.copy()
I6_NATIVE_OPS_REPO_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_NATIVE_OPS"

RECIPE_ROOT = tk.Path(
    "/u/zyang/setups/mini/recipe/i6_experiments",
    hash_overwrite="LIBRISPEECH_LOCAL_RECIPE_ROOT",
).copy()

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12").copy()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"

SUBWORD_NMT_REPO = get_returnn_subword_nmt(
    commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
).copy()
SUBWORD_NMT_REPO.hash_overwrite = "I6_SUBWORD_NMT_V2"

LIBRASR_WHEEL = tk.Path(
    "/u/hwu/repositories/rasr/dist/rasr-0.0.1-cp310-cp310-linux_x86_64.whl",
    hash_overwrite="LIBRISPEECH_PLACEHOLDER_LIBRASR_WHEEL",
).copy()

LM_UTIL_EXE = tk.Path(
    "/u/hwu/repositories/rasr/src/Tools/Lm/lm-util.linux-x86_64-standard",
    hash_overwrite="LIBRISPEECH_LOCAL_RASR_LM_UTIL",
).copy()
