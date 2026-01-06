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
    "https://github.com/JackTemaki/MiniReturnn", commit="0dc69329b21ce0acade4fcb2bf1be0dc8cc0b121"
).out_repository.copy()
MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

I6_MODELS_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_models",
    commit="5aa74f878cc0d8d7bbc623a3ced681dcb31955ec",
    checkout_folder_name="i6_models",
).out_repository.copy()
I6_MODELS_REPO_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_MODELS"

I6_NATIVE_OPS_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_native_ops",
    commit="9ea83b59d23d631fb0388c76164fece2e5ae7fb3",
    checkout_folder_name="i6_native_ops",
).out_repository.copy()
I6_NATIVE_OPS_REPO_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_NATIVE_OPS"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12").copy()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"

SUBWORD_NMT_REPO = get_returnn_subword_nmt(
    commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
).copy()
SUBWORD_NMT_REPO.hash_overwrite = "I6_SUBWORD_NMT_V2"

# TORCH_MEMRISTOR_PATH = CloneGitRepositoryJob(
#     url="https://github.com/rwth-i6/SynaptogenML",
#     commit="7f5320d9331d4f27a0a7a5a58c2b697e608f0272",
#     checkout_folder_name="SynaptogenML",
# ).out_repository.copy()
TORCH_MEMRISTOR_PATH = CloneGitRepositoryJob(
    url="git@git.rwth-aachen.de:mlhlt/torch-memristor.git",
    commit="88af8c663fa8ce55ac3b559581081653da3e1610",
    checkout_folder_name="torch_memristor",
    branch="bene_cycle",
).out_repository.copy()
TORCH_MEMRISTOR_PATH.hash_overwrite = "LIBRISPEECH_STANDALONE_DEFAULT_TORCH_MEMRISTOR"

rasr_path = "/work/asr4/hilmes/dev/rasr_librasr_19_09_25/"
rasr_root = tk.Path(rasr_path, hash_overwrite="LIBRISPEECH_STANDALONE_DEFAULT_RASR_ROOT").copy()
rasr_binary_path = tk.Path(
    f"{rasr_path}arch/linux-x86_64-standard",
    hash_overwrite="RASR_BINARY_PATH",
).copy()

I6_CORE_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_core",
    commit="fd5b07bd902e4027d5ca9b15e9d88687699ea274",
    checkout_folder_name="i6_core",
).out_repository.copy()
I6_CORE_REPO_PATH.hash_overwrite = "LIBRISPEECH_STANDALONE_DEFAULT_I6_CORE"
