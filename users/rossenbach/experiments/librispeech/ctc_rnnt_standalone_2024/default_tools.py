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
    # commit="918143c1011fe5a19c5fcfb61fe05050a8d58a2b",
    # commit="5aa74f878cc0d8d7bbc623a3ced681dcb31955ec",
    commit="bb8fa4e690117bae6ab7694b908ee3366376c54b", # with ATT decoder branch
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

NISQA_REPO = CloneGitRepositoryJob("https://github.com/gabrielmittag/NISQA").out_repository.copy()
NISQA_REPO.hash_overwrite = "LIBRISPEECH_DEFAULT_NISQA_REPO"

TORCH_MEMRISTOR_PATH = CloneGitRepositoryJob(
    url="git@git.rwth-aachen.de:mlhlt/torch-memristor.git",
    commit="b064e60120d96e7a542ac858591aae2ba600b797",
    checkout_folder_name="torch_memristor",
).out_repository.copy()
TORCH_MEMRISTOR_PATH.hash_overwrite = "LIBRISPEECH_STANDALONE_DEFAULT_TORCH_MEMRISTOR_V2"

