from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob, CreateBinaryLMJob


def arpa_to_kenlm_bin(arpa_lm_file: tk.Path) -> tk.Path:
    ken_lm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm.git").out_repository
    ken_lm_binaries = CompileKenLMJob(repository=ken_lm_repo).out_binaries

    return CreateBinaryLMJob(arpa_lm=arpa_lm_file, kenlm_binary_folder=ken_lm_binaries).out_lm
