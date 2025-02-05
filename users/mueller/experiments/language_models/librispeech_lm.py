"""
Language model helpers
"""
from sisyphus import tk

from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob

def get_binary_lm(arpa_path: tk.Path) -> tk.Path:
    """
    Returns a manually created LM

    :return: path to a binary LM file
    """
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    
    arpa_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=arpa_path, kenlm_binary_folder=KENLM_BINARY_PATH
    )
    return arpa_binary_lm_job.out_lm
