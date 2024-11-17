"""
Language model helpers
"""
from sisyphus import tk

from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob

from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"


def get_4gram_binary_lm() -> tk.Path:
    """
    Returns the official LibriSpeech 4-gram ARPA LM

    :return: path to a binary LM file
    """
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=get_arpa_lm_dict()["4gram"], kenlm_binary_folder=KENLM_BINARY_PATH
    )
    return arpa_4gram_binary_lm_job.out_lm
