"""
Language model helpers
"""
from sisyphus import tk

from i6_core.lm.kenlm import CreateBinaryLMJob

from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

from .default_tools import KENLM_BINARY_PATH


def get_4gram_binary_lm() -> tk.Path:
    """
    Returns the official LibriSpeech 4-gram ARPA LM

    :return: path to a binary LM file
    """
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=get_arpa_lm_dict()["4gram"],
        kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_binary_lm_job.add_alias("experiments/jaist_project/standalone_2024/lm/create_4gram_binary_lm")
    return arpa_4gram_binary_lm_job.out_lm
