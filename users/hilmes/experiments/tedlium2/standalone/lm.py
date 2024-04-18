"""
Language model helpers
"""
from sisyphus import tk

from i6_core.lm.kenlm import CreateBinaryLMJob

from i6_experiments.users.hilmes.common.tedlium2.lm.ngram_config import run_tedlium2_ngram_lm

from .default_tools import KENLM_BINARY_PATH


def get_4gram_binary_lm(prefix_name) -> tk.Path:
    """
    Returns the i6 TEDLIUMv2 LM

    :return: path to a binary LM file
    """
    lm = run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=False).interpolated_lms["dev-pruned"]["4gram"]  # TODO
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=lm, kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_binary_lm_job.add_alias(prefix_name + "/create_4gram_binary_lm")
    return arpa_4gram_binary_lm_job.out_lm
