from i6_core.lm.kenlm import CreateBinaryLMJob

from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

from .default_tools import KENLM_BINARY_PATH


def get_4gram_binary_lm():
    """

    :param output_prefix:
    :return:
    """
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=get_arpa_lm_dict()["4gram"], kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_binary_lm_job.add_alias("experiments/librispeech/standalone_2023/lm/create_4gram_binary_lm")
    return arpa_4gram_binary_lm_job.out_lm
