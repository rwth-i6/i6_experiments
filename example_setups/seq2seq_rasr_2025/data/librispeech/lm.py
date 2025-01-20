from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from sisyphus import tk

from ...tools import ken_lm_binaries


def get_binary_lm(lm_name: str) -> tk.Path:
    arpa_lm = get_arpa_lm_dict()[lm_name]
    return CreateBinaryLMJob(arpa_lm=arpa_lm, kenlm_binary_folder=ken_lm_binaries).out_lm
