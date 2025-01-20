from i6_experiments.common.baselines.tedlium2.data import get_corpus_data_inputs
from i6_core.lm.kenlm import CreateBinaryLMJob
from sisyphus import tk

from ...tools import ken_lm_binaries


def get_binary_lm(lm_name: str) -> tk.Path:
    lm_dict = {"4gram": get_corpus_data_inputs()["dev"]["dev"].lm["filename"]}
    arpa_lm = lm_dict[lm_name]
    assert arpa_lm is not None
    return CreateBinaryLMJob(arpa_lm=arpa_lm, kenlm_binary_folder=ken_lm_binaries).out_lm
