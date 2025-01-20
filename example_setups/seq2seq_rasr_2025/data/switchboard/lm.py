from i6_core.lm.kenlm import CreateBinaryLMJob
from sisyphus import tk

from ...tools import ken_lm_binaries


def get_binary_lm(lm_name: str) -> tk.Path:
    arpa_lm = {"4gram": tk.Path("/work/asr4/berger/dependencies/switchboard/lm/zoltan_4gram.gz")}[lm_name]
    return CreateBinaryLMJob(arpa_lm=arpa_lm, kenlm_binary_folder=ken_lm_binaries).out_lm
