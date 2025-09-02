from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr import CommonRasrParameters, RasrConfig, crp_add_default_output
from i6_experiments.common.baselines.tedlium2.data import get_corpus_data_inputs
from sisyphus import tk

from ...tools import ken_lm_binaries, rasr_binary_path


def get_binary_lm(lm_name: str) -> tk.Path:
    lm_dict = {"4gram": get_corpus_data_inputs()["dev"]["dev"].lm["filename"]}
    arpa_lm = lm_dict[lm_name]
    assert arpa_lm is not None
    return CreateBinaryLMJob(arpa_lm=arpa_lm, kenlm_binary_folder=ken_lm_binaries).out_lm


def get_arpa_lm_config(lm_name: str, lexicon_file: tk.Path, scale: float = 1.0) -> RasrConfig:
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    rasr_config.file = {"4gram": get_corpus_data_inputs()["dev"]["dev"].lm["filename"]}[lm_name]
    rasr_config.scale = scale

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = rasr_binary_path.join_right("lm-util.linux-x86_64-standard")
    crp.language_model_config = rasr_config
    crp.lexicon_config = RasrConfig()
    crp.lexicon_config.file = lexicon_file
    rasr_config.image = CreateLmImageJob(crp, mem=8).out_image

    return rasr_config
