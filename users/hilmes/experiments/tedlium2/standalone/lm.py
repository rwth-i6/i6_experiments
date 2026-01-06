"""
Language model helpers
"""
from sisyphus import tk

from i6_core.lm.kenlm import CreateBinaryLMJob

from i6_experiments.users.hilmes.common.tedlium2.lm.ngram_config import run_tedlium2_ngram_lm

from .default_tools import KENLM_BINARY_PATH, rasr_binary_path

from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr import CommonRasrParameters, RasrConfig, crp_add_default_output

def get_4gram_binary_lm(prefix_name) -> tk.Path:
    """
    Returns the i6 TEDLIUMv2 LM

    :return: path to a binary LM file
    """
    lm = run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=False).interpolated_lms["dev-pruned"]["4gram"].ngram_lm  # TODO
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=lm, kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_binary_lm_job.add_alias(prefix_name + "/create_4gram_binary_lm")
    return arpa_4gram_binary_lm_job.out_lm


def get_arpa_lm_config(lm_name: str, lexicon_file: tk.Path, scale: float = 1.0) -> RasrConfig:
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    rasr_config.file = {"4gram": run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=False).interpolated_lms["dev-pruned"]["4gram"].ngram_lm}[lm_name]
    rasr_config.scale = scale

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = rasr_binary_path.join_right("lm-util.linux-x86_64-standard")
    crp.language_model_config = rasr_config
    crp.lexicon_config = RasrConfig()
    crp.lexicon_config.file = lexicon_file
    rasr_config.image = CreateLmImageJob(crp, mem=8).out_image

    return rasr_config