"""
Language model helpers
"""
from sisyphus import tk

from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr import CommonRasrParameters, RasrConfig, crp_add_default_output

from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

from .default_tools import KENLM_BINARY_PATH, LM_UTIL_EXE


def get_4gram_binary_lm(prefix_name) -> tk.Path:
    """
    Returns the official LibriSpeech 4-gram ARPA LM

    :return: path to a binary LM file
    """
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=get_arpa_lm_dict()["4gram"], kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_binary_lm_job.add_alias(prefix_name + "/create_4gram_binary_lm")
    return arpa_4gram_binary_lm_job.out_lm


def get_4gram_lm_rasr_config(lexicon_file: tk.Path, scale: float = 1.0) -> RasrConfig:
    """
    Returns a standard RASR LM config for the official LibriSpeech 4-gram LM.

    :param lexicon_file: lexicon used to build the LM image
    :param scale: LM scale used by lexical search
    :return: RASR LM config
    """
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    rasr_config.file = get_arpa_lm_dict()["4gram"]
    rasr_config.scale = scale

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = LM_UTIL_EXE
    crp.language_model_config = rasr_config
    crp.lexicon_config = RasrConfig()
    crp.lexicon_config.file = lexicon_file
    crp.lexicon_config.normalize_pronunciation = False
    # LibRASR loads the LM in-process and does not expand Sisyphus cached-path markers like `cf ...`,
    # so the image path must render as a plain filesystem path. `out_image` is a cached Path
    # (renders as `cf ...`), and `tk.uncached_path()` would give a plain string -- but a bare string
    # carries no `.creator`, so the CreateLmImageJob dependency is lost and the image is never built
    # (RASR then fails to open the missing image for writing). Use Sisyphus' Path.copy() (which keeps
    # the `.creator`, unlike copy.copy() which drops it when INCLUDE_CREATOR_STATE is False) and clear
    # `cached`: this keeps the dependency (image gets built) while rendering a plain path.
    lm_image = CreateLmImageJob(crp, mem=8).out_image.copy()
    lm_image.cached = False
    rasr_config.image = lm_image

    return rasr_config
