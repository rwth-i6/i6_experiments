"""
Language model helpers
"""
import os
from typing import Optional, Tuple

from sisyphus import tk

from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr import CommonRasrParameters, RasrConfig, crp_add_default_output

from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

from .phmm_default_tools import KENLM_BINARY_PATH, LM_UTIL_EXE


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
    # LibRASR loads the LM in-process and does not expand Sisyphus cached-path markers like `cf ...`.
    # Use the plain filesystem path to the prebuilt LM image instead.
    rasr_config.image = tk.uncached_path(CreateLmImageJob(crp, mem=8).out_image)

    return rasr_config




def create_lm_image_for_lexicon(
    lexicon_file: tk.Path,
    scale: float = 1.0,
    *,
    output_prefix: Optional[str] = None,
    mem: int = 8,
) -> Tuple[RasrConfig, tk.Path]:
    """
    Create and register a RASR LM image for a specific lexicon.

    :param lexicon_file: lexicon used to build the LM image
    :param scale: LM scale used by lexical search
    :param output_prefix: optional Sisyphus output prefix for the LM image
    :param mem: memory requirement for CreateLmImageJob
    :return: tuple of RASR LM config and LM image path
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

    lm_image_job = CreateLmImageJob(crp, mem=mem)
    lm_image = lm_image_job.out_image
    # LibRASR loads the LM in-process and does not expand Sisyphus cached-path markers like `cf ...`.
    # Use the plain filesystem path to the prebuilt LM image instead.
    rasr_config.image = tk.uncached_path(lm_image)

    if output_prefix is not None:
        lm_image_job.add_alias(output_prefix + "/create_lm_image")
        tk.register_output(output_prefix + "/lm.image", lm_image)
    else:
        tk.register_output("lm_images/%s/lm.image" % os.path.basename(str(lexicon_file)), lm_image)

    return rasr_config, lm_image
