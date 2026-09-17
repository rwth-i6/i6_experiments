from dataclasses import dataclass

from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr.config import RasrConfig
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
from sisyphus import tk

from ...tools import rasr_binary_path


@dataclass
class ArpaLmParams:
    scale: float = 1.0


def _get_arpa_lm_config(lexicon_file: tk.Path, params: ArpaLmParams) -> RasrConfig:
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    # rasr_config.file = tk.Path(
    #     "/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/4gram-pruned.arpa.gz",
    #     hash_overwrite="LOQUACIOUS_4GRAM",
    # )
    rasr_config.file = tk.Path(
        "/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/4gram-pruned-test2.arpa.gz",
        hash_overwrite="LOQUACIOUS_4GRAM_V2",
    )
    rasr_config.scale = params.scale

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = rasr_binary_path.join_right("lm-util.linux-x86_64-standard")  # type: ignore
    crp.language_model_config = rasr_config  # type: ignore
    crp.lexicon_config = RasrConfig()  # type: ignore
    crp.lexicon_config.file = lexicon_file  # type: ignore
    rasr_config.image = CreateLmImageJob(crp, mem=8).out_image

    return rasr_config


WordLmParams = ArpaLmParams


def get_word_lm_config(lexicon_file: tk.Path, params: WordLmParams) -> RasrConfig:
    if isinstance(params, ArpaLmParams):
        return _get_arpa_lm_config(lexicon_file=lexicon_file, params=params)
