__all__ = ["ArpaLmRasrConfig"]

from dataclasses import dataclass
from typing import Optional

from sisyphus import tk

import i6_core.lm as lm
import i6_core.rasr as rasr

from ..lex_config import LexiconRasrConfig


@dataclass()
class ArpaLmRasrConfig:
    """
    Class for ARPA LM Params in RASR Config

    lm_path: path to ARPA LM
    scale: LM scale
    image: global cache image
    """

    lm_path: tk.Path
    scale: Optional[float] = None
    image: Optional[tk.Path] = None
    lexicon_config: Optional[LexiconRasrConfig] = None

    def get(self) -> rasr.RasrConfig:
        lm_config = rasr.RasrConfig()

        lm_config.type = "ARPA"
        lm_config.file = self.lm_path

        if self.scale is not None:
            lm_config.scale = self.scale

        if self.image is not None:
            lm_config.image = self.image
        elif self.lexicon_config is not None:
            crp = rasr.CommonRasrParameters()
            crp.lexicon_config = self.lexicon_config.get()
            crp.language_model_config = lm_config
            lm_config.image = lm.CreateLmImageJob(crp=crp).out_image

        return lm_config
