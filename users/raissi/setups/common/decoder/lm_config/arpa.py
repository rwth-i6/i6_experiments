__all__ = ["ArpaLmRasrConfig"]

from dataclasses import dataclass
from typing import Optional

from sisyphus import tk

import i6_core.rasr as rasr


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

    def get(self) -> rasr.RasrConfig:
        lm_config = rasr.RasrConfig()

        lm_config.type = "ARPA"
        lm_config.file = self.lm_path

        if self.scale is not None:
            lm_config.scale = self.scale

        if self.image is not None:
            lm_config.image = self.image

        return lm_config
