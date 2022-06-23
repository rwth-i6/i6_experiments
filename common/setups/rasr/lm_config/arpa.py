__all__ = ["ArpaLmRasrConfig"]

from typing import Optional

from sisyphus import tk

import i6_core.rasr as rasr


class ArpaLmRasrConfig:
    def __init__(self, lm_path: tk.Path, scale: Optional[float] = None):
        self.lm_path = lm_path
        self.scale = scale

    def get(self) -> rasr.RasrConfig:
        lm_config = rasr.RasrConfig()

        lm_config.type = "ARPA"
        lm_config.file = self.lm_path
        if self.scale is not None:
            lm_config.scale = self.scale

        return lm_config
