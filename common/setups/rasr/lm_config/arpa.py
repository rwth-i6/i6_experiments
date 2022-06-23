__all__ = ["ArpaLmRasrConfig"]

from typing import Optional

from sisyphus import tk

import i6_core.rasr as rasr


class ArpaLmRasrConfig:
    def __init__(
        self,
        lm_path: tk.Path,
        scale: Optional[float] = None,
        image: Optional[tk.Path] = None,
    ):
        """

        :param lm_path: path to ARPA LM
        :param scale: LM scale
        :param image: global cache image
        """
        self.lm_path = lm_path
        self.scale = scale
        self.image = image

    def get(self) -> rasr.RasrConfig:
        lm_config = rasr.RasrConfig()

        lm_config.type = "ARPA"
        lm_config.file = self.lm_path

        if self.scale is not None:
            lm_config.scale = self.scale

        if self.image is not None:
            lm_config.image = self.image

        return lm_config
