__all__ = ["CombineLmRasrConfig"]

from typing import List, Optional

import i6_core.rasr as rasr


class CombineLmRasrConfig:
    def __init__(
        self,
        combine_scale: float,
        lm_configs: List[rasr.RasrConfig],
        *,
        lookahead_lm_idx: Optional[int] = None,
        recombine_lm_idx: Optional[int] = None,
        linear_combination: bool = False,
    ):
        self.combine_scale = combine_scale
        self.lm_configs = lm_configs
        self.linear_combination = linear_combination
        self.lookahead_lm_idx = lookahead_lm_idx
        self.recombine_lm_idx = recombine_lm_idx

    def get(self) -> rasr.RasrConfig:
        lm_config = rasr.RasrConfig()

        lm_config.type = "combine"
        lm_config.scale = self.combine_scale
        lm_config.num_lms = len(self.lm_configs)

        for i, lm in enumerate(self.lm_configs):
            lm_config[f"lm-{i+1}"] = lm

        if self.lookahead_lm_idx is not None:
            lm_config.lookahead_lm = self.lookahead_lm_idx

        if self.recombine_lm_idx is not None:
            lm_config.recombination_lm = self.recombine_lm_idx

        if self.linear_combination:
            lm_config.linear_combination = True

        return lm_config
