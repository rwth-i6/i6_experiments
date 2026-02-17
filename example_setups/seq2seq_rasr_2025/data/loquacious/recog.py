from dataclasses import dataclass
from typing import Optional
from .lm import WordLmParams
from ...model_pipelines.common.recog_rasr_config import TreeTimesyncRecogParams


@dataclass
class LoquaciousTreeTimesyncRecogParams(TreeTimesyncRecogParams):
    word_lm_params: Optional[WordLmParams] = None
