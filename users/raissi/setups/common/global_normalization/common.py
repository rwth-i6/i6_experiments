from dataclasses import dataclass
from enum import Enum
from typing import Optional

from i6_core import returnn

from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import PriorType



class FactorizationType(Enum):
    POST = "posterior"
    POST_LAB = "posterior_label"

    def __str__(self):
        return self.value



class Criterion(Enum):
    LOCAL_CRF = "local_crf"
    GLOBAL_CRF = "global_crf"

    def __str__(self):
        return self.value


@dataclass(frozen=True, eq=True)
class CRF:
    criterion: Criterion
    factorization_type: FactorizationType
    label_prior_type: PriorType
    label_prior_estimation_axes: str
    label_prior: Optional[returnn.CodeWrapper] = None


