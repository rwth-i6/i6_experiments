from dataclasses import dataclass
from enum import Enum

from i6_core import returnn

from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import PriorType


class Criterion(Enum):
    LOCAL_CRF = "local_crf"
    GLOBAL_CRF = "global_crf"

    def __str__(self):
        return self.value


@dataclass(frozen=True, eq=True)
class CRF:
    criterion: Criterion
    factorization_type: str
    label_prior: returnn.CodeWrapper
    label_prior_type: PriorType
    label_prior_estimation_axes: str


