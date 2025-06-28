from enum import Enum

from i6_core import returnn


class Criterion(Enum):
    LOCAL_CRF = "local_crf"
    LABEL_CRF = "label_crf"
    LABEL_TRANS_CRF = "label_trans_crf"


    def __str__(self):
        return self.value


@dataclass(frozen=True, eq=True)
class CRF:
    criterion: Criterion
    label_prior_type: PriorType
    label_prior: returnn.CodeWrapper
    label_prior_estimation_axes: str


