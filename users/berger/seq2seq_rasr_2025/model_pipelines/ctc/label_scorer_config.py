__all__ = ["get_ctc_prefix_label_scorer_config"]


from i6_core.rasr.config import RasrConfig

from ..common.label_scorer_config import get_no_op_label_scorer_config
from .pytorch_modules import ConformerCTCConfig


def get_ctc_prefix_label_scorer_config(model_config: ConformerCTCConfig, scale: float = 1.0) -> RasrConfig:
    rasr_config = RasrConfig()
    rasr_config.type = "ctc-prefix"
    rasr_config.blank_label_index = model_config.target_size - 1
    rasr_config.vocab_size = model_config.target_size

    rasr_config.ctc_scorer = get_no_op_label_scorer_config()

    if scale != 1.0:
        rasr_config.scale = scale

    return rasr_config
