__all__ = ["compute_priors"]

from i6_core.returnn import PtCheckpoint
from sisyphus import tk

from ...data.base import DataConfig
from ..common.prior import compute_priors as _compute_priors
from ..common.serializers import get_model_serializers
from .pytorch_modules import CombinationCTCPriorModel, CombinationModelConfig


def compute_priors(
    prior_data_config: DataConfig,
    model_config: CombinationModelConfig,
    checkpoint: PtCheckpoint,
) -> tk.Path:
    model_serializers = get_model_serializers(model_class=CombinationCTCPriorModel, model_config=model_config)
    return _compute_priors(
        prior_data_config=prior_data_config, model_serializers=model_serializers, checkpoint=checkpoint
    )
