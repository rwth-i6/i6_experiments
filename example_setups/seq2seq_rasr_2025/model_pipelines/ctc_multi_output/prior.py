__all__ = ["compute_priors"]

from dataclasses import fields
from i6_core.returnn import PtCheckpoint
from sisyphus import tk

from ...data.base import DataConfig
from ..common.prior import compute_priors as _compute_priors
from ..common.serializers import get_model_serializers
from .pytorch_modules import (
    ConformerCTCMultiOutputConfig,
    ConformerCTCMultiOutputPriorConfig,
    ConformerCTCMultiOutputPriorModel,
)


def compute_priors(
    prior_data_config: DataConfig,
    model_config: ConformerCTCMultiOutputConfig,
    output_idx: int,
    checkpoint: PtCheckpoint,
) -> tk.Path:
    prior_model_config = ConformerCTCMultiOutputPriorConfig(
        **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        output_idx=output_idx,
    )
    model_serializers = get_model_serializers(
        model_class=ConformerCTCMultiOutputPriorModel, model_config=prior_model_config
    )
    return _compute_priors(
        prior_data_config=prior_data_config, model_serializers=model_serializers, checkpoint=checkpoint
    )
