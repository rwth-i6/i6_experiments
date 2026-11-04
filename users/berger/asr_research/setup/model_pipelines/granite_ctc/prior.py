__all__ = ["compute_priors"]

from typing import Optional

from i6_core.returnn import PtCheckpoint
from sisyphus import tk

from ...data.base import DataConfig
from ..common.prior import compute_priors as _compute_priors
from ..common.serializers import get_model_serializers
from .pytorch_modules import GraniteCtcConfig, GraniteCtcModel


def compute_priors(
    prior_data_config: DataConfig,
    model_config: GraniteCtcConfig,
    checkpoint: Optional[PtCheckpoint] = None,
) -> tk.Path:
    """
    Estimate label priors of an external CTC model on corpus data.

    The model carries its own weights, so ``checkpoint`` stays None. The priors are corpus specific
    even though the model is not, since they describe the label distribution on the given data.
    """
    model_serializers = get_model_serializers(model_class=GraniteCtcModel, model_config=model_config)
    return _compute_priors(
        prior_data_config=prior_data_config, model_serializers=model_serializers, checkpoint=checkpoint
    )
