__all__ = ["compute_priors"]

from i6_core.returnn import PtCheckpoint
from i6_experiments.common.setups.serialization import ExternalImport
from sisyphus import tk

from ....data.base import DataConfig
from ...common.prior import compute_priors as _compute_priors
from ...common.serializers import get_model_serializers
from .pytorch_modules import QATConformerCTCConfig, QATConformerCTCModel

from ....tools import synaptogen_ml_root



def compute_priors(
    prior_data_config: DataConfig,
    model_config: QATConformerCTCConfig,
    checkpoint: PtCheckpoint,
) -> tk.Path:
    model_serializers = get_model_serializers(model_class=QATConformerCTCModel, model_config=model_config)
    model_serializers.serializer_objects.insert(0, ExternalImport(synaptogen_ml_root))
    return _compute_priors(
        prior_data_config=prior_data_config, model_serializers=model_serializers, checkpoint=checkpoint
    )
