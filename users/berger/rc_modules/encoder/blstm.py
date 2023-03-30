import copy
from returnn_common import nn

from returnn_common.nn.encoder.blstm import BlstmEncoder

hybrid_blstm_defaults = {
    "dim": 512,
    "num_layers": 6,
    "time_reduction": [],
    "l2": 1e-02,
    "dropout": 0.1,
}

transducer_blstm_defaults = {
    "dim": 512,
    "num_layers": 6,
    "time_reduction": [1, 2, 2],
    "l2": 1e-04,
    "dropout": 0.1,
}


class Blstm(BlstmEncoder):
    def __init__(
        self,
        dim: int = 512,
        **kwargs,
    ):
        super().__init__(dim=nn.FeatureDim("lstm_dim", dim), **kwargs)
