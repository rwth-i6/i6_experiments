from i6_experiments.users.berger.modules.blstm import BLSTMNetwork
from i6_experiments.users.berger.modules.ffnn import FFNetwork
from i6_experiments.users.berger.modules.data_augmentation import SpecAugment
from returnn_common import nn
from returnn_common.nn import hybrid_hmm
from typing import Tuple


class BLSTMHybridEncoder(nn.Module):
    def __init__(
        self,
        num_blstm_layers: int = 6,
        blstm_layer_size: int = 500,
        num_mlp_layers: int = 2,
        mlp_layer_size: int = 1024,
        **kwargs
    ):
        super().__init__()
        self.spec = SpecAugment(**kwargs)
        self.blstm = BLSTMNetwork(
            num_blstm_layers=num_blstm_layers, layer_size=blstm_layer_size, **kwargs
        )
        self.use_mlp = mlp_layer_size > 0
        self.mlp = FFNetwork(
            num_layers=num_mlp_layers, layer_size=mlp_layer_size, **kwargs
        )

    def __call__(self, input_data: nn.Tensor) -> Tuple[nn.Tensor, nn.Dim]:
        augmented_input = self.spec(input_data)
        out, out_dim = self.blstm(augmented_input)
        if self.use_mlp:
            out, out_dim = self.mlp(blstm_out)
        return out, out_dim


def make_blstm_hybrid_model(output_size: int, **kwargs) -> nn.Module:
    encoder = BLSTMHybridEncoder(**kwargs)

    output_dim = nn.FeatureDim("output", output_size)
    return hybrid_hmm.HybridHMM(encoder=encoder, out_dim=output_dim)
