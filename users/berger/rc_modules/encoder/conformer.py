import copy
from i6_core.returnn.config import CodeWrapper
from returnn_common import nn

from returnn_common.nn.conformer import ConformerEncoder, ConformerConvSubsample


hybrid_vgg_defaults = {
    "out_dims": [32, 64, 64, 32],
    "filter_sizes": [3, 3, 3, 3],
    "strides": [1, 1, 1, (3, 1)],
    "pool_sizes": [(1, 2)],
    "activation": CodeWrapper("nn.swish"),
    "padding": "same",
}

hybrid_conformer_defaults = {
    "num_layers": 12,
    "input_dropout": 0.1,
    "out_dim": 384,
    "ff_dim": 1536,
    "ff_activation": CodeWrapper("nn.swish"),
    "dropout": 0.1,
    "conv_kernel_size": 8,
    "conv_norm": CodeWrapper("nn.LayerNorm"),
    "num_heads": 6,
    "att_dropout": 0.1,
}

transducer_vgg_defaults = {
    "out_dims": [32, 64, 64],
    "filter_sizes": [3, 3, 3],
    "strides": [1, (2, 1), (2, 1)],
    "pool_sizes": [(1, 2)],
    "activation": CodeWrapper("nn.swish"),
}

transducer_conformer_defaults = {
    "num_layers": 12,
    "input_dropout": 0.1,
    "out_dim": 512,
    "ff_dim": 2048,
    "ff_activation": CodeWrapper("nn.swish"),
    "dropout": 0.1,
    "conv_kernel_size": 32,
    "conv_norm": CodeWrapper("nn.BatchNorm"),
    "num_heads": 8,
    "att_dropout": 0.1,
}


class VGGConformer(ConformerEncoder):
    def __init__(
        self,
        in_dim: nn.Dim,
        vgg_args: dict = transducer_vgg_defaults,
        conformer_args: dict = transducer_conformer_defaults,
    ):
        vgg_args_mod = copy.deepcopy(vgg_args)
        conformer_args_mod = copy.deepcopy(conformer_args)
        if "out_dims" in vgg_args:
            vgg_args_mod["out_dims"] = [
                nn.FeatureDim(f"vgg_out_{i}", dim) for i, dim in enumerate(vgg_args["out_dims"])
            ]
        if "out_dim" in conformer_args:
            conformer_args_mod["out_dim"] = nn.FeatureDim("conformer_out_dim", conformer_args["out_dim"])
        if "ff_dim" in conformer_args:
            conformer_args_mod["ff_dim"] = nn.FeatureDim("conformer_ff_dim", conformer_args["ff_dim"])

        vgg = ConformerConvSubsample(in_dim=in_dim, **vgg_args_mod)
        super().__init__(in_dim=in_dim, input_layer=vgg, **conformer_args_mod)
