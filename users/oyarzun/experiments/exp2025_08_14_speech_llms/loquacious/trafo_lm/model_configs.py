from typing import Tuple

from sisyphus import tk

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ...pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2_cfg import (
    generate_transformer_block_config,
    TransformerLMConfig
)

hidden_dim = 768
trafo_block_config = generate_transformer_block_config(
    input_dim=hidden_dim,
    ff_dim=4096,
    output_dim=hidden_dim,
    num_heads=8,
    dropout=0.0,
)


def get_trafo_lm_config_v1(vocab_size: tk.Variable) -> Tuple[str, TransformerLMConfig]:
    """

    Args:
        vocab_size:

    Returns: network_module, net_args

    """
    return "lm.trafo.kazuki_trafo_zijian_variant_v2", TransformerLMConfig(
        embed_dim=128,
        hidden_dim=hidden_dim,
        vocab_dim=vocab_size,
        num_layers=24,
        block_config=trafo_block_config,
        batch_first=True,  # very important, state management in decoder does not work otherwise
        dropout=0.0,
    )
