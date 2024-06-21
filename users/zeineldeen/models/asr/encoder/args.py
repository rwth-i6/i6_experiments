from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class EncoderArgs:
    pass


@dataclass
class ConformerEncoderCommonArgs(EncoderArgs):
    num_blocks: int = 12
    enc_key_dim: int = 512
    att_num_heads: int = 8
    ff_dim: int = 2048
    conv_kernel_size: int = 32
    input: str = "data"
    input_layer: str = "lstm-6"
    input_layer_conv_act: str = "relu"
    add_abs_pos_enc_to_input: bool = False
    pos_enc: str = "rel"

    sandwich_conv: bool = False
    subsample: Optional[str] = None
    use_causal_layers: bool = False

    # ctc
    with_ctc: bool = True
    native_ctc: bool = True
    ctc_loss_scale: Optional[float] = None
    ctc_self_align_delay: Optional[int] = None
    ctc_self_align_scale: float = 0.5
    ctc_dropout: float = 0.0

    # param init
    ff_init: Optional[str] = None
    mhsa_init: Optional[str] = None
    mhsa_out_init: Optional[str] = None
    conv_module_init: Optional[str] = None
    start_conv_init: Optional[str] = None

    # dropout
    dropout: float = 0.1
    dropout_in: float = 0.1
    att_dropout: float = 0.1
    lstm_dropout: float = 0.1

    # weight dropout
    ff_weight_dropout: Optional[float] = None
    mhsa_weight_dropout: Optional[float] = None
    conv_weight_dropout: Optional[float] = None

    # norms
    batch_norm_opts: Optional[Dict[str, Any]] = None
    use_ln: bool = False

    # other regularization
    l2: float = 0.0001
    frontend_conv_l2: float = 0.0001
    self_att_l2: float = 0.0
    rel_pos_clipping: int = 16

    use_sqrd_relu: bool = False

    convolution_first: bool = False


@dataclass
class ConformerEncoderArgs(ConformerEncoderCommonArgs):
    weight_noise: Optional[float] = None
    weight_noise_layers: Optional[List[str]] = None


@dataclass
class ConformerEncoderV2Args(ConformerEncoderCommonArgs):
    # weight noise
    ff_weight_noise: Optional[float] = None
    mhsa_weight_noise: Optional[float] = None
    conv_weight_noise: Optional[float] = None
    frontend_conv_weight_noise: Optional[float] = None

    # weight dropout
    frontend_conv_weight_dropout: Optional[float] = None


@dataclass
class EBranchformerEncoderArgs(ConformerEncoderV2Args):
    pass
