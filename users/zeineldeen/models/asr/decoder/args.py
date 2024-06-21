from dataclasses import dataclass
from typing import Optional


class DecoderArgs:
    pass


@dataclass
class TransformerDecoderArgs(DecoderArgs):
    num_layers: int = 6
    att_num_heads: int = 8
    ff_dim: int = 2048
    ff_act: str = "relu"
    pos_enc: Optional[str] = None
    embed_pos_enc: bool = False

    # param init
    ff_init: Optional[str] = None
    mhsa_init: Optional[str] = None
    mhsa_out_init: Optional[str] = None

    # dropout
    dropout: float = 0.1
    att_dropout: float = 0.1
    embed_dropout: float = 0.1
    softmax_dropout: float = 0.0

    ff_weight_noise: Optional[float] = None
    mhsa_weight_noise: Optional[float] = None
    ff_weight_dropout: Optional[float] = None
    mhsa_weight_dropout: Optional[float] = None

    # other regularization
    l2: float = 0.0
    self_att_l2: float = 0.0
    rel_pos_clipping: int = 16
    label_smoothing: float = 0.1
    apply_embed_weight: bool = False

    length_normalization: bool = True

    # ILM
    replace_cross_att_w_masked_self_att: bool = False
    create_ilm_decoder: bool = False
    ilm_type: bool = None
    ilm_args: Optional[dict] = None


@dataclass
class ConformerDecoderArgs(DecoderArgs):
    num_layers: int = 6
    att_num_heads: int = 8
    ff_dim: int = 2048
    pos_enc: Optional[str] = "rel"

    # conv module
    conv_kernel_size: int = 32

    # param init
    ff_init: Optional[str] = None
    mhsa_init: Optional[str] = None
    mhsa_out_init: Optional[str] = None
    conv_module_init: Optional[str] = None

    # dropout
    dropout: float = 0.1
    att_dropout: float = 0.1
    embed_dropout: float = 0.1
    softmax_dropout: float = 0.1

    # other regularization
    l2: float = 0.0001
    frontend_conv_l2: float = 0.0001
    rel_pos_clipping: int = 16
    label_smoothing: float = 0.1
    apply_embed_weight: bool = False

    length_normalization: bool = True

    use_sqrd_relu: bool = False

    # ILM
    replace_cross_att_w_masked_self_att: bool = False
    create_ilm_decoder: bool = False
    ilm_type: bool = None
    ilm_args: Optional[dict] = None


@dataclass
class RNNDecoderArgs(DecoderArgs):
    att_num_heads: int = 1
    lstm_num_units: int = 1024
    output_num_units: int = 1024
    embed_dim: int = 640
    enc_key_dim: int = 1024  # also attention dim  # also attention dim

    # location feedback
    loc_conv_att_filter_size: Optional[int] = None

    # param init
    lstm_weights_init: Optional[str] = None
    embed_weight_init: Optional[str] = None

    # dropout
    dropout: float = 0.0
    softmax_dropout: float = 0.3
    att_dropout: float = 0.0
    embed_dropout: float = 0.1
    rec_weight_dropout: float = 0.0

    # other regularization
    l2: float = 0.0001
    zoneout: bool = True
    reduceout: bool = True

    # lstm lm
    lstm_lm_dim: int = 1024
    add_lstm_lm: bool = False

    length_normalization: bool = True
    length_normalization_exponent: float = 1.0

    coverage_scale: float = None
    coverage_threshold: float = None
    coverage_update: str = "sum"

    ce_loss_scale: Optional[float] = 1.0

    label_smoothing: float = 0.1

    use_zoneout_output: bool = False

    monotonic_att_weights_loss_opts: Optional[dict] = None
    use_monotonic_att_weights_loss_in_recog: Optional[bool] = False

    include_eos_in_search_output: bool = False
