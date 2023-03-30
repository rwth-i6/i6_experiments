import copy


from .layers import DEFAULT_INIT

# ----------------------------------------------------------

# 'num_heads': num_heads,
# 'model_dim': model_dim,
# 'key_dim': key_dim,
# 'value_dim': value_dim,
# 'ff_dim': ff_dim,
# 'kernel_size': kernel_size,
#
# 'emb_dropout': emb_dropout,
# 'sa_dropout': dropout,
# 'sa_post_dropout': dropout,
# 'conv_post_dropout': dropout,
# 'ff_activation_dropout': dropout,
# 'ff_post_dropout': dropout,
#
# 'initialization': DEFAULT_INIT,
# 'ff_activation': "swish",
# 'end_layernorm': False,
#
# 'normal_conv': False,
# 'output_channels': 16,
# 'kernel_size_for_feature': 3,
#
# 'attention_left_only': False,
# 'relative_pe': False,
# 'fixed': False,
# 'clipping': 100,
#
# 'linear_mapping': True,
# 'linear_mapping_bias': False,
# 'switch': False
# ---------


## encoder block specific parameters
def get_encoder_args(
    num_heads,
    key_dim_per_head,
    value_dim_per_head,
    model_dim,
    ff_dim,
    kernel_size,
    dropout=0.1,
    emb_dropout=0.0,
    **kwargs,
):

    key_dim = key_dim_per_head * num_heads
    value_dim = value_dim_per_head * num_heads

    enc_args = {
        "num_heads": num_heads,
        "model_dim": model_dim,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "ff_dim": ff_dim,
        "kernel_size": kernel_size,
        "emb_dropout": emb_dropout,
        "sa_dropout": dropout,
        "sa_post_dropout": dropout,
        "conv_post_dropout": dropout,
        "ff_activation_dropout": dropout,
        "ff_post_dropout": dropout,
        "initialization": DEFAULT_INIT,
        "ff_activation": "swish",
        "end_layernorm": False,
    }

    if kwargs:
        enc_args.update(kwargs)

    return enc_args


## acoustic model specific parameters: frond-end, downsampling,
def get_network_args(num_enc_layers, type, enc_args, target="classes", num_classes=12001, **kwargs):

    assert type in ["transformer", "conformer"]
    if type == "conformer":
        assert enc_args["ff_activation"] == "swish"

    network_args = {
        "target": target,
        "num_classes": num_classes,
        "num_enc_layers": num_enc_layers,
        "enc_args": enc_args,
        "type": type,
        "use_spec_augment": True,
        "use_pos_encoding": False,
        "add_to_input": True,
        "add_blstm_block": False,
        "blstm_args": None,
        "add_conv_block": True,
        "conv_args": None,
        "feature_stacking": False,
        "feature_stacking_before_frontend": False,
        "feature_stacking_window": None,
        "feature_stacking_stride": None,
        "reduction_factor": None,
        "alignment_reduction": False,
        "transposed_conv": False,
        "transposed_conv_args": None,
        "frame_repetition": False,
        "loss_layer_idx": None,
        "loss_scale": 0.3,
        "aux_loss_mlp_dim": 256,
        "mlp": False,
        "mlp_dim": 256,
        "feature_repre_idx": None,
        "att_weights_inspection": False,
        "inspection_idx": None,
        "window_limit_idx": None,
        "window_size": None,
        "was_idx": None,
        "upsilon": 0.5,
    }
    if kwargs:
        if kwargs.get("enc_args", None):
            network_args.update(kwargs.pop("enc_args"))

        if kwargs.get("src_embed_args", None):
            network_args.update(kwargs.pop("src_embed_args"))

        network_args.update(kwargs)

    return network_args


def add_time_chunking_and_unchunking_to_network(
    network,
    chunk_size,
    chunk_step,
    first_layer="source",
    next_layers="output",
    before_unchunking_layers="encoder",
):
    if isinstance(next_layers, str):
        next_layers = [next_layers]

    if isinstance(before_unchunking_layers, str):
        before_unchunking_layers = [before_unchunking_layers]

    assert len(next_layers) == len(before_unchunking_layers)

    network_new = copy.deepcopy(network)

    network_new["time_chunk"] = {
        "class": "time_chunking",
        "chunk_size": chunk_size,
        "chunk_step": chunk_step,
    }

    network_new[first_layer]["from"] = ["time_chunk"]

    for before_unchunking_layer, next_layer in zip(before_unchunking_layers, next_layers):

        network_new[f"{before_unchunking_layer}_time_unchunk"] = {
            "class": "time_unchunking",
            "chunking_layer": "time_chunk",
            "from": [before_unchunking_layer],
        }

        network_new[f"{before_unchunking_layer}_time_unchunk_swapped"] = {
            "class": "transpose",
            "perm": {"b": "t", "t": "b"},
            "from": [f"{before_unchunking_layer}_time_unchunk"],
        }

        network_new[next_layer]["from"] = [f"{before_unchunking_layer}_time_unchunk_swapped"]

    return network_new


enc_half_args = get_encoder_args(4, 64, 64, 256, 1024, 32)
enc_orig_half_args = get_encoder_args(4, 64, 64, 256, 1024, 32, **{"end_layernorm": True})

enc_standard_args = get_encoder_args(8, 64, 64, 512, 2048, 32)

# --------------------------------------------------------------
transformer_half_args = get_network_args(12, "transformer", enc_half_args)
transformer_standard_args = get_network_args(12, "transformer", enc_standard_args)

conformer_half_args = get_network_args(12, "conformer", enc_half_args)
conformer_standard_args = get_network_args(12, "transformer", enc_standard_args)
