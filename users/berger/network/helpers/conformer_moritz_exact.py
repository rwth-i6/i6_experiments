from i6_core.returnn import CodeWrapper


def add_frontend(network: dict, from_list: str, prefix: str) -> str:
    network.update(
        {
            f"{prefix}source0": {
                "axis": "F",
                "class": "split_dims",
                "dims": (-1, 1),
                "from": [from_list],
            },
            f"{prefix}conv0_0": {
                "activation": None,
                "class": "conv",
                "filter_size": (3, 3),
                "from": f"{prefix}source0",
                "in_spatial_dims": ["T", "dim:50"],
                "n_out": 32,
                "padding": "same",
                "with_bias": True,
            },
            f"{prefix}conv0_1": {
                "activation": "relu",
                "class": "conv",
                "filter_size": (3, 3),
                "from": f"{prefix}conv0_0",
                "in_spatial_dims": ["T", "dim:50"],
                "n_out": 32,
                "padding": "same",
                "with_bias": True,
            },
            f"{prefix}conv0p": {
                "class": "pool",
                "from": f"{prefix}conv0_1",
                "in_spatial_dims": ["T", "dim:50"],
                "mode": "max",
                "padding": "same",
                "pool_size": (1, 2),
                "strides": (1, 2),
            },
            f"{prefix}conv1_0": {
                "activation": None,
                "class": "conv",
                "filter_size": (3, 3),
                "from": f"{prefix}conv0p",
                "in_spatial_dims": ["T", "dim:25"],
                "n_out": 64,
                "padding": "same",
                "with_bias": True,
            },
            f"{prefix}conv1_1": {
                "activation": "relu",
                "class": "conv",
                "filter_size": (3, 3),
                "from": f"{prefix}conv1_0",
                "in_spatial_dims": ["T", "dim:25"],
                "n_out": 64,
                "padding": "same",
                "with_bias": True,
            },
            f"{prefix}conv1p": {
                "class": "pool",
                "from": f"{prefix}conv1_1",
                "in_spatial_dims": ["T", "dim:25"],
                "mode": "max",
                "padding": "same",
                "pool_size": (1, 1),
                "strides": (1, 1),
            },
            f"{prefix}conv_merged": {
                "axes": [f"stag:{prefix}conv0p:conv:s1", "dim:64"],
                "class": "merge_dims",
                "from": f"{prefix}conv1p",
            },
            f"{prefix}feature_stacking_merged": {
                "axes": (2, 3),
                "class": "merge_dims",
                "from": [f"{prefix}feature_stacking_window"],
            },
            f"{prefix}feature_stacking_window": {
                "class": "window",
                "from": [f"{prefix}conv_merged"],
                "stride": 3,
                "window_left": 2,
                "window_right": 0,
                "window_size": 3,
            },
            f"{prefix}embedding": {
                "activation": None,
                "class": "linear",
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}feature_stacking_merged"],
                "n_out": 512,
                "with_bias": True,
            },
            f"{prefix}embedding_dropout": {"class": "dropout", "dropout": 0.0, "from": [f"{prefix}embedding"]},
        }
    )

    return f"{prefix}embedding_dropout"


def add_conformer_block(network: dict, from_list: str, block_idx: int, prefix: str) -> str:
    network.update(
        {
            f"{prefix}enc_{block_idx:03d}": {"class": "copy", "from": [f"{prefix}enc_{block_idx:03d}_ff2_out"]},
            f"{prefix}enc_{block_idx:03d}_conv_GLU": {
                "activation": "identity",
                "class": "gating",
                "from": [f"{prefix}enc_{block_idx:03d}_conv_pointwise1"],
            },
            f"{prefix}enc_{block_idx:03d}_conv_act": {
                "activation": "swish",
                "class": "activation",
                "from": [f"{prefix}enc_{block_idx:03d}_conv_layer_norm"],
            },
            f"{prefix}enc_{block_idx:03d}_conv_depthwise": {
                "activation": None,
                "class": "conv",
                "filter_size": (32,),
                "from": [f"{prefix}enc_{block_idx:03d}_conv_GLU"],
                "groups": 512,
                "n_out": 512,
                "padding": "same",
                "with_bias": True,
            },
            f"{prefix}enc_{block_idx:03d}_conv_dropout": {
                "class": "dropout",
                "dropout": 0.1,
                "from": [f"{prefix}enc_{block_idx:03d}_conv_pointwise2"],
            },
            f"{prefix}enc_{block_idx:03d}_conv_layer_norm": {
                "class": "layer_norm",
                "from": [f"{prefix}enc_{block_idx:03d}_conv_depthwise"],
            },
            f"{prefix}enc_{block_idx:03d}_conv_laynorm": {
                "class": "layer_norm",
                "from": [f"{prefix}enc_{block_idx:03d}_self_att_out"],
            },
            f"{prefix}enc_{block_idx:03d}_conv_output": {
                "class": "combine",
                "from": [f"{prefix}enc_{block_idx:03d}_self_att_out", f"{prefix}enc_{block_idx:03d}_conv_dropout"],
                "kind": "add",
                "n_out": 512,
            },
            f"{prefix}enc_{block_idx:03d}_conv_pointwise1": {
                "activation": None,
                "class": "linear",
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}enc_{block_idx:03d}_conv_laynorm"],
                "n_out": 1024,
                "with_bias": False,
            },
            f"{prefix}enc_{block_idx:03d}_conv_pointwise2": {
                "activation": None,
                "class": "linear",
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}enc_{block_idx:03d}_conv_act"],
                "n_out": 512,
                "with_bias": False,
            },
            f"{prefix}enc_{block_idx:03d}_ff1_conv1": {
                "activation": "swish",
                "class": "linear",
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}enc_{block_idx:03d}_ff1_laynorm"],
                "n_out": 2048,
                "with_bias": True,
            },
            f"{prefix}enc_{block_idx:03d}_ff1_conv2": {
                "activation": None,
                "class": "linear",
                "dropout": 0.1,
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}enc_{block_idx:03d}_ff1_conv1"],
                "n_out": 512,
                "with_bias": True,
            },
            f"{prefix}enc_{block_idx:03d}_ff1_drop": {
                "class": "dropout",
                "dropout": 0.1,
                "from": [f"{prefix}enc_{block_idx:03d}_ff1_conv2"],
            },
            f"{prefix}enc_{block_idx:03d}_ff1_drop_half": {
                "class": "eval",
                "eval": "0.5 * source(0)",
                "from": [f"{prefix}enc_{block_idx:03d}_ff1_drop"],
            },
            f"{prefix}enc_{block_idx:03d}_ff1_laynorm": {"class": "layer_norm", "from": [from_list]},
            f"{prefix}enc_{block_idx:03d}_ff1_out": {
                "class": "combine",
                "from": [from_list, f"{prefix}enc_{block_idx:03d}_ff1_drop_half"],
                "kind": "add",
            },
            f"{prefix}enc_{block_idx:03d}_ff2_conv1": {
                "activation": "swish",
                "class": "linear",
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}enc_{block_idx:03d}_ff2_laynorm"],
                "n_out": 2048,
                "with_bias": True,
            },
            f"{prefix}enc_{block_idx:03d}_ff2_conv2": {
                "activation": None,
                "class": "linear",
                "dropout": 0.1,
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}enc_{block_idx:03d}_ff2_conv1"],
                "n_out": 512,
                "with_bias": True,
            },
            f"{prefix}enc_{block_idx:03d}_ff2_drop": {
                "class": "dropout",
                "dropout": 0.1,
                "from": [f"{prefix}enc_{block_idx:03d}_ff2_conv2"],
            },
            f"{prefix}enc_{block_idx:03d}_ff2_drop_half": {
                "class": "eval",
                "eval": "0.5 * source(0)",
                "from": [f"{prefix}enc_{block_idx:03d}_ff2_drop"],
            },
            f"{prefix}enc_{block_idx:03d}_ff2_laynorm": {
                "class": "layer_norm",
                "from": [f"{prefix}enc_{block_idx:03d}_conv_output"],
            },
            f"{prefix}enc_{block_idx:03d}_ff2_out": {
                "class": "combine",
                "from": [f"{prefix}enc_{block_idx:03d}_conv_output", f"{prefix}enc_{block_idx:03d}_ff2_drop_half"],
                "kind": "add",
            },
            f"{prefix}enc_{block_idx:03d}_rel_pos": {
                "class": "relative_positional_encoding",
                "clipping": 400,
                "fixed": False,
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}enc_{block_idx:03d}_self_att_laynorm"],
                "n_out": 64,
            },
            f"{prefix}enc_{block_idx:03d}_self_att_att": {
                "attention_dropout": 0.1,
                "attention_left_only": False,
                "class": "self_attention",
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}enc_{block_idx:03d}_self_att_laynorm"],
                "key_shift": f"{prefix}enc_{block_idx:03d}_rel_pos",
                "n_out": 512,
                "num_heads": 8,
                "total_key_dim": 512,
            },
            f"{prefix}enc_{block_idx:03d}_self_att_drop": {
                "class": "dropout",
                "dropout": 0.1,
                "from": [f"{prefix}enc_{block_idx:03d}_self_att_lin"],
            },
            f"{prefix}enc_{block_idx:03d}_self_att_laynorm": {
                "class": "layer_norm",
                "from": [f"{prefix}enc_{block_idx:03d}_ff1_out"],
            },
            f"{prefix}enc_{block_idx:03d}_self_att_lin": {
                "activation": None,
                "class": "linear",
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": [f"{prefix}enc_{block_idx:03d}_self_att_att"],
                "n_out": 512,
                "with_bias": False,
            },
            f"{prefix}enc_{block_idx:03d}_self_att_out": {
                "class": "combine",
                "from": [f"{prefix}enc_{block_idx:03d}_ff1_out", f"{prefix}enc_{block_idx:03d}_self_att_drop"],
                "kind": "add",
                "n_out": 512,
            },
        }
    )

    return f"{prefix}enc_{block_idx:03d}"


def add_aux_output(network: dict, from_list: str, target: str) -> None:
    network.update(
        {
            "aux_6_ff1": {
                "activation": "relu",
                "class": "linear",
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": ["aux_6_length_masked"],
                "n_out": 256,
                "with_bias": True,
            },
            "aux_6_ff2": {
                "activation": None,
                "class": "linear",
                "forward_weights_init": "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
                "from": ["aux_6_ff1"],
                "n_out": 256,
                "with_bias": True,
            },
            "aux_6_length_masked": {
                "axis": "T",
                "class": "slice_nd",
                "from": ["aux_6_upsampled0"],
                "size": CodeWrapper("__time_tag__"),
                "start": 0,
            },
            "aux_6_output_prob": {
                "class": "softmax",
                "dropout": 0.0,
                "from": ["aux_6_ff2"],
                "loss": "ce",
                "loss_opts": {
                    "focal_loss_factor": 2.0,
                    "label_smoothing": 0.0,
                    "use_normalized_loss": False,
                },
                "loss_scale": 0.5,
                "target": target,
            },
            "aux_6_upsampled0": {
                "activation": "relu",
                "class": "transposed_conv",
                "filter_size": (3,),
                "from": [from_list],
                "n_out": 512,
                "strides": (3,),
                "with_bias": True,
            },
        }
    )


def add_output(network: dict, from_list: str, target: str) -> None:
    network.update(
        {
            "upsampled0": {
                "activation": "relu",
                "class": "transposed_conv",
                "filter_size": (3,),
                "from": [from_list],
                "n_out": 512,
                "strides": (3,),
                "with_bias": True,
            },
            "length_masked": {
                "axis": "T",
                "class": "slice_nd",
                "from": ["upsampled0"],
                "size": CodeWrapper("__time_tag__"),
                "start": 0,
            },
            "output": {
                "class": "softmax",
                "dropout": 0.0,
                "from": ["length_masked"],
                "loss": "ce",
                "loss_opts": {
                    "focal_loss_factor": 2.0,
                    "label_smoothing": 0.0,
                    "use_normalized_loss": False,
                },
                "n_out": 12001,
                "target": target,
            },
        }
    )
