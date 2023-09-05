from typing import Dict


def get_preload_config(num_prim_layers: int, num_sec_layers: int, num_mas_layers: int) -> Dict:
    name_map = {}

    # frontend
    name_map["vgg_conv_prim_conv_1/W"] = "conv0_0/W"
    name_map["vgg_conv_prim_conv_1/bias"] = "conv0_0/bias"
    name_map["vgg_conv_sec_conv_1/W"] = "conv0_0/W"
    name_map["vgg_conv_sec_conv_1/bias"] = "conv0_0/bias"

    name_map["vgg_conv_prim_conv_2/W"] = "conv0_1/W"
    name_map["vgg_conv_prim_conv_2/bias"] = "conv0_1/bias"
    name_map["vgg_conv_sec_conv_2/W"] = "conv0_1/W"
    name_map["vgg_conv_sec_conv_2/bias"] = "conv0_1/bias"

    name_map["vgg_conv_prim_conv_3/W"] = "conv1_0/W"
    name_map["vgg_conv_prim_conv_3/bias"] = "conv1_0/bias"
    name_map["vgg_conv_sec_conv_3/W"] = "conv1_0/W"
    name_map["vgg_conv_sec_conv_3/bias"] = "conv1_0/bias"

    name_map["vgg_conv_prim_conv_4/W"] = "conv1_1/W"
    name_map["vgg_conv_prim_conv_4/bias"] = "conv1_1/bias"
    name_map["vgg_conv_sec_conv_4/W"] = "conv1_1/W"
    name_map["vgg_conv_sec_conv_4/bias"] = "conv1_1/bias"

    name_map["vgg_conv_prim_linear/W"] = "embedding/W"
    name_map["vgg_conv_prim_linear/b"] = "embedding/b"

    name_map["vgg_conv_sec_linear/W"] = "embedding/W"
    name_map["vgg_conv_sec_linear/b"] = "embedding/b"

    name_map["vgg_conv_prim_ln/scale"] = "enc_001_ff1_laynorm/scale"
    name_map["vgg_conv_prim_ln/bias"] = "enc_001_ff1_laynorm/bias"

    name_map["vgg_conv_sec_ln/scale"] = "enc_001_ff1_laynorm/scale"
    name_map["vgg_conv_sec_ln/bias"] = "enc_001_ff1_laynorm/bias"

    # prim enc
    for name in ["prim", "sec", "mas"]:
        if name == "prim":
            num_layers = num_prim_layers
        elif name == "sec":
            num_layers = num_sec_layers
        else:
            num_layers = num_mas_layers
        for layer in range(1, num_layers + 1):
            orig_layer = layer if name != "mas" else layer + num_prim_layers
            name_map[f"conformer_{name}_{layer}_ffmod_1_ff_1/W"] = f"enc_{orig_layer:03d}_ff1_conv1/W"
            name_map[f"conformer_{name}_{layer}_ffmod_1_ff_1/b"] = f"enc_{orig_layer:03d}_ff1_conv1/b"
            name_map[f"conformer_{name}_{layer}_ffmod_1_ff_2/W"] = f"enc_{orig_layer:03d}_ff1_conv2/W"
            name_map[f"conformer_{name}_{layer}_ffmod_1_ff_2/b"] = f"enc_{orig_layer:03d}_ff1_conv2/b"
            name_map[f"conformer_{name}_{layer}_ffmod_1_ln/scale"] = f"enc_{orig_layer:03d}_self_att_laynorm/scale"
            name_map[f"conformer_{name}_{layer}_ffmod_1_ln/bias"] = f"enc_{orig_layer:03d}_self_att_laynorm/bias"
            name_map[
                f"conformer_{name}_{layer}_mhsamod_rel_pos_enc/encoding_matrix"
            ] = f"enc_{orig_layer:03d}_rel_pos/encoding_matrix"
            name_map[f"conformer_{name}_{layer}_mhsamod_self_attention/QKV"] = f"enc_{orig_layer:03d}_self_att_att/QKV"
            name_map[f"conformer_{name}_{layer}_mhsamod_att_linear/W"] = f"enc_{orig_layer:03d}_self_att_lin/W"
            name_map[f"conformer_{name}_{layer}_mhsamod_ln/scale"] = f"enc_{orig_layer:03d}_conv_laynorm/scale"
            name_map[f"conformer_{name}_{layer}_mhsamod_ln/bias"] = f"enc_{orig_layer:03d}_conv_laynorm/bias"
            name_map[
                f"conformer_{name}_{layer}_convmod_2_pointwise_conv_1/W"
            ] = f"enc_{orig_layer:03d}_conv_pointwise1/W"
            name_map[f"conformer_{name}_{layer}_convmod_2_depthwise_conv/W"] = f"enc_{orig_layer:03d}_conv_depthwise/W"
            name_map[
                f"conformer_{name}_{layer}_convmod_2_depthwise_conv/bias"
            ] = f"enc_{orig_layer:03d}_conv_depthwise/bias"
            name_map[
                f"conformer_{name}_{layer}_convmod_2_pointwise_conv_2/W"
            ] = f"enc_{orig_layer:03d}_conv_pointwise2/W"
            name_map[
                f"conformer_{name}_{layer}_convmod_2_replace_bn/scale"
            ] = f"enc_{orig_layer:03d}_conv_layer_norm/scale"
            name_map[
                f"conformer_{name}_{layer}_convmod_2_replace_bn/bias"
            ] = f"enc_{orig_layer:03d}_conv_layer_norm/bias"
            name_map[f"conformer_{name}_{layer}_convmod_2_ln/scale"] = f"enc_{orig_layer:03d}_ff2_laynorm/scale"
            name_map[f"conformer_{name}_{layer}_convmod_2_ln/bias"] = f"enc_{orig_layer:03d}_ff2_laynorm/bias"
            name_map[f"conformer_{name}_{layer}_ffmod_2_ff_1/W"] = f"enc_{orig_layer:03d}_ff2_conv1/W"
            name_map[f"conformer_{name}_{layer}_ffmod_2_ff_1/b"] = f"enc_{orig_layer:03d}_ff2_conv1/b"
            name_map[f"conformer_{name}_{layer}_ffmod_2_ff_2/W"] = f"enc_{orig_layer:03d}_ff2_conv2/W"
            name_map[f"conformer_{name}_{layer}_ffmod_2_ff_2/b"] = f"enc_{orig_layer:03d}_ff2_conv2/b"

            if orig_layer < 12:
                name_map[f"conformer_{name}_{layer}_ffmod_2_ln/scale"] = f"enc_{orig_layer + 1:03d}_ff1_laynorm/scale"
                name_map[f"conformer_{name}_{layer}_ffmod_2_ln/bias"] = f"enc_{orig_layer + 1:03d}_ff1_laynorm/bias"
            else:
                name_map[f"conformer_{name}_{layer}_ffmod_2_ln/scale"] = "encoder/scale"
                name_map[f"conformer_{name}_{layer}_ffmod_2_ln/bias"] = "encoder/bias"

    name_map["aux_prim_transposed_conv/W_native_transposed_conv"] = "aux_6_upsampled0/W_native_transposed_conv"
    name_map["aux_prim_transposed_conv/bias"] = "aux_6_upsampled0/bias"
    name_map["aux_prim_mlp_1/W"] = "aux_6_ff1/W"
    name_map["aux_prim_mlp_1/b"] = "aux_6_ff1/b"
    name_map["aux_prim_mlp_2/W"] = "aux_6_ff2/W"
    name_map["aux_prim_mlp_2/b"] = "aux_6_ff2/b"
    name_map["aux_output_prim/W"] = "aux_6_output_prob/W"
    name_map["aux_output_prim/b"] = "aux_6_output_prob/b"
    name_map["output_transposed_conv/W_native_transposed_conv"] = "upsampled0/W_native_transposed_conv"
    name_map["output_transposed_conv/bias"] = "upsampled0/bias"
    name_map["output/W"] = "output/W"
    name_map["output/b"] = "output/b"

    return {
        "librispeech_am": {
            "filename": "/work/asr4/vieting/setups/converse/dependencies/librispeech_hybrid_conformer_training_job/output/models/epoch.600",
            "ignore_missing": True,
            "init_for_train": True,
            "var_name_mapping": name_map,
        }
    }
