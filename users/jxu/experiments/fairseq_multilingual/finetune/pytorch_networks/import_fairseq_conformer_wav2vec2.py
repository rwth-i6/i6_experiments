"""Import fariseq's wav2vec2.0 pretrained weights to torchaudios's format.

For this module to work, you need `fairseq`.
"""
import re

from torch.nn import Module

from torchaudio.models import Wav2Vec2Model
from i6_experiments.users.jxu.experiments.fairseq_multilingual.finetuning.pytorch_networks.torch_conformer_wav2vec2 import wav2vec2_conformer_model

def _parse_config(w2v_model):
    encoder = w2v_model.encoder
    conv_layers = w2v_model.feature_extractor.conv_layers

    extractor_mode = "layer_norm"
    if "GroupNorm" in conv_layers[0][2].__class__.__name__:
        extractor_mode = "group_norm"
    else:
        extractor_mode = "layer_norm"

    conv_layer_config = [(l[0].out_channels, l[0].kernel_size[0], l[0].stride[0]) for l in conv_layers]

    if all(l[0].bias is None for l in conv_layers):
        conv_bias = False
    elif all(l[0].bias is not None for l in conv_layers):
        conv_bias = True
    else:
        raise ValueError("Either all the convolutions layers have bias term or none of them should.")

    config = {
        "extractor_mode": extractor_mode,
        "extractor_conv_layer_config": conv_layer_config,
        "extractor_conv_bias": conv_bias,
        "encoder_embed_dim": w2v_model.post_extract_proj.out_features,
        "encoder_projection_dropout": w2v_model.dropout_input.p,
        "encoder_pos_conv_kernel": encoder.pos_conv[0].kernel_size[0],
        "encoder_pos_conv_groups": encoder.pos_conv[0].groups,
        "encoder_num_layers": len(encoder.layers),
        "encoder_num_heads": encoder.layers[0].self_attn.num_heads,
        "encoder_attention_dropout": encoder.layers[0].self_attn.dropout_module.p,
        "encoder_ff_interm_features": encoder.layers[0].ffn1.w_1.out_features,
        "encoder_ff_interm_dropout": encoder.layers[0].ffn1.dropout1.p,
        "encoder_dropout": encoder.layers[0].ffn1.dropout2.p,
        "encoder_layer_norm_first": encoder.layer_norm_first,
        "encoder_layer_drop": encoder.layerdrop,
    }
    return config


def _map_key(key):
    key_ = key
    if key.startswith("wav2vec_model."):
        key = key.replace("wav2vec_model.", "")
    if re.match(r"(mask_emb|quantizer|project_q|final_proj|mask_emb)", key):
        return None
    # Feature Extractor
    # Group norm when "extractor_mode" is "default".
    # (Only the first layer)
    # "conv_layers.0.2.weight" -> "conv_layers.0.layer_norm.weight"
    # "conv_layers.0.2.bias"   -> "conv_layers.0.layer_norm.bias"
    match = re.match(r"feature_extractor\.conv_layers\.0\.2\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.0.layer_norm.{match.group(1)}"
    # Convolutions
    # "conv_layers.X.0.weight" -> "conv_layers.X.conv.weight"
    # "conv_layers.X.0.bias"   -> "conv_layers.X.conv.bias"
    match = re.match(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.conv.{match.group(2)}"
    # Layer norm when "extractor_mode" is "layer_norm".
    # "conv_layers.X.2.1.weight" -> "conv_layers.X.layer_norm.weight"
    # "conv_layers.X.2.1.bias"   -> "conv_layers.X.layer_norm.bias"
    match = re.match(r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.layer_norm.{match.group(2)}"
    match = re.match(r"post_extract_proj\.(weight|bias)", key)
    # Encoder - Feature projection
    if match:
        return f"encoder.feature_projection.projection.{match.group(1)}"
    match = re.match(r"layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.feature_projection.layer_norm.{match.group(1)}"
    
    # Encoder - Transformer - Convolutional positional embedding
    match = re.match(r"encoder\.pos_conv\.0\.(bias|weight_g|weight_v)", key)
    if match:
        return f"encoder.transformer.pos_conv_embed.conv.{match.group(1)}"
    match = re.match(r"encoder\.layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layer_norm.{match.group(1)}"
    
    # Encoder - Conformer - FFN1
    match = re.match(r"encoder\.layers\.(\d+)\.ffn(\d+)\.layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.ffn{match.group(2)}.sequential.0.{match.group(3)}"
    match = re.match(r"encoder\.layers\.(\d+)\.ffn(\d+)\.w_1\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.ffn{match.group(2)}.sequential.1.{match.group(3)}"
    match = re.match(r"encoder\.layers\.(\d+)\.ffn(\d+)\.w_2\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.ffn{match.group(2)}.sequential.4.{match.group(3)}"
    
    # Encoder - Conformer - MHSA
    match = re.match(r"encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.self_attn_layer_norm.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.self_attn\.((k_|v_|q_|out_)proj\.(weight|bias))", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.self_attn.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.self_attn.out_proj.{match.group(2)}"

    # Encoder - Conformer - Conv
    match = re.match(r"encoder\.layers\.(\d+)\.conv_module\.layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.conv_module.layer_norm.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.conv_module\.pointwise_conv1\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.conv_module.sequential.0.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.conv_module\.depthwise_conv\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.conv_module.sequential.2.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.conv_module\.batch_norm\.(weight|bias|running_mean|running_var|num_batches_tracked)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.conv_module.sequential.3.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.conv_module\.pointwise_conv2\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.conv_module.sequential.5.{match.group(2)}"
    # final layer norm in each block
    match = re.match(r"encoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.final_layer_norm.{match.group(2)}"
    match = re.match(r"proj\.(weight|bias)", key)
    # Auxiliary Module
    # Only relevant when loading fine-tuned models
    if match:
        return f"aux.{match.group(1)}"
    # Auxiliary Module
    # Only relevant when loading fine-tuned models
    if match:
        return f"aux.{match.group(1)}"
    # HuBERT Extension
    if key in ["label_embs_concat"]:
        return key
    raise ValueError(f"Unexpected key: {key_}")

def _convert_state_dict(state_dict):
    converted = {}
    for k, v in state_dict.items():
        k = _map_key(k)
        if k is not None:
            converted[k] = v
    return converted

def _import_wav2vec2_pretraining(original: Module) -> Wav2Vec2Model:
    config = _parse_config(original)
    model = wav2vec2_conformer_model(**config, aux_num_out=None)
    model.load_state_dict(_convert_state_dict(original.state_dict()), strict=True)
    return model

