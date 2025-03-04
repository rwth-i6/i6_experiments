from typing import Dict
import numpy

import returnn.frontend as rf


_ParamMapping = {}  # type: Dict[str,str]


def _add_params():
  # frontend
  for layer_idx in [1, 2, 3]:
    # Tina uses 1-based indexing for the layers
    orig_name = f"conv_{layer_idx}"
    _ParamMapping.update(
      {
        f"encoder.input_layer.conv_layers.{layer_idx - 1}.filter": f"{orig_name}/W",
        f"encoder.input_layer.conv_layers.{layer_idx - 1}.bias": f"{orig_name}/bias",
      }
    )
  _ParamMapping.update(
    {
      "encoder.input_projection.weight": "input_linear/W",
    }
  )
  # conformer
  for layer_idx in range(12):
    # FF
    for sub in [1, 2]:
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.weight"
      ] = f"conformer_{layer_idx + 1}_ffmod_{sub}_linear_swish/W"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.bias"
      ] = f"conformer_{layer_idx + 1}_ffmod_{sub}_linear_swish/b"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.weight"
      ] = f"conformer_{layer_idx + 1}_ffmod_{sub}_dropout_linear/W"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.bias"
      ] = f"conformer_{layer_idx + 1}_ffmod_{sub}_dropout_linear/b"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.scale"
      ] = f"conformer_{layer_idx + 1}_ffmod_{sub}_ln/scale"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.bias"
      ] = f"conformer_{layer_idx + 1}_ffmod_{sub}_ln/bias"
    # conv
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.weight"
    ] = f"conformer_{layer_idx + 1}_conv_mod_pointwise_conv_1/W"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.bias"
    ] = f"conformer_{layer_idx + 1}_conv_mod_pointwise_conv_1/b"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.filter"
    ] = f"conformer_{layer_idx + 1}_conv_mod_depthwise_conv/W"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.bias"
    ] = f"conformer_{layer_idx + 1}_conv_mod_depthwise_conv/bias"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.weight"
    ] = f"conformer_{layer_idx + 1}_conv_mod_pointwise_conv_2/W"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.bias"
    ] = f"conformer_{layer_idx + 1}_conv_mod_pointwise_conv_2/b"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_layer_norm.scale"
    ] = f"conformer_{layer_idx + 1}_conv_mod_ln/scale"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_layer_norm.bias"
    ] = f"conformer_{layer_idx + 1}_conv_mod_ln/bias"
    # self-att
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att.qkv.weight"
    ] = f"conformer_{layer_idx + 1}_mhsa_mod_self_attention/QKV"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att.proj.weight"
    ] = f"conformer_{layer_idx + 1}_mhsa_mod_att_linear/W"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att_layer_norm.scale"
    ] = f"conformer_{layer_idx + 1}_mhsa_mod_ln/scale"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att_layer_norm.bias"
    ] = f"conformer_{layer_idx + 1}_mhsa_mod_ln/bias"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att.learned_pos_emb.pos_emb"
    ] = f"conformer_{layer_idx + 1}_mhsa_mod_relpos_encoding/encoding_matrix"
    # final layer norm
    _ParamMapping[
      f"encoder.layers.{layer_idx}.final_layer_norm.scale"
    ] = f"conformer_{layer_idx + 1}_output/scale"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.final_layer_norm.bias"
    ] = f"conformer_{layer_idx + 1}_output/bias"


_add_params()


def map_param_func_v2(reader, name: str, var: rf.Parameter) -> numpy.ndarray:
  """map params, TF to RF"""
  from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
  from i6_experiments.users.zeyer.returnn.convert.params import numpy as convert_params_np
  from i6_experiments.users.zeyer.returnn.convert.params import tf_to_rf_np as convert_params_tf_to_rf_np

  assert isinstance(reader, CheckpointReader)
  assert isinstance(var, rf.Parameter)

  tf_var_name = name.replace(".", "/")
  if reader.has_tensor(tf_var_name):
    return reader.get_tensor(tf_var_name)

  if name in _ParamMapping:
    var_name = _ParamMapping[name]
    assert reader.has_tensor(var_name)
    value = reader.get_tensor(var_name)
    assert isinstance(value, numpy.ndarray)
    if name.endswith(".filter"):
      value = convert_params_np.convert_tf_conv_to_pt_conv_filter(value)
    assert (
            value.shape == var.batch_shape
    ), f"new param {name} {var.batch_shape} vs ckpt param {var_name} {value.shape}"
    assert value.dtype.name == var.dtype, f"new param {name} {var.dtype} vs ckpt param {var_name} {value.dtype}"
    return value

  if ".conv_block.norm." in name:
    assert name.startswith("encoder.layers.")
    layer_idx = int(name.split(".")[2])
    value = convert_params_tf_to_rf_np.convert_tf_batch_norm_to_rf(
      reader=reader,
      rf_name=name,
      rf_prefix_name=f"encoder.layers.{layer_idx}.conv_block.norm.",
      tf_prefix_name=f"conformer_{layer_idx + 1}_conv_mod_bn/batch_norm/",
      var=var,
    )
    assert value.shape == var.batch_shape, name
    assert value.dtype.name == var.dtype, name
    return value

  raise NotImplementedError(f"cannot map {name!r} {var}")
