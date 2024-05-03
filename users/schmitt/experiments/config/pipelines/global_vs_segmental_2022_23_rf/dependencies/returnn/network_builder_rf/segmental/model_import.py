from typing import Dict
import numpy

import returnn.frontend as rf


_ParamMapping = {}  # type: Dict[str,str]


def _add_params():
  # frontend
  for layer_idx in [0, 1, 2]:
    orig_name = "conv0" if layer_idx == 0 else f"subsample_conv{layer_idx - 1}"
    _ParamMapping.update(
      {
        f"encoder.input_layer.conv_layers.{layer_idx}.filter": f"{orig_name}/W",
        f"encoder.input_layer.conv_layers.{layer_idx}.bias": f"{orig_name}/bias",
      }
    )
  _ParamMapping.update(
    {
      "encoder.input_projection.weight": "source_linear/W",
      "enc_ctx.weight": "enc_ctx/W",
      "enc_ctx.bias": "enc_ctx/b",
      "inv_fertility.weight": "inv_fertility/W",
      "target_embed.weight": "output/rec/target_embed0/W",
      "weight_feedback.weight": "output/rec/weight_feedback/W",
      "s_transformed.weight": "output/rec/s_transformed/W",
      "energy.weight": "output/rec/energy/W",
      "readout_in.weight": "output/rec/readout_in/W",
      "readout_in.bias": "output/rec/readout_in/b",
      "output_prob.weight": "output/rec/output_prob/W",
      "output_prob.bias": "output/rec/output_prob/b",
      "target_embed_length_model.weight": "output/rec/target_embed_length_model/W",
      "emit_prob.weight": "output/rec/emit_prob0/W",
      "emit_prob.bias": "output/rec/emit_prob0/b",
    }
  )
  # conformer
  for layer_idx in range(12):
    # FF
    for sub in [1, 2]:
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.weight"
      ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/W"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.bias"
      ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/b"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.weight"
      ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/W"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.bias"
      ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/b"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.scale"
      ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/scale"
      _ParamMapping[
        f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.bias"
      ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/bias"
    # conv
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.weight"
    ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/W"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.bias"
    ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/b"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.filter"
    ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/W"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.bias"
    ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/bias"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.weight"
    ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/W"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.bias"
    ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/b"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_layer_norm.scale"
    ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_ln/scale"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.conv_layer_norm.bias"
    ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_ln/bias"
    # self-att
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att.qkv.weight"
    ] = f"conformer_block_{layer_idx + 1:02d}_self_att/QKV"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att.proj.weight"
    ] = f"conformer_block_{layer_idx + 1:02d}_self_att_linear/W"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att_layer_norm.scale"
    ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln/scale"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att_layer_norm.bias"
    ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln/bias"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.self_att.learned_pos_emb.pos_emb"
    ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln_rel_pos_enc/encoding_matrix"
    # final layer norm
    _ParamMapping[
      f"encoder.layers.{layer_idx}.final_layer_norm.scale"
    ] = f"conformer_block_{layer_idx + 1:02d}_ln/scale"
    _ParamMapping[
      f"encoder.layers.{layer_idx}.final_layer_norm.bias"
    ] = f"conformer_block_{layer_idx + 1:02d}_ln/bias"


_add_params()


def map_param_func_v2(reader, name: str, var: rf.Parameter) -> numpy.ndarray:
  """map params, TF to RF"""
  from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
  from i6_experiments.users.schmitt.returnn_frontend.convert import params as convert_params_np

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

  if name == "s.ff_weight":
    value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
    value = convert_params_np.convert_tf_lstm_to_native_lstm_ff(value)
    assert value.shape == var.batch_shape, name
    assert value.dtype.name == var.dtype, name
    return value

  if name == "s.rec_weight":
    value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
    value = convert_params_np.convert_tf_lstm_to_native_lstm_rec(value)
    assert value.shape == var.batch_shape, name
    assert value.dtype.name == var.dtype, name
    return value

  if name == "s.bias":
    value = reader.get_tensor("output/rec/s/rec/lstm_cell/bias")
    value = convert_params_np.convert_tf_lstm_to_native_lstm_bias(value, forget_gate_bias=1.0)
    assert value.shape == var.batch_shape, name
    assert value.dtype.name == var.dtype, name
    return value

  if name == "s_length_model.ff_weight":
    value = reader.get_tensor("output/rec/s_length_model/rec/W")
    value = convert_params_np.convert_tf_rec_lstm_to_native_lstm_ff(value)
    assert value.shape == var.batch_shape, name
    assert value.dtype.name == var.dtype, name
    return value

  if name == "s_length_model.rec_weight":
    value = reader.get_tensor("output/rec/s_length_model/rec/W_re")
    value = convert_params_np.convert_tf_rec_lstm_to_native_lstm_rec(value)
    assert value.shape == var.batch_shape, name
    assert value.dtype.name == var.dtype, name
    return value

  if name == "s_length_model.bias":
    value = reader.get_tensor("output/rec/s_length_model/rec/b")
    value = convert_params_np.convert_tf_rec_lstm_to_native_lstm_bias(value, forget_gate_bias=0.0)
    assert value.shape == var.batch_shape, name
    assert value.dtype.name == var.dtype, name
    return value

  if ".conv_block.norm." in name:
    assert name.startswith("encoder.layers.")
    layer_idx = int(name.split(".")[2])
    value = convert_params_np.convert_tf_batch_norm_to_rf(
      reader=reader,
      rf_name=name,
      rf_prefix_name=f"encoder.layers.{layer_idx}.conv_block.norm.",
      tf_prefix_name=f"conformer_block_{layer_idx + 1:02d}_conv_mod_bn/batch_norm/",
      var=var,
    )
    assert value.shape == var.batch_shape, name
    assert value.dtype.name == var.dtype, name
    return value

  raise NotImplementedError(f"cannot map {name!r} {var}")
