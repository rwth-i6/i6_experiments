# def load_params_v2(*, name, shape, reader):
#   import numpy
#
#   if name == "output/rec/s_wo_c/rec/W":
#     in_dim = 512 + 640
#     old_w_ff, _ = numpy.split(reader.get_tensor("output/rec/s/rec/lstm_cell/kernel"), [in_dim], axis=0)
#     return old_w_ff[:640]
#
#   return None

def load_missing_params(name, shape, preload_model_state):
  import returnn.frontend as rf
  import math
  import torch

  def _get_glorot_ff_init(shape):
    fan_in, fan_out = shape[0], shape[1]
    scale = 1.0 / max(1.0, (fan_in + fan_out) / 2.0)
    limit = math.sqrt(3.0 * scale)
    return torch.empty(
      shape,
      dtype=torch.float32,
      device=rf.get_default_device(),
    ).uniform_(-limit, limit)

  if name == "label_decoder.s_wo_att.ff_weight":
    return preload_model_state["label_decoder.s.ff_weight"][:, :640]
  elif name.startswith("label_decoder.s_wo_att"):
    return preload_model_state[name.replace("s_wo_att", "s")]
  elif name in ("label_decoder.target_embed_reset_eos.weight", "label_decoder.output_prob_reset_eos.weight"):
      return _get_glorot_ff_init(shape)
  elif name == "label_decoder.output_prob_reset_eos.bias":
      return torch.zeros(shape, dtype=torch.float32, device=rf.get_default_device())

  return None
