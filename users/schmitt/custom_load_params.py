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
  elif name.startswith("label_decoder.s_wo_att."):
    return preload_model_state[name.replace("s_wo_att", "s")]
  elif name in ("label_decoder.target_embed_reset_eos.weight", "label_decoder.output_prob_reset_eos.weight"):
      # TODO: this is wrong since it randomly inits the whole matrix instead of only the EOS part
      return _get_glorot_ff_init(shape)
  elif name == "label_decoder.output_prob_reset_eos.bias":
      # TODO: this is wrong since it randomly inits the whole vector instead of only the EOS part
      return torch.zeros(shape, dtype=torch.float32, device=rf.get_default_device())
  elif name.startswith("label_decoder.readout_in_w_current_frame"):
    if name == "label_decoder.readout_in_w_current_frame.weight":
      init = _get_glorot_ff_init(shape)
      # overwrite the first 2176 rows with the pretrained weights
      # the remaining rows are the new weights which did not exist in the pretrained model
      init[:2176] = preload_model_state["label_decoder.readout_in.weight"]
      return init
    else:
      assert name == "label_decoder.readout_in_w_current_frame.bias"
      return preload_model_state["label_decoder.readout_in.bias"]

  return None
