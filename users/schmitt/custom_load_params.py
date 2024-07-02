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
  if name == "label_decoder.s_wo_att.ff_weight":
    return preload_model_state["label_decoder.s.ff_weight"][:, :640]
  elif name.startswith("label_decoder.s_wo_att"):
    return preload_model_state[name.replace("s_wo_att", "s")]

  return None
