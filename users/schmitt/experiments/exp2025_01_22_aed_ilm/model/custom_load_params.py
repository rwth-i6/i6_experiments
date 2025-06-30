def load_missing_params(name, shape, preload_model_state):
  import returnn.frontend as rf
  import math
  import torch

  if "mini_att" in name:
    return None
  if name.startswith("decoder.enc_ctx") or name.startswith("decoder.inv_fertility"):
    return preload_model_state[f"encoder.{name[len('decoder.'):]}"]
  if name.startswith("decoder"):
    return preload_model_state[f"label_{name}"]

  return None
