def load_missing_params_aed(name, shape, preload_model_state):
  if name.startswith("decoder.enc_ctx") or name.startswith("decoder.inv_fertility"):
    return preload_model_state[f"encoder.{name[len('decoder.'):]}"]
  if name.startswith("decoder"):
    return preload_model_state[f"label_{name}"]

  return None
