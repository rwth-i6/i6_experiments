def get_mini_lstm_params_freeze_str():
  freeze_str = """
def _fix_layer(layer_dict):
  if layer_dict["class"] == "rec":
      unit = layer_dict["unit"]
      if isinstance(unit, dict):
          _fix_net(unit)
          return
  layer_dict["trainable"] = False
  if "dropout" in layer_dict:
      layer_dict["dropout"] = 0


def _fix_net(net_dict):
    for key, value in net_dict.items():
        _fix_layer(value)

_fix_net(network)

network["output"]["unit"]["att_lstm"]["trainable"] = True
network["output"]["unit"]["att"]["trainable"] = True
  """

  return freeze_str

def get_mini_lstm_params_freeze_str_w_drop():
  freeze_str = """
def _fix_layer(layer_dict):
  if layer_dict["class"] == "rec":
      unit = layer_dict["unit"]
      if isinstance(unit, dict):
          _fix_net(unit)
          return
  layer_dict["trainable"] = False

def _fix_net(net_dict):
    for key, value in net_dict.items():
        _fix_layer(value)

_fix_net(network)

network["output"]["unit"]["att_lstm"]["trainable"] = True
network["output"]["unit"]["att"]["trainable"] = True
  """

  return freeze_str


def get_mini_self_att_params_freeze_str_w_drop(layers):
  freeze_str = f"""
def _fix_layer(layer_dict):
  if layer_dict["class"] == "rec":
      unit = layer_dict["unit"]
      if isinstance(unit, dict):
          _fix_net(unit)
          return
  layer_dict["trainable"] = False

def _fix_net(net_dict):
    for key, value in net_dict.items():
        if key.startswith('ilm_'):
            continue
        _fix_layer(value)

_fix_net(network)
"""

  return freeze_str