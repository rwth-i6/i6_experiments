

class NetworkDict:
  _network = {}
  _trainable = True

  def get_network(self):
    return self._network

  def get_as_subnetwork(self, source=None):
    source = source or ["data"]
    return {
      "class": "subnetwork",
      "from": source,
      "subnetwork": self._network,
      "trainable": self._trainable}


class PreemphasisNetwork(NetworkDict):
  def __init__(self, alpha, output_name="output"):
    """
    Network for preemphasis of audio signal
    """
    self._network = {
      "shift_0": {"axis": "T", "class": "slice", "slice_end": -1},
      "shift_0_mul": {"class": "eval", "from": "shift_0", "eval": f"source(0) * {alpha}"},
      "shift_1_raw": {"axis": "T", "class": "slice", "slice_start": 1},
      "shift_1": {
        "class": "reinterpret_data",
        "enforce_batch_major": True,
        "from": ["shift_1_raw"],
        "set_axes": {"T": "time"},
        "size_base": "shift_0",
      },
      output_name: {"class": "combine", "from": ["shift_1", "shift_0_mul"], "kind": "sub"},
    }


class LogMelNetwork(NetworkDict):
  def __init__(
          self, output_dims=80, wave_norm=False, frame_size=400, frame_shift=160, fft_size=512, norm=True,
          output_name="output", **kwargs
  ):
    """
    Log Mel filterbank feature network dict as used by Mohammad
    """
    self._network = {}
    if wave_norm:
      self._network["wave_norm"] = {"class": "norm", "axes": "T", "from": "data"}
    self._network.update({
      "stft": {
        "class": "stft",
        "frame_shift": frame_shift,
        "frame_size": frame_size,
        "fft_size": fft_size,
        "from": "wave_norm" if wave_norm else "data",
      },
      "abs": {
        "class": "activation",
        "from": "stft",
        "activation": "abs",
      },
      "power": {
        "class": "eval",
        "from": "abs",
        "eval": "source(0) ** 2",
      },
      "mel_filterbank": {
        "class": "mel_filterbank",
        "from": "power",
        "fft_size": fft_size,
        "nr_of_filters": output_dims,
        "n_out": output_dims,
      },
      "log": {
        "from": "mel_filterbank",
        "class": "activation",
        "activation": "safe_log",
        "opts": {"eps": 1e-10},
      },
      "log10": {
        "from": "log",
        "class": "eval",
        "eval": "source(0) / 2.3026"
      },
      output_name: {
        "class": "copy",
        "from": "log10",
      }
    })
    if norm:
      self._network[output_name] = {
        "class": "batch_norm",
        "from": "log10",
        "momentum": 0.01,
        "epsilon": 0.001,
        "update_sample_only_in_training": True,
        "delay_sample_update": True,
      }


class GammatoneNetwork(NetworkDict):
  """
  Wrapper class for subnetwork that extracts gammatone features
  """
  def __init__(
    self, output_dim=50, sample_rate=16000, freq_max=7500., freq_min=100.,
    gt_filterbank_size=0.04, gt_filterbank_padding="valid",
    temporal_integration_size=0.025, temporal_integration_strides=None, temporal_integration_padding="valid",
    normalization="time", trainable=False, **kwargs
  ):
    """
    :param int output_dim: gammatone feature output dimension
    :param int|float sample_rate: sampling rate of input waveform
    :param int gt_filterbank_size: size of gammatone filterbank (in seconds)
    :param str|int gt_filterbank_padding: padding for gammatone filterbank
    :param int temporal_integration_size: size of filter for temporal integration (in seconds)
    :param int|None temporal_integration_strides: strides of filter for temporal integration
    :param str temporal_integration_padding: padding for temporal integration
    :param str normalization: type of normalization.
      time: normalize mean and variance over time dim only
    :param bool auto_use_channel_first: use corresponding option in ConvLayer to speed up computations
    :param bool trainable: if True, parameters can be updated during training
    :param kwargs:
    """
    assert gt_filterbank_padding in ["valid", "same"] or isinstance(gt_filterbank_padding, int), "Unknown padding"
    assert temporal_integration_padding in ["valid", "same"], "Unknown padding"
    assert freq_max < sample_rate / 2, "Likely a misconfiguration"
    temporal_integration_size = int(temporal_integration_size * sample_rate)
    temporal_integration_strides = temporal_integration_strides or sample_rate // 100

    self._trainable = trainable
    self._network = {
      "gammatone_filterbank": {
        "activation": "abs",
        "class": "conv",
        "filter_size": (int(gt_filterbank_size * sample_rate),),
        "forward_weights_init": {
          "class": "GammatoneFilterbankInitializer",
          "num_channels": output_dim,
          "length": gt_filterbank_size,
          "sample_rate": sample_rate,
          "freq_max": freq_max,
          "freq_min": freq_min,
        },
        "n_out": output_dim,
        "padding": gt_filterbank_padding,
        "trainable": self._trainable,
        },
      "gammatone_filterbank_split": {
        "axis": "F",
        "class": "split_dims",
        "dims": (-1, 1),
        "from": ["gammatone_filterbank"]},
      "temporal_integration": {
        "class": "conv",
        "filter_size": (temporal_integration_size, 1),
        "forward_weights_init": "numpy.hanning({}).reshape(({}, 1, 1, 1))".format(
          temporal_integration_size, temporal_integration_size),
        "from": ["gammatone_filterbank_split"],
        "n_out": 1,
        "padding": "valid",
        "strides": (temporal_integration_strides, 1),
        "trainable": self._trainable},
      "temporal_integration_merge": {
        "axes": "except_time",
        "class": "merge_dims",
        "from": ["temporal_integration"]},
      "compression": {
        "class": "eval",
        "eval": "tf.pow(source(0) + 1e-06, 0.1)",
        "from": ["temporal_integration_merge"]},
      "dct": {"class": "dct", "from": ["compression"]},
    }

    source = "dct"
    if normalization == "time":
      self._network["output"] = {
          "class": "norm", "axes": "T", "trainable": self._trainable, "from": source}
    else:
      raise NotImplementedError


class ScfNetwork(NetworkDict):
  def __init__(
    self, num_tf=150, size_tf=256, stride_tf=10, activation_tf=None, num_env=5, size_env=40, stride_env=16,
    activation_env=None, normalization_env="layer", padding="valid", wave_norm=False, **kwargs
  ):
    """
    Network which applies conv layers to the raw waveform and pools using multi resolutional learned filters similar to
    Z. Tüske, R. Schlüter, and H. Ney.
    Acoustic modeling of Speech Waveform based on Multi-Resolution, Neural Network Signal Processing.
    ICASSP 2018
    https://www-i6.informatik.rwth-aachen.de/publications/download/1097/Tueske-ICASSP-2018.pdf

    Was also used and referred to as supervised convolutional features (SCF) in
    Peter Vieting, Christoph Lüscher, Wilfried Michel, Ralf Schlüter, Hermann Ney
    ON ARCHITECTURES AND TRAINING FOR RAW WAVEFORM FEATURE EXTRACTION IN ASR
    ASRU 2021
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9688123&tag=1

    :param int num_tf: number of filters for time frequency decomposition
    :param int size_tf: size of filters for time frequency decomposition
    :param int stride_tf: stride of filters for time frequency decomposition (160t'' = 10t' = t in paper)
    :param dict[str, dict] activation_tf: activation after time frequency decomposition (f_1 in paper)
    :param int num_env: number of filters for envelope extraction (max(i) in paper)
    :param int size_env: size of filters for envelope extraction (N_ENV in paper)
    :param int stride_env: stride of filters for envelope extraction (160t'' = 10t' = t in paper)
    :param dict[str, dict] activation_env: activation after envelope extraction (f_2 in paper)
    :param Optional[str] normalization_env: normalization applied after envelope extraction, e.g. 'batch' or 'layer'
    :param str padding: padding to use for convolutions ('valid' (default) or 'same')
    :param kwargs: arguments passed to parent class `NetworkDict`
    """
    activation_tf = activation_tf or {"abs": {}}
    activation_env = activation_env or {"abs": {}, "root": {}}

    self._network = {}
    if wave_norm:
      self._network["wave_norm"] = {"class": "norm", "axes": "T", "from": "data"}
    self._network.update({
      "conv_h_filter": {
        "class": "variable",
        "shape": (size_tf, 1, num_tf),
        "init": "glorot_uniform",
      },
      "conv_h": {
        "class": "conv",
        "filter_size": (size_tf,),
        "strides": stride_tf,
        "n_out": num_tf,
        "padding": padding,
        "filter": "conv_h_filter",
        "from": "wave_norm" if wave_norm else "data"},
      "conv_h_split": {
        "class": "split_dims",
        "axis": "F",
        "dims": (-1, 1),
        "from": "conv_h_act"},
      "conv_l": {
        "class": "conv",
        "filter_size": (size_env, 1),
        "strides": (stride_env, 1),
        "n_out": num_env,
        "padding": padding,
        "from": "conv_h_split"},
      "conv_l_merge": {
        "class": "merge_dims",
        "axes": "except_time",
        "from": "conv_l"},
      "output": {"class": "copy", "from": "conv_l_act"}
    })

    self.add_activation_layer("conv_h_act", "conv_h", activation_tf)
    self.add_activation_layer("conv_l_act", ["conv_l_merge"], activation_env)
    self.add_normalization_to_layer("conv_l_act", normalization_env)

  def add_activation_layer(self, name, source_layers, activation):
    """
    adds a layer for activations, corresponding to f_1 and f_2 in the paper
    :param str name: layer name
    :param list|str source_layers: input to the layer
    :param dict[str, dict] activation: keys of the dict specify the functions applied and values are dicts of arguments
      for that key, e.g. {'abs': {}, 'root': {'nth-root': 2.5, 'eps': 1e-5}} for absolute value plus 2.5th root with an
      epsilon of 1e-5 to prevent numerical instabilities
    """
    expr = None
    if activation.get("abs", None) is not None:
      expr = "tf.abs(source(0))"
    if activation.get("root", None) is not None:
      eps = " + {}".format(activation["root"].get("eps", 1e-5)) if activation["root"].get("eps", True) else ""
      expr = "tf.pow({}{}, 1 / {})".format(expr, eps, activation["root"].get("nth-root", 2.5))
    assert expr is not None, "activation '{}' could not be parsed successfully".format(activation)
    self._network[name] = {"class": "eval", "eval": expr, "from": source_layers}

  def add_normalization_to_layer(self, layer_name, norm):
    """
    adds normalization to a layer
    :param str layer_name: name of the layer which should be normalized
    :param Optional[str] norm: normalization type, e.g. 'batch' or 'layer'
    """
    if norm is None:
      pass
    elif norm == "batch":
      self._network[layer_name + "_no_norm"] = self._network[layer_name].copy()
      self._network[layer_name] = {
        "class": "batch_norm",
        "from": layer_name + "_no_norm",
        "momentum": 0.01,
        "epsilon": 0.001,
        "update_sample_only_in_training": True,
        "delay_sample_update": True,
      }
    elif norm == "layer":
      self._network[layer_name + "_no_norm"] = self._network[layer_name].copy()
      self._network[layer_name] = {"class": "layer_norm", "from": [layer_name + "_no_norm"]}
    elif norm in ["T", "F", "TF"]:
      self._network[layer_name + "_no_norm"] = self._network[layer_name].copy()
      self._network[layer_name] = {"class": "norm", "axes": norm, "from": layer_name + "_no_norm"}
    else:
      raise NotImplementedError
