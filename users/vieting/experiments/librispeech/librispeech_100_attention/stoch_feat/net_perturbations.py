"""
Contains helpers to add perturbations to the RETURNN network dicts for feature extraction.

Mostly, it is assumed that the input network has the structure of log10_net_10ms_v2.
"""
import copy
from typing import Dict
from i6_core.returnn.config import CodeWrapper

# dummies that avoid importing RETURNN here and that are written to the config via python prolog
batch_dim = CodeWrapper("batch_dim")
center_freqs_range_dim = CodeWrapper("center_freqs_range_dim")
range_batch_out_shape = CodeWrapper("{batch_dim, center_freqs_range_dim}")


def utterance_level_filterbanks(net: Dict[str, Dict]) -> Dict[str, Dict]:
  net = copy.deepcopy(net)
  subnet = net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]["center_freqs"]["subnetwork"]
  subnet["linear_range_raw"] = copy.deepcopy(subnet["linear_range"])
  subnet["batch_dummy"] = {"class": "constant", "value": 1.0, "with_batch_dim": True}
  subnet["linear_range"] = {
    "class": "combine",
    "kind": "mul",
    "from": ["linear_range_raw", "batch_dummy"],
    "out_shape": range_batch_out_shape,
  }
  net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]["center_freqs"]["subnetwork"] = subnet
  for layer in net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]:
    if "out_shape" in net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"][layer]:
      out_shape = net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"][layer]["out_shape"]
      if isinstance(out_shape, set):
        out_shape.add(batch_dim)
      elif isinstance(out_shape, CodeWrapper):
        out_shape = CodeWrapper(str(out_shape).replace("{", "{batch_dim, "))
      else:
        raise NotImplementedError
      net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"][layer]["out_shape"] = out_shape
  net["log_mel_features"]["subnetwork"]["mel_filterbank"]["var1"] = "T"
  return net


def add_center_freq_perturb_const(
    net: Dict[str, Dict], scale: float, level="utterance", probability=1.0
) -> Dict[str, Dict]:
  """
  Perturb center frequencies by adding the same offset to all center frequencies in Mel domain.
  """
  net = copy.deepcopy(net)
  subnet = net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]["center_freqs"]["subnetwork"]
  if probability == 1.0:
    eval_str = {
      "utterance": f"source(0) + {scale} * (tf.random.uniform((tf.shape(source(0))[0], 1)) - 0.5)",
      "batch": f"source(0) + {scale} * (tf.random.uniform((1,)) - 0.5)",
    }
  else:
    eval_str = {
      "utterance": (
        f"source(0) + "
        f"tf.where(tf.random.uniform((1,)) < {probability}, "
        f"{scale} * (tf.random.uniform((tf.shape(source(0))[0], 1)) - 0.5), 1)"),
      "batch": (
        f"source(0) + "
        f"tf.where(tf.random.uniform((1,)) < {probability}, {scale} * (tf.random.uniform((1,)) - 0.5), 1)"),
    }
  subnet["noisy_range"] = {
    "class": "eval",
    "eval": eval_str[level],
    "from": "linear_range",
  }
  subnet["output"]["from"] = "noisy_range"
  net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]["center_freqs"]["subnetwork"] = subnet
  return net


def vtlp_piecewise_linear(x, alpha_range, probability):
  import tensorflow as tf
  f_hi = 256 * 4800 / 8000  # 4800 Hz is taken from Hinton paper
  if x.shape.rank == 1:  # batch-level random factor
    alpha_shape = (1,)
  else:  # utterance-level random factor, so add batch dim
    alpha_shape = (tf.shape(x)[0], 1)
  alpha = (tf.random.uniform(alpha_shape) - 0.5) * alpha_range + 1
  x_1 = alpha * x
  x_2 = 256 - (256 - f_hi * tf.minimum(alpha, 1)) / \
    (256 - f_hi * tf.minimum(alpha, 1) / alpha) * (256 - x)
  x_vtlp = tf.where(alpha < 1, tf.maximum(x_1, x_2), tf.minimum(x_1, x_2))
  x = tf.where(tf.random.uniform((1,)) < probability, x_vtlp, x)
  return x


def add_center_freq_perturb_vtlp_piecewise_linear(net: Dict[str, Dict], alpha_range, probability):
  net = copy.deepcopy(net)
  subnet = net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]
  for layer in ["center_freqs_l_fft", "center_freqs_r_fft", "center_freqs_c_fft"]:
    subnet[layer + "_raw"] = copy.deepcopy(subnet[layer])
    subnet[layer] = {
      "class": "eval",
      # "eval": CodeWrapper(f"partial(vtlp_piecewise_linear, alpha_range={alpha_range}, probability={probability})"),
      "eval": (
        f"self.network.get_config().typed_value('vtlp_piecewise_linear')"
        f"(source(0), alpha_range={alpha_range}, probability={probability})"),
      "from": layer + "_raw",
    }
  net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"] = subnet
  return net, vtlp_piecewise_linear


def add_center_freq_perturb_vtlp_bilinear(net: Dict[str, Dict], alpha):
  def vtlp_bilinear(self, source, alpha):
    x = source(0) * np.pi / 256
    x = x + 2 * tf.math.atan(((1 - alpha) * tf.math.sin(x)) / (1 - (1 - alpha) * tf.math.cos(x)))
    x = x / np.pi * 256
    return x

  net = copy.deepcopy(net)
  # perturb after mel scale is applied
  subnet = net["mel_filterbank_weights"]["subnetwork"]
  for layer in ["center_freqs_l_fft", "center_freqs_r_fft", "center_freqs_c_fft"]:
    subnet[layer + "_raw"] = copy.deepcopy(subnet[layer])
    subnet[layer] = {
      "class": "eval",
      "eval": partial(vtlp_bilinear, alpha=alpha),
      "from": layer + "_raw",
    }
  net["mel_filterbank_weights"]["subnetwork"] = subnet
  return net
