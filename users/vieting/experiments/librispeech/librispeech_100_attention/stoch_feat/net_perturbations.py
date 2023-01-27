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
  """
  Apply piecewise linear VTLP to given center frequencies x.

  alpha_range specifies the range around 1.0 from which to sample alpha. E.g. alpha_rang=0.2 means, alpha is uniformly
  distributed between 0.9 and 1.1.

  probability denotes with which probability the perturbation is applied. E.g. probability=0.4 means, the perturbation
  is applied to 40% of the utterances.
  """
  import tensorflow as tf
  f_hi = 256 * 4800 / 8000  # 4800 Hz is taken from Hinton paper
  if x.shape.rank == 1:  # batch-level random factor
    alpha_shape = (1,)
  else:  # utterance-level random factor, so add batch dim
    alpha_shape = (tf.shape(x)[0], 1)
  alpha = (tf.random.uniform(alpha_shape) - 0.5) * alpha_range + 1
  alpha = tf.where(tf.random.uniform(alpha_shape) < probability, alpha, 1.0)  # only keep percentage of alphas != 1
  x_1 = alpha * x
  x_2 = 256 - (256 - f_hi * tf.minimum(alpha, 1)) / \
    (256 - f_hi * tf.minimum(alpha, 1) / alpha) * (256 - x)
  x = tf.where(alpha < 1, tf.maximum(x_1, x_2), tf.minimum(x_1, x_2))
  return x


def vtlp_bilinear(x, alpha_range, probability):
  """
  Apply bilinear VTLP to given center frequencies x.

  alpha_range specifies the range around 1.0 from which to sample alpha. E.g. alpha_rang=0.2 means, alpha is uniformly
  distributed between 0.9 and 1.1.

  probability denotes with which probability the perturbation is applied. E.g. probability=0.4 means, the perturbation
  is applied to 40% of the utterances.
  """
  import numpy as np
  import tensorflow as tf
  if x.shape.rank == 1:  # batch-level random factor
    alpha_shape = (1,)
  else:  # utterance-level random factor, so add batch dim
    alpha_shape = (tf.shape(x)[0], 1)
  alpha = (tf.random.uniform(alpha_shape) - 0.5) * alpha_range + 1
  alpha = tf.where(tf.random.uniform(alpha_shape) < probability, alpha, 1.0)  # only keep percentage of alphas != 1
  x = x * np.pi / 256
  x = x + 2 * tf.math.atan(((1 - alpha) * tf.math.sin(x)) / (1 - (1 - alpha) * tf.math.cos(x)))
  x = x / np.pi * 256
  return x


def add_center_freq_perturb_vtlp(net: Dict[str, Dict], warping: str, alpha_range: float, probability: float):
  """
  Add a center frequency perturbation using VTLP to a network.

  The warping can be piecewise_linear or bilinear, see
  Zhan, P., & Waibel, A. (1997). Vocal tract length normalization for large vocabulary continuous speech recognition.
  https://apps.dtic.mil/sti/pdfs/ADA333514.pdf

  alpha_range specifies the range around 1.0 from which to sample alpha. E.g. alpha_rang=0.2 means, alpha is uniformly
  distributed between 0.9 and 1.1.

  probability denotes with which probability the perturbation is applied. E.g. probability=0.4 means, the perturbation
  is applied to 40% of the utterances.
  """
  net = copy.deepcopy(net)
  subnet = net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]
  for layer in ["center_freqs_l_fft", "center_freqs_r_fft", "center_freqs_c_fft"]:
    subnet[layer + "_raw"] = copy.deepcopy(subnet[layer])
    subnet[layer] = {
      "class": "eval",
      "eval": (
        f"self.network.get_config().typed_value('vtlp_{warping}')"
        f"(source(0), alpha_range={alpha_range}, probability={probability})"),
      "from": layer + "_raw",
    }
  net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"] = subnet
  tf_func = {
    "piecewise_linear": vtlp_piecewise_linear,
    "bilinear": vtlp_bilinear,
  }
  return net, tf_func[warping]


def filter_width_perturb_const(x, min_noise, max_noise, probability):
  """
  Apply constant noise to all filter width. Constant in the sense that all filters are perturbed in the same way for a
  given utterance.

  min_noise is the minimum noise value. Negative numbers make the window narrower.
  For reference: with -1 * mel_scale(f_max) / (nr_of_filters + 1), only a very sharp peak at the center freq is left
  assuming f_min = 0. This would be -35 with standard settings of 80 filters and f_max=8kHz. With half of that, the
  filters just start not overlapping anymore.

  max_noise is the maximum noise value. Positive numbers make the window wider.
  For reference: with mel_scale(f_max) / (7 + 1) * 0.5, the filters start overlapping with one more filter assuming
  f_min = 0. This would be 17.5 with standard settings.

  probability denotes with which probability the perturbation is applied. E.g. probability=0.4 means, the perturbation
  is applied to 40% of the utterances.
  """
  import tensorflow as tf

  if x.shape.rank == 1:
    noise_shape = (1,)
  else:
    noise_shape = (tf.shape(x)[0], 1)
  noise = tf.random.uniform(noise_shape) * (max_noise - min_noise) + min_noise
  noise = tf.where(tf.random.uniform(noise_shape) < probability, noise, 0.0)  # only keep percentage of noise != 0
  noise = noise * tf.ones(tf.shape(x))
  return noise

def add_filter_width_perturb_const(
    net: Dict[str, Dict], min_noise: float, max_noise: float, level="utterance", probability=1.0,
) -> Dict[str, Dict]:
  """
  Perturb center frequencies by adding the same offset to all center frequencies in Mel domain.
  """
  net = copy.deepcopy(net)
  subnet = net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]
  subnet["filter_width_noise"] = {
    "class": "eval",
    "eval": (
      f"self.network.get_config().typed_value('filter_width_perturb_const')"
      f"(source(0), min_noise={min_noise}, max_noise={max_noise}, probability={probability})"),
    "from": "center_freqs_c",
  }
  subnet["center_freqs_l_clean"] = copy.deepcopy(subnet["center_freqs_l"])
  subnet["center_freqs_l"] = {
    "class": "combine", "kind": "sub", "from": ["center_freqs_l_clean", "filter_width_noise"]}
  subnet["center_freqs_r_clean"] = copy.deepcopy(subnet["center_freqs_r"])
  subnet["center_freqs_r"] = {
    "class": "combine", "kind": "add", "from": ["center_freqs_r_clean", "filter_width_noise"]}
  net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"] = subnet
  return net, filter_width_perturb_const

