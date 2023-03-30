"""
Implementation of MixUp data augmentation method

An investigation of mixup training strategies for acoustic models in ASR
(https://www.isca-speech.org/archive/pdfs/interspeech_2018/medennikov18_interspeech.pdf)
"""


def mixup(data, network, **kwargs):
  """
  Weighted addition of randomly sampled data and labels for regularization (adding noise) and data augmentation.
  We use uniform distribution to select the lambda mixing weights.
  We allow the possible of applying multiple mixes also (max_min=1 is same as original paper).

    x = lambda * x + (1 - lambda) * x'   where x' is randomly sampled

  if len(x') < len(x), then we randomly sample the start position for addition to avoid always adding to the beginning
  and add more variability of having mixed regions. If len(x') > len(x), then all x will be mixed and we just add them.
  For labels, length does not matter and thus we just do weighted addition.
  """
  from returnn.tf.compat import v1 as tf

  x = data.placeholder

  min_lambda = kwargs['min_lambda']
  max_lambda = kwargs['max_lambda']
  max_mix = kwargs['max_mix']  # 1 is default
  prob_ratio = kwargs['prob_ratio']  # how much samples to mix from the batch

  batch_size = tf.shape(x)[data.batch_dim_axis]
  time_dim = tf.shape(x)[data.time_dim_axis]

  mix_lambda = tf.random_uniform(
    shape=(batch_size,), minval=min_lambda, maxval=max_lambda + 1)

  # TODO: randomly select data samples based on some probability ratio

  #