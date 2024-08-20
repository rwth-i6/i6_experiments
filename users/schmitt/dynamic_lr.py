# one cycle LR: triangular linear w.r.t. iterations(steps)
dynamic_lr_str = """
def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
      initialLR  = {initial_lr}
      peakLR     = {peak_lr}
      finalLR    = {final_lr}
      cycleEpoch = {cycle_ep}
      totalEpoch = {total_ep}
      nStep      = {n_step}

      steps     = cycleEpoch * nStep
      stepSize  = (peakLR - initialLR) / steps
      steps2    = (totalEpoch - 2 * cycleEpoch) * nStep
      stepSize2 = (initialLR - finalLR) / steps2

      import tensorflow as tf
      n = tf.cast(global_train_step, tf.float32)
      return tf.where(global_train_step <= steps, initialLR + stepSize * n,
                 tf.where(global_train_step <= 2*steps, peakLR - stepSize * (n - steps), 
                     tf.maximum(initialLR - stepSize2 * (n - 2*steps), finalLR)))
"""


def dyn_lr_lin_warmup_invsqrt_decay(*, global_train_step: int, learning_rate: float, **_kwargs) -> float:
    """
    Linear warmup, inv sqrt decay
    """
    from returnn.config import get_global_config

    config = get_global_config()

    warmup_steps = config.int("learning_rate_warmup_steps", None)
    assert warmup_steps and warmup_steps > 0, f"set learning_rate_warmup_steps in config >0, got {warmup_steps}"

    # In ESPnet, inv_norm_factor = warmup_steps, but we allow it to be configured separately.
    norm = config.float("learning_rate_invsqrt_norm", None)
    assert norm and norm > 0, f"set learning_rate_invsqrt_norm in config >0, got {norm}"

    i = global_train_step + 1
    if i <= warmup_steps:
        return learning_rate * (i / warmup_steps)  # linear warmup
    base = 1.0 + (i - warmup_steps) / norm
    return learning_rate * base**-0.5


def dyn_lr_piecewise_linear(*, global_train_step: int, learning_rate: float, **_kwargs) -> float:
  """
  Piecewise linear
  """
  from returnn.config import get_global_config

  config = get_global_config()

  steps = config.int_list("learning_rate_piecewise_steps")
  lrs = config.float_list("learning_rate_piecewise_values")
  assert len(steps) + 1 == len(lrs)

  last_step = 0
  for i, step in enumerate(steps):
    assert step > last_step
    assert global_train_step >= last_step
    if global_train_step < step:
      factor = (global_train_step + 1 - last_step) / (step - last_step)
      return learning_rate * (lrs[i + 1] * factor + lrs[i] * (1 - factor))
    last_step = step

  return learning_rate * lrs[-1]


def plot_dyn_lr_piecewise_linear():
  import matplotlib.pyplot as plt
  import numpy as np

  def _dyn_lr_piecewise_linear(global_train_step: int, learning_rate: float, steps, lrs) -> float:
    last_step = 0
    for i, step in enumerate(steps):
      assert step > last_step
      assert global_train_step >= last_step
      if global_train_step < step:
        factor = (global_train_step + 1 - last_step) / (step - last_step)
        return learning_rate * (lrs[i + 1] * factor + lrs[i] * (1 - factor))
      last_step = step

    return learning_rate * lrs[-1]

  steps = [295_000, 590_000, 652_000]
  steps = [600000, 900000, 982000]
  peak_lr = 1e-3
  lrs = [peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3]

  effective_lrs = []
  for train_step in range(982000):
    effective_lrs.append(_dyn_lr_piecewise_linear(train_step, 1.0, steps, lrs))
  plt.plot(effective_lrs)
  plt.show()
  print(effective_lrs)


if __name__ == "__main__":
  plot_dyn_lr_piecewise_linear()
