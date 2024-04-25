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
