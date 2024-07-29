
def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    """
    One cycle learning rate: triangular linear w.r.t. iterations (steps)
    """
    from returnn.config import get_global_config
    config = get_global_config()
    initial_lr = config.typed_dict.get("learning_rate_initial_lr", 8e-5)
    peak_lr = config.typed_dict.get("learning_rate_peak_lr", 8e-4)
    final_lr = config.typed_dict.get("learning_rate_final_lr", 1e-6)
    cycle_epoch = config.typed_dict.get("learning_rate_cycle_epoch", 135)
    total_epoch = config.typed_dict.get("learning_rate_total_epoch", 300)
    n_step = config.typed_dict.get("learning_rate_n_step", 2650)
    # -- derived -- #
    steps = cycle_epoch * n_step
    step_size = (peak_lr - initial_lr) / steps
    steps2 = (total_epoch - 2 * cycle_epoch) * n_step
    step_size2 = (initial_lr - final_lr) / steps2

    import tensorflow as tf

    n = tf.cast(global_train_step, tf.float32)
    return tf.where(
        global_train_step <= steps,
        initial_lr + step_size * n,
        tf.where(
            global_train_step <= 2 * steps,
            peak_lr - step_size * (n - steps),
            tf.maximum(initial_lr - step_size2 * (n - 2 * steps), final_lr),
        ),
    )