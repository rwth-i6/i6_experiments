def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    """
    One cycle learning rate: triangular linear w.r.t. iterations (steps)
    """
    # -- need to be adjusted w.r.t. training -- #
    initial_lr = 8e-5
    peak_lr = 8e-4
    final_lr = 1e-6
    cycle_epoch = 135
    total_epoch = 300
    n_step = 2680  # steps/epoch depending on batch_size

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