def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    """
    Adjustment of one cycle learning rate for full-sum fine-tuning:
    Half a cycle with constant _lr, half a cycle decay, final decay
    """
    # -- need to be adjusted w.r.t. training -- #
    const_lr = 8e-5
    decay_lr = 1e-5
    final_lr = 1e-6
    cycle_epoch = 110
    total_epoch = 240
    n_step = 7165  # steps/epoch depending on batch_size

    # -- derived -- #
    steps = cycle_epoch * n_step
    step_size = (const_lr - decay_lr) / steps
    steps2 = (total_epoch - 2 * cycle_epoch) * n_step
    step_size2 = (decay_lr - final_lr) / steps2

    import tensorflow as tf
    n = tf.cast(global_train_step, tf.float32)
    return tf.where(
        global_train_step <= steps,
        const_lr,
        tf.where(
            global_train_step <= 2 * steps,
            const_lr - step_size * (n - steps),
            tf.maximum(decay_lr - step_size2 * (n - 2*steps), final_lr),
        ),
    )

