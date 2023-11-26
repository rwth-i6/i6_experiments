__all__ = ["get_learning_rates"]

import copy

from i6_core.returnn import ReturnnConfig

def get_learning_rates(lrate=0.001, pretrain=0, warmup=False, warmup_ratio=0.1,
                       increase=90, inc_min_ratio=0.01, inc_max_ratio=0.3, const_lr=0,
                       decay=90, dec_min_ratio=0.01, dec_max_ratio=0.3, exp_decay=False, reset=False):
    #example fine tuning: get_learning_rates(lrate=5e-5, increase=0, constLR=150, decay=60, decMinRatio=0.1, decMaxRatio=1)
    #example for n epochs get_learning_rates(increase=n/2, cdecay=n/2)
    learning_rates = []
    # pretrain (optional warmup)
    if warmup:
        learning_rates += warmup_lrates(initial=lrate * warmup_ratio, final=lrate, epochs=pretrain)
    else:
        learning_rates += [lrate] * pretrain
    # linear increase and/or const
    if increase > 0:
        step = lrate * (inc_max_ratio - inc_min_ratio) / increase
        for i in range(1, increase + 1):
            learning_rates += [lrate * inc_min_ratio + step * i]
        if const_lr > 0:
            learning_rates += [lrate * inc_max_ratio] * const_lr
    elif const_lr > 0:
        learning_rates += [lrate] * const_lr
    # linear decay
    if decay > 0:
        if exp_decay:  # expotential decay (closer to newBob)
            import numpy as np
            factor = np.exp(np.log(dec_min_ratio / dec_max_ratio) / decay)
            for i in range(1, decay + 1):
                learning_rates += [lrate * dec_max_ratio * (factor ** i)]
        else:
            step = lrate * (dec_max_ratio - dec_min_ratio) / decay
            for i in range(1, decay + 1):
                learning_rates += [lrate * dec_max_ratio - step * i]
    # reset and default newBob(cv) afterwards
    if reset: learning_rates += [lrate]
    return learning_rates

def noam(n, warmup_n, model_d):
    """
    Noam style learning rate scheduling

    (k is identical to the global learning rate)

    :param int|float|tf.Tensor n:
    :param int|float|tf.Tensor warmup_n:
    :param int|float|tf.Tensor model_d:
    :return:
    """
    from returnn.tf.compat import v1 as tf
    model_d = tf.cast(model_d, tf.float32)
    n = tf.cast(n, tf.float32)
    warmup_n = tf.cast(warmup_n, tf.float32)
    return tf.pow(model_d, -0.5) * tf.minimum(tf.pow(n, -0.5), n * tf.pow(warmup_n, -1.5))

def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    """
    :param TFNetwork network:
    :param tf.Tensor global_train_step:
    :param tf.Tensor learning_rate: current global learning rate
    :param kwargs:
    :return:
    """
    return learning_rate * noam(n=global_train_step, warmup_n=noam_warmup_steps, model_d=noam_model_dim)

_bad_patterns = [
    "newbob", "learning_rates", "learning_rate_control_"
]

def configure_noam(returnn_config, model_dim=1536, warmup=25_000, lr=1, recognition=False):
    assert all(
        infix not in key
        for infix in _bad_patterns
        for key in returnn_config.config.keys()
    ), "Bad pattern among config keys: {}".format(list(returnn_config.config.keys()))
    assert returnn_config.config["learning_rate_control"] == "constant"
    res = copy.deepcopy(returnn_config)
    res.config["noam_warmup_steps"] = warmup
    res.config["noam_model_dim"] = model_dim
    # assert res.python_prolog is res.python_prolog_hash
    # res.python_prolog.extend([noam, dynamic_learning_rate]) # should extend python_prolog_hash, as well since they point to same object
    # return ReturnnConfig
    return ReturnnConfig(
        config=res.config,
        post_config=res.post_config,
        python_prolog=list(res.python_prolog) + [noam, dynamic_learning_rate],
        python_epilog=res.python_epilog,
        hash_full_python_code=True,
    )
