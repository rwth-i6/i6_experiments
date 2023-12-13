__all__ = ["configure_normalized_bw"]

from i6_core.returnn import ReturnnConfig, CodeWrapper

def compute_normalized_fullsum(self, source, **kwargs):
    import tensorflow as tf
    gammas = source(1)
    lps = source(0)
    lp_bm = source(0, enforce_batch_major=True)

    # compute gradients
    grad_z = lps / tf.reduce_sum(lps, axis=-1, keepdims=True)
    grads = grad_z - gammas
    grad_before_activation = grads * (1 - lps)

    # compute loss
    numerator_loss = source(1, as_layer=True).output_loss
    denominator_loss = tf.reduce_sum(
        tf.math.log(tf.reduce_sum(lp_bm, axis=-1)), axis=1
    )
    self.output_loss = numerator_loss - denominator_loss

    return grad_before_activation

def configure_normalized_bw(config, recog=False):
    net = config.config["network"]
    extra_prolog = []
    output_layer_pattern = "aux_output_block_{}_ce"
    layer_idxs = [4, 8]
    if not recog:
        for layer in [
            output_layer_pattern.format(i) for i in layer_idxs
        ] + ["output"]:
            assert net[layer]["class"] == "softmax", "Maybe something went wrong, better aborting..."
            net[layer].update({
                "class": "linear",
                "activation": "sigmoid",
            })
            net[f"{layer}_normalize_bw"] = {
                "class": "eval",
                "eval": CodeWrapper("compute_normalized_fullsum"),
                "from": [layer, "fast_bw"]
            }
            net[f"{layer}_bw"]["loss_opts"] = {
                "error_signal_layer": f"{layer}_normalize_bw",
                "loss_wrt_to_act_in": "sigmoid",
            }
        extra_prolog = [compute_normalized_fullsum]
    else:
        net["output"].update({
            "class": "linear",
            "activation": "tf.math.log_sigmoid(x) - tf.math.reduce_logsumexp(tf.math.log_sigmoid(x), axis=-1, keepdims=True)",
        })
    # print(config.config["use_tensorflow"])
    return ReturnnConfig(
        config=config.config,
        post_config=config.post_config,
        python_prolog=list(config.python_prolog) + extra_prolog,
        python_epilog=config.python_epilog,
        hash_full_python_code=True,
    )
