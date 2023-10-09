__all__ = ["get_base_returnn_dict"]


def get_base_returnn_dict(debug=False):
    debug_params = {
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
    }

    base = {
        "use_tensorflow": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.0,
    }

    if debug:
        base.update(**debug_params)

    return base
