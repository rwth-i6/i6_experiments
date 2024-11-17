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


def get_base_returnn_dict_v2(debug=False):

    params = get_base_returnn_dict(debug=debug)
    params["gradient_clip"] = 0.0

    return params


def get_base_returnn_dict_v3(debug=False):

    params = get_base_returnn_dict(debug=debug)
    params["gradient_clip"] = 0.0
    params["batching"] = "random"

    return params


def get_base_returnn_dict_zhou(debug=False):
    base_params = get_base_returnn_dict(debug=debug)
    base_params["optimizer"] = {"class": "nadam", "epsilon": 1e-08}
    del base_params["optimizer_epsilon"]

    params = {**base_params, "max_seqs": 128, "batching": "random"}

    return params


def get_base_returnn_dict_zhou_v2(debug=False):

    params = get_base_returnn_dict(debug=debug)
    params["gradient_clip"] = 20.0
    params["batching"] = "random"

    return params


def get_base_returnn_dict_smbr(debug=False):

    params = get_base_returnn_dict(debug=debug)
    params["optimizer"] = {"class": "nadam", "epsilon": 1e-08}
    params["gradient_clip"] = 20.0
    params["batching"] = "random"

    return params
