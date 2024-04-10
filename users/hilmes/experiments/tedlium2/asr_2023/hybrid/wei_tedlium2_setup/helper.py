from sisyphus import tk
import copy


from .nn_setup import build_encoder_network, add_output_layer, add_specaug_source_layer, get_spec_augment_mask_python


default_nn_config_args = {  # batching #
    "batch_size": 10000,
    "max_seqs": 128,
    "chunking": "64:32",  # better than 50:25
    "batching": "random",
    "min_seq_len": {"classes": 1},
    "min_seq_length": {"classes": 1},
    # optimization #
    #'nadam': True,
    "learning_rate": 0.0009,
    "gradient_clip": 0,
    "gradient_noise": 0.1,  # together with l2 and dropout for overfit
    # Note: (default 1e-8) likely not too much impact
    #'optimizer_epsilon': 1e-8,
    "optimizer": {"class": "nadam", "epsilon": 1e-08},
    # let it stop and adjust in time
    # Note: for inf or nan, sth. is too big (e.g. lr warm up)
    # 'stop_on_nonfinite_train_score' : False,
    "learning_rate_control": "newbob_multi_epoch",
    "newbob_multi_num_epochs": 5,
    "newbob_multi_update_interval": 1,
    "newbob_learning_rate_decay": 0.9,
    # 'newbob_relative_error_threshold' : -0.02, # default -0.01
    # 'min_learning_rate' : 1e-5 #
    "learning_rate_control_relative_error_relative_lr": True,
    "learning_rate_control_min_num_epochs_per_new_lr": 3,
    # pretraining #
    # better initialization and convergence (benefit momentum ?)
    "pretrain": "default",
    # default #
    "start_epoch": "auto",
    "start_batch": "auto",
    "use_tensorflow": True,
    "update_on_device": True,
    "multiprocessing": True,
    "cache_size": "0",
    "truncation": -1,
    "window": 1,
    # clean up done by sisyphus: probably better leave to RETURNN #
    #'cleanup_old_models': False
}


def get_network(num_layers=6, layer_size=512, spec_augment=False, **kwargs):

    lstm_args = {
        "num_layers": num_layers,
        "size": layer_size,
        "l2": 0.01,
        "dropout": 0.1,
        "bidirectional": True,
        "unit": "nativelstm2",
    }
    network, from_list = build_encoder_network(**lstm_args)

    output_args = {
        "loss": "ce",
        "loss_opts": {  # less weight on loss of easy samples (larger p)
            "focal_loss_factor": 2.0,
        },
    }

    network = add_output_layer(network, from_list, **output_args)

    if spec_augment:
        network, from_list2 = add_specaug_source_layer(network)

    return copy.deepcopy(network)


def make_nn_config(network, nn_config_args=default_nn_config_args, **kwargs):

    nn_config = copy.deepcopy(nn_config_args)
    nn_config["network"] = network

    # common training settings
    optimizer = kwargs.pop("optimizer", None)
    if optimizer is not None and not optimizer == "nadam":
        del nn_config["nadam"]
        nn_config[optimizer] = True
    if kwargs.pop("no_pretrain", False):
        del nn_config["pretrain"]
    if kwargs.pop("no_chunking", False):
        del nn_config["chunking"]
    # Note: whatever left !
    nn_config.update(kwargs)
    return nn_config
