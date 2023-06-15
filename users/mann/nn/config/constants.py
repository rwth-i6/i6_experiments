from ..learning_rates import get_learning_rates

BASE_CRNN_CONFIG = {
    # "num_input" : 50,
    "lr" : 0.00025,
    "batch_size" : 5000,
    "max_seqs" : 200,
    "nadam" : True,
    "gradient_clip" : 0,
    "learning_rate_control" : "newbob_multi_epoch",
    "learning_rate_control_error_measure" : 'dev_score_output',
    "min_learning_rate" : 1e-6,
    "update_on_device" : True,
    "cache_size" : "0",
    "batching" : "random",
    "chunking" : "50:25",
    "truncation" : -1,
    "gradient_noise" : 0.1,
    "learning_rate_control_relative_error_relative_lr" : True,
    "learning_rate_control_min_num_epochs_per_new_lr" : 4,
    "newbob_learning_rate_decay" : 0.9,
    "newbob_multi_num_epochs" : 8,
    "newbob_multi_update_interval" : 4,
    "optimizer_epsilon" : 1e-08,
    "use_tensorflow" : True,
    "multiprocessing" : True,
    # "cleanup_old_models" : {'keep': epochs}
}

BASE_NETWORK_CONFIG = {
    "l2" : 0.01,
    "dropout" : 0.1,
}

BASE_LSTM_CONFIG = {
    "layers": 6 * [512],
}

BASE_FFNN_LAYERS = {
    "layers": 6 * [2048],
    "feature_window": 15,
    "activation": "relu",
}

BASE_VITERBI_LRS = {
    "lr": 0.0008,
    "learning_rates": [ 
        0.0003,
        0.0003555555555555555,
        0.0004111111111111111,
        0.00046666666666666666,
        0.0005222222222222222,
        0.0005777777777777778,
        0.0006333333333333334,
        0.0006888888888888888,
        0.0007444444444444445,
        0.0008
    ]
}

BASE_BW_LRS = {
    "lr": 0.00025,
    "learning_rates": get_learning_rates(increase=70, decay=70)
}

TINA_UPDATES_1K = {
    "batch_size": 10000,
    "gradient_noise": 0,
    "learning_rate_control_min_num_epochs_per_new_lr": 3,
    "max_seqs": 128,
    "nadam": True,
    "newbob_multi_num_epochs" : 20,
    "newbob_multi_update_interval" : 1,
    "learning_rates": get_learning_rates(increase=225, decay=225)
}

TINA_NETWORK_CONFIG = {
    "l2": 0.0001,
}

TINA_UPDATES_SWB = {
    "batch_size": 10000,
    "gradient_noise": 0,
    "learning_rate_control_min_num_epochs_per_new_lr": 3,
    "max_seqs": 128,
    "nadam": True,
    "newbob_multi_num_epochs" : 6,
    "newbob_multi_update_interval" : 1,
    "learning_rates": get_learning_rates(increase=120, decay=120)
}

