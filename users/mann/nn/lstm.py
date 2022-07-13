from .configs import blstm_config
from .networks import blstm_network

BASE_CRNN_CONFIG = {
    # "num_input" : 50,
    "l2" : 0.01,
    "lr" : 0.00025,
    "dropout" : 0.1,
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

BASE_LSTM_CONFIG = {
    "layers": 6 * [512],
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

def blstm_network_helper(layers, dropout, l2, **_ignored):
    return blstm_network(layers, dropout, l2)

def viterbi_lstm(num_input, epochs, **kwargs):
    kwargs = {**BASE_CRNN_CONFIG, **BASE_VITERBI_LRS, **kwargs}
    network_kwargs = kwargs.copy()
    lr = kwargs.pop("lr")
    del kwargs["dropout"], kwargs["l2"]
    config = blstm_config(
        num_input,
        network = blstm_network_helper(**BASE_LSTM_CONFIG, **network_kwargs),
        learning_rate=lr,
        **kwargs
    )
    config_ = config.config
    del config_['max_seq_length'], config_['adam']
    config_['network']['output']['loss_opts'] = {"focal_loss_factor": 2.0}
    config_["network"]["output"]["loss"] = "ce"
    return config
