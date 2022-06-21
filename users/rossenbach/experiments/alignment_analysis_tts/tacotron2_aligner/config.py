from i6_core.returnn.config import ReturnnConfig

from .tacotron2_network import Tacotron2NetworkBuilderV2

post_config_template = {
    "cleanup_old_models": True,
    "use_tensorflow": True,
    "tf_log_memory_usage": True,
    "stop_on_nonfinite_train_score": False,
    "log_batch_size": True,
    "debug_print_layer_output_template": True,
    "cache_size": "0",
}


def get_training_config(network_options, train_dataset, cv_dataset, extern_data, do_eval=False):
    """

    :param dict[str, Any] network_options:
    :param datasets.GenericDataset train_dataset:
    :param datasets.GenericDataset cv_dataset:
    :param dict[str, dict[str, Any]] extern_data:
    :return:
    :rtype: ReturnnConfig
    """
    config = {
        "behavior_version": 1,
        ############
        'optimizer': {'class': 'adam', 'epsilon': 1e-8},
        "accum_grad_multiple_step": 2,
        "gradient_clip": 1,
        "gradient_noise": 0,
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rates": [0.001],
        "use_learning_rate_control_always": True,
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        "batch_size": 16000,  # number of audio frames
        "max_seq_length": {'data': 1290},  # 3*0.0125*430 seconds
        "max_seqs": 200,
        #############
        "train": train_dataset.as_returnn_opts(),
        "dev": cv_dataset.as_returnn_opts(),
        "extern_data": extern_data,
    }

    if do_eval is True:
        builder = Tacotron2NetworkBuilderV2(network_options=network_options)
        config['network'] = builder.create_network()
        config = builder.add_decoding(config, dump_attention=True)
        config = ReturnnConfig(config=config, post_config=post_config_template.copy())
    else:
        staged_network_dict = {}
        for idx in range(5):
            postnet_loss_scale = max(min((idx/5*0.25), 0.25), 0.01)
            stop_token_loss_scale = min(idx/5, 1.0)
            builder = Tacotron2NetworkBuilderV2(
                network_options=network_options,
                stop_token_loss_scale=stop_token_loss_scale,
                postnet_loss_scale=postnet_loss_scale
            )
            staged_network_dict[idx*5 + 1] = builder.create_network()
        config = ReturnnConfig(config=config, post_config=post_config_template.copy(), staged_network_dict=staged_network_dict)

    return config