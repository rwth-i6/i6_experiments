import numpy as np
from sisyphus import tk
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob, ReturnnForwardJob
from .data import TTSTrainingDatasets


def get_config_dict(output_size: int = 512, extract_embedding: bool = False):
    """

    :param output_size:
    :param extract_embedding:
    :return:
    """

    return {
        "concat_reconstruct_btn": {"class": "copy", "from": ["tdnn_5_ff", "reconstruct_bnt"]},
        "output": {
            "class": "softmax",
            "dropout": 0.0,
            "from": "tdnn_7_ff",
            "loss": "ce",
            "loss_opts": {},
            "loss_scale": 1,
            "target": "speaker_label_notime",
        } if not extract_embedding else {"class": "copy", "from": "tdnn_7_ff"},
        "reconstruct_bnt": {
            "activation": "tanh",
            "class": "linear",
            "forward_weights_init": {
                "class": "VarianceScaling",
                "distribution": "uniform",
                "mode": "fan_in",
                "scale": 0.78,
            },
            "from": "tdnn_5_ff_dropout",
            "n_out": 80,
            "with_bias": True,
        },
        "reconstruct_output": {
            "activation": "tanh",
            "class": "linear",
            "forward_weights_init": {
                "class": "VarianceScaling",
                "distribution": "uniform",
                "mode": "fan_in",
                "scale": 0.78,
            },
            "from": "reconstruct_bnt",
            "loss": "mse",
            "loss_scale": 5,
            "n_out": 80,
            "target": "data:audio_features",
            "trainable": True,
            "with_bias": True,
        },
        "speaker_label_notime": {
            "axis": "T",
            "class": "squeeze",
            "from": ["data:speaker_labels"],
            "register_as_extern_data": "speaker_label_notime",
        },
        "tdnn_1": {
            "activation": "relu",
            "batch_norm": True,
            "class": "conv",
            "dilation_rate": 1,
            "dropout": 0.1,
            "filter_size": (3,),
            "from": "data:audio_features",
            "n_out": 512,
            "padding": "same",
            "strides": 1,
            "trainable": True,
            "with_bias": True,
        },
        "tdnn_2": {
            "activation": "relu",
            "batch_norm": True,
            "class": "conv",
            "dilation_rate": 2,
            "dropout": 0.1,
            "filter_size": (3,),
            "from": "tdnn_1",
            "n_out": 512,
            "padding": "same",
            "strides": 1,
            "trainable": True,
            "with_bias": True,
        },
        "tdnn_3": {
            "activation": "relu",
            "batch_norm": True,
            "class": "conv",
            "dilation_rate": 3,
            "dropout": 0.1,
            "filter_size": (3,),
            "from": "tdnn_2",
            "n_out": 512,
            "padding": "same",
            "strides": 1,
            "trainable": True,
            "with_bias": True,
        },
        "tdnn_4_ff": {
            "activation": "tanh",
            "class": "linear",
            "forward_weights_init": {
                "class": "VarianceScaling",
                "distribution": "uniform",
                "mode": "fan_in",
                "scale": 0.78,
            },
            "from": "tdnn_3",
            "n_out": 512,
            "trainable": True,
            "with_bias": True,
        },
        "tdnn_5_ff": {
            "activation": "tanh",
            "class": "linear",
            "forward_weights_init": {
                "class": "VarianceScaling",
                "distribution": "uniform",
                "mode": "fan_in",
                "scale": 0.78,
            },
            "from": "tdnn_4_ff",
            "n_out": 1500,
            "trainable": True,
            "with_bias": True,
        },
        "tdnn_5_ff_dropout": {"class": "copy", "dropout": None, "from": "tdnn_5_ff"},
        "tdnn_6": {"class": "copy", "from": ["tdnn_6_att_mu", "tdnn_6_att_delta"], "trainable": True},
        "tdnn_6_att_delta": {
            "class": "eval",
            "eval": "tf.math.sqrt(tf.clip_by_value(source(0)-source(1)*source(1), clip_value_min=1e-31, clip_value_max=1e7))",
            "from": ["tdnn_6_att_weighted_x_2_merged", "tdnn_6_att_mu"],
            "trainable": True,
        },
        "tdnn_6_att_e": {
            "activation": None,
            "class": "linear",
            "forward_weights_init": {
                "class": "VarianceScaling",
                "distribution": "uniform",
                "mode": "fan_in",
                "scale": 0.78,
            },
            "from": "tdnn_6_att_ff",
            "n_out": 1,
            "trainable": True,
            "with_bias": True,
        },
        "tdnn_6_att_ff": {
            "activation": "tanh",
            "class": "linear",
            "forward_weights_init": {
                "class": "VarianceScaling",
                "distribution": "uniform",
                "mode": "fan_in",
                "scale": 0.78,
            },
            "from": "concat_reconstruct_btn",
            "n_out": 384,
            "trainable": True,
            "with_bias": True,
        },
        "tdnn_6_att_mu": {
            "axes": "except_batch",
            "class": "merge_dims",
            "from": ["tdnn_6_att_weighted_x"],
            "trainable": True,
        },
        "tdnn_6_att_weighted_x": {
            "auto_squeeze": False,
            "base": "concat_reconstruct_btn",
            "class": "generic_attention",
            "trainable": True,
            "weights": "tdnn_6_att_weights",
        },
        "tdnn_6_att_weighted_x_2": {
            "auto_squeeze": False,
            "base": "tdnn_6_squared",
            "class": "generic_attention",
            "trainable": True,
            "weights": "tdnn_6_att_weights",
        },
        "tdnn_6_att_weighted_x_2_merged": {
            "axes": "except_batch",
            "class": "merge_dims",
            "from": ["tdnn_6_att_weighted_x_2"],
            "trainable": True,
        },
        "tdnn_6_att_weights": {"class": "softmax_over_spatial", "from": "tdnn_6_att_e", "trainable": True},
        "tdnn_6_avg": {
            "axes": "T",
            "class": "reduce",
            "from": "concat_reconstruct_btn",
            "mode": "avg",
            "trainable": True,
        },
        "tdnn_6_squared": {
            "class": "eval",
            "eval": "tf.math.square(source(0))",
            "from": ["concat_reconstruct_btn"],
            "trainable": True,
        },
        "tdnn_7_ff": {
            "activation": "tanh",
            "class": "linear",
            "forward_weights_init": {
                "class": "VarianceScaling",
                "distribution": "uniform",
                "mode": "fan_in",
                "scale": 0.78,
            },
            "from": "tdnn_6",
            "n_out": output_size,
            "trainable": True,
            "with_bias": True,
        },
    }


def get_training_config(training_datasets: TTSTrainingDatasets, output_size: int = 512):
    """

    :param training_datasets:
    :param output_size:
    :return:
    """

    post_config = {
        "cleanup_old_models": {"keep": []},
        "use_tensorflow": True,
        "tf_log_memory_usage": True,
        "stop_on_nonfinite_train_score": False,
        "log_batch_size": True,
        "debug_print_layer_output_template": True,
        "cache_size": "0",
    }

    config = {
        "optimizer": {"class": "nadam", "epsilon": 1e-8},
        "gradient_noise": 0.0,
        "learning_rate": 1e-05,
        "learning_rate_control": "constant",
        "learning_rates": [0.001],
        "min_learning_rate": 1e-05,
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        "batch_size": 5000,
        "batching": "sort_bin_shuffle:.64",
        "max_seq_length": {"audio_features": 1600},
        "max_seqs": 60,
        "use_spec_augment": False,
    }
    cycle_epoch = 300
    peak_lr = 2e-4
    lr_1 = peak_lr / 10
    lr_2 = peak_lr / 10
    final_lr = 1e-8
    total_epoch = cycle_epoch + 100
    learning_rates = (
        list(np.linspace(lr_1, peak_lr, num=cycle_epoch // 2))
        + list(np.linspace(peak_lr, lr_2, num=cycle_epoch // 2))
        + list(np.linspace(lr_2, final_lr, num=total_epoch - cycle_epoch))
    )
    config["learning_rates"] = learning_rates

    network = get_config_dict(output_size=output_size, extract_embedding=False)
    config["train"] = training_datasets.train.as_returnn_opts()
    config["dev"] = training_datasets.cv.as_returnn_opts()
    config["extern_data"] = {
        key: datastream.as_returnn_extern_data_opts() for key, datastream in training_datasets.datastreams.items()
    }
    config["network"] = network

    returnn_config = ReturnnConfig(config=config, post_config=post_config)

    return returnn_config


def get_forward_config(training_datasets: TTSTrainingDatasets, output_size: int = 512):

    post_config = {
        "use_tensorflow": True,
        "tf_log_memory_usage": True,
        "log_batch_size": True,
        "debug_print_layer_output_template": True,
        "cache_size": "0",
    }

    config = {
        "batch_size": 5000,
        "max_seqs": 60,
        "forward_use_search": True,
    }
    network = get_config_dict(output_size=output_size, extract_embedding=True)
    config["eval"] = training_datasets.train.as_returnn_opts()
    config["eval"]["datasets"]["audio"]["segment_file"] = None
    config["extern_data"] = {
        key: datastream.as_returnn_extern_data_opts(available_for_inference=True) for key, datastream in training_datasets.datastreams.items()
    }
    config["network"] = network

    returnn_config = ReturnnConfig(config=config, post_config=post_config)

    return returnn_config


def training(config, returnn_exe, returnn_root, prefix, num_epochs=600, mem=8):
    """

    :param config:
    :param returnn_exe:
    :param returnn_root:
    :param prefix:
    :param num_epochs:
    :return:
    """
    train_job = ReturnnTrainingJob(
        config,
        log_verbosity=5,
        num_epochs=num_epochs,
        time_rqmt=120,
        mem_rqmt=mem,
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    train_job.add_alias(prefix + "/training")
    tk.register_output(prefix + "/training.models", train_job.out_model_dir)

    return train_job


def forward(checkpoint, config, returnn_exe, returnn_root, prefix, hdf_outputs=None):
    if not hdf_outputs:
        hdf_outputs = []
    forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=hdf_outputs,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    forward_job.add_alias(prefix + "/forward")

    return forward_job
