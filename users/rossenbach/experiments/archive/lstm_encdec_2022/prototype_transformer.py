import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream

from .prototype_pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint, TrainingDatasets
from .specaugment import \
    SpecAugmentSettings

from .get_transformer_network import get_network


def trainig_network_mohammad(datastreams, specaug_settings=None, fix_regularization=False, fix_act=False):
    """
    Network derived from Mohammads Librispeech-100h system with new encoder pre-training

    :param dict[str, Datastream] datastreams
    :return:
    """
    from returnn_common import nn

    audio_features, (in_time_dim, in_feature_dim) = datastreams["audio_features"].as_returnn_common_data_and_dims(name="audio_features")
    bpe_labels, (out_time_dim, out_label_dim) = datastreams["bpe_labels"].as_returnn_common_data_and_dims(name="bpe_labels")

    data = {
        "audio_features": audio_features,
        "bpe_labels": bpe_labels,
    }

    dim_tags_proxy = nn.ReturnnDimTagsProxy()

    import time

    specaug_settings_full = specaug_settings or SpecAugmentSettings()

    specaug_settings_light = SpecAugmentSettings(
        min_frame_masks=0,
        max_mask_each_n_frames=100,
        max_frames_per_mask=10,
        min_feature_masks=0,
        max_feature_masks=2,
        max_features_per_mask=4,
    )

    start = time.time()
    stage_nets = []
    base_string = None
    for i in range(5):
        local_start = time.time()
        network_string, base_string = get_network(dim_tags_proxy, audio_features, bpe_labels, time_dim=in_time_dim, feature_dim=in_feature_dim, label_dim=out_label_dim, label_time_dim=out_time_dim)
        stage_nets.append(network_string)
        print("local_time: %f" % (time.time() - local_start))

    network_dict = {
        1: stage_nets[0], # network 0
    }

    print("total_time: %f" % (time.time() - start))

    return network_dict, base_string


def get_config(
        training_datasets,
        **kwargs):
    """

    :param prefix_name:
    :param TrainingDatasets training_datasets:
    :param returnn_exe:
    :param returnn_root:
    :param kwargs:
    :return:
    """

    # changing these does not change the hash
    post_config = {
        'use_tensorflow': True,
        'tf_log_memory_usage': True,
        'cleanup_old_models': True,
        'log_batch_size': True,
        'debug_print_layer_output_template': True,
    }

    wup_start_lr = 0.0003
    initial_lr = 0.0008

    learning_rates = [wup_start_lr] * 10 + list(numpy.linspace(wup_start_lr, initial_lr, num=10))

    config = {
        'gradient_clip': 0,
        'optimizer': {'class': 'Adam', 'epsilon': 1e-8},
        'accum_grad_multiple_step': 2,
        'gradient_noise': 0.0,
        'learning_rates': learning_rates,
        'min_learning_rate': 0.00001,
        'learning_rate_control': "newbob_multi_epoch",
        'learning_rate_control_relative_error_relative_lr': True,
        'learning_rate_control_min_num_epochs_per_new_lr': 3,
        'use_learning_rate_control_always': True,
        'newbob_multi_num_epochs': 3,
        'newbob_multi_update_interval': 1,
        'newbob_learning_rate_decay': 0.9,
        'batch_size': 10000,
        'max_seqs': 200,
    }

    network_dict, base_string = trainig_network_mohammad(training_datasets.datastreams, **kwargs)

    #config["extern_data"] = extern_data
    config["train"] = training_datasets.train.as_returnn_opts()
    config["dev"] = training_datasets.cv.as_returnn_opts()
    config["eval_datasets"] =  {'devtrain': training_datasets.cv.as_returnn_opts()}

    from .specaugment import get_funcs
    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        staged_network_dict=network_dict,
        python_prolog=get_funcs() + [base_string],
        hash_full_python_code=True,
    )
    return returnn_config


def transformer():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root_datasets = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                  commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="0c045ec027bd32ce279a91632a8c758d1900d0dd").out_repository

    prefix_name = "experiments/librispeech/librispeech_100_attention/lstm_encdec_2022/transformer"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root_datasets, output_path=prefix_name)

    # use default specaugment
    specaug_settings = SpecAugmentSettings()

    # Initial experiment
    exp_prefix = prefix_name + "/test_baseline"
    returnn_config = get_config(training_datasets, specaug_settings=specaug_settings)
    train_job = training(exp_prefix, returnn_config, num_epochs=250, returnn_exe=returnn_exe, returnn_root=returnn_root)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root)
    search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root)


