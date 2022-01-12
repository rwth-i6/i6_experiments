import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .prototype_pipeline import \
    build_training_datasets, build_test_dataset, training, search
from .specaugment_clean import \
    SpecAugmentSettings


def trainig_network_mohammad(specaug_settings=None, fix_regularization=False):
    """
    Network derived from Mohammads Librispeech-100h system with new encoder pre-training

    :return:
    """
    from .prototype_network import EncoderWrapper, static_decoder

    specaug_settings_full = specaug_settings or SpecAugmentSettings()

    specaug_settings_light = SpecAugmentSettings(
        min_frame_masks=0,
        mask_each_n_frames=100,
        max_frames_per_mask=10,
        min_feature_masks=0,
        max_feature_masks=2,
        max_features_per_mask=4,
    )

    stage_nets = []
    for i in range(6):
        # pooling pretraing
        network_stage = 0 if i == 0 else i - 1  # the first two networks are the same
        if network_stage == 0:
            lstm_pool_sizes = [6]
        else:
            lstm_pool_sizes = [3, 2]
        if fix_regularization:
            # dropout from 0.15 to 0.3
            lstm_dropout = 0.15 + (i/5.0 * 0.15)
            # l2 from 0.0005 to 0.001
            l2 = 0.0005 + (i/5.0 * 0.0005)
        else:
            # dropout from 0 to 0.3
            lstm_dropout = (i/5.0 * 0.3)
            l2 = (i/5.0 * 0.001)
        # grow lstm dim from 512 to 1024
        lstm_dim = int(512 + (network_stage/4.0 * 512))
        #enable specaugment after epoch 10

        encoder = EncoderWrapper(audio_feature_key="audio_features", target_label_key="bpe_labels",
                                 num_lstm_layers=2 + network_stage, lstm_pool_sizes=lstm_pool_sizes, lstm_dropout=lstm_dropout,
                                 lstm_single_dim=lstm_dim, l2=l2,
                                 specaugment_settings=specaug_settings_light if i < 3 else specaug_settings_full)
        encoder_dict = encoder.make_root_net_dict()
        # Eval layer hack for specaugment
        encoder_dict['encoder']['subnetwork']['specaug_block']['subnetwork']['eval_layer'].pop("kind")

        local_static_decoder = copy.deepcopy(static_decoder)
        if i < 5:
            local_static_decoder["output"]["unit"]["output_prob"]["loss_opts"]["label_smoothing"] = 0

        # the network is the combination of the dynamic encoder and the legacy static decoder
        stage_net = {**encoder_dict, **local_static_decoder, "#copy_param_mode": "subset"}

        # the "pretraining" phase has a larger batch size
        if i < 5:
            stage_net["#config"] = {}
            stage_net["#config"]["batch_size"] = 15000

        stage_nets.append(stage_net)

    network_dict = {
        1: stage_nets[0], # network 0
        11: stage_nets[1], # still network 0, only specaug
        26: stage_nets[2], # network 1
        31: stage_nets[3], # network 2
        36: stage_nets[4], # network 3
        41: stage_nets[5], # network 4
    }
    return network_dict


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
        # 'truncation': -1
    }

    network_dict = trainig_network_mohammad(**kwargs)

    config["extern_data"] = training_datasets.extern_data
    config["train"] = training_datasets.train.as_returnn_opts()
    config["dev"] = training_datasets.cv.as_returnn_opts()

    from .specaugment_clean import get_funcs
    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        staged_network_dict=network_dict,
        python_prolog=get_funcs(),
        hash_full_python_code=True,
    )
    return returnn_config


def test():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="e799984d109029c42816e6d5e0b5361b8bd7f05d").out_repository
    returnn_root_search = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="58ff79074d4d9580f79b41b66b5ceceea892dd16").out_repository

    prefix_name = "experiments/librispeech/librispeech_100_attention/lstm_encdec_2021/prototype_pipelines"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets = build_training_datasets(returnn_exe, returnn_root, prefix_name, bpe_size=2000)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root, output_path=prefix_name)

    # use default specaugment
    specaug_settings = SpecAugmentSettings()

    # Initial experiment
    exp_prefix = prefix_name + "/test_enc_only_fixed_specaug"
    returnn_config = get_config(training_datasets, specaug_settings=specaug_settings)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root)
    search(exp_prefix + "/default", returnn_config, train_job, test_dataset_tuples, returnn_exe, returnn_root_search)

    # Initial experiment
    exp_prefix = prefix_name + "/test_enc_only_fixed_regularizaion"
    returnn_config = get_config(training_datasets, specaug_settings=specaug_settings, fix_regularization=True)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root)
    search(exp_prefix + "/default", returnn_config, train_job, test_dataset_tuples, returnn_exe, returnn_root_search)

