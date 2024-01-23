from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from .data import build_bpe_training_datasets, TrainingDatasetSettings, get_text_lexicon
from ..data import build_test_dataset
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from ..pipeline import training, search, compute_prior

from .config import get_training_config, get_search_config, get_prior_config


def conformer_rnnt_espnet_like():
    """

    ESPNet like means BPE 500 and subsampling 4

    :return:
    """

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_rnnt/espnet_like"


    BPE_SIZE = 500

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=5,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_bpe_training_datasets(
        bpe_size=BPE_SIZE,
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev", "test"]:
    for testset in ["dev"]:
            test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
        )
    from i6_experiments.common.baselines.tedlium2.lm.ngram_config import run_tedlium2_ngram_lm
    lms_system = run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=False)
    lm = lms_system.interpolated_lms["dev-pruned"]["4gram"]
    arpa_ted_lm = lm.ngram_lm

    # ---------------------------------------------------------------------------------------------------------------- #

    def run_exp(ft_name, datasets, train_args, search_args=None, num_epochs=250, decoder="rnnt.decoder.experimental_rnnt_decoder", with_prior=False, evaluate_epoch=None):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)
        _, _, search_jobs = search(ft_name + "/default_%i" % evaluate_epoch, returnn_search_config,
                                   train_job.out_checkpoints[evaluate_epoch], test_dataset_tuples, RETURNN_EXE,
                                   MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False))

        return train_job, search_jobs
    
    from ..pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, PredictorConfig

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=256,
        activation=None,
    )
    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        emebdding_dropout=0.2,
        num_lstm_layers=1,
        lstm_hidden_dim=256,
        lstm_dropout=0.1,
    )
    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        predictor_config=predictor_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=256,
        num_layers=12,
        num_heads=4,
        ff_dim=1024,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=10,
        joiner_dim=320,
        joiner_activation="tanh",
        joiner_dropout=0.1,
    )
    
    train_args_adamw03_24gb_jjlr = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
        np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 200 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
        "debug": True,
    }

    train_args = {
        **copy.deepcopy(train_args_adamw03_24gb_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7_transparent",
        "net_args": {"model_config_dict": asdict(model_config)},
    }

    search_args_gpu = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
        "use_gpu": True,  # also for new hash
    }
    train_job, _  = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_transparent_JJLR_sub4_small_bs200ac2/bs12_gpu",
        datasets=train_data, train_args=train_args, search_args=search_args_gpu, with_prior=False)
    train_job.rqmt["gpu_mem"] = 24
    
    model_config_ff2048 = copy.deepcopy(model_config)
    model_config_ff2048.ff_dim = 2048
    train_args_ff2048 = {
        **copy.deepcopy(train_args_adamw03_24gb_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7_transparent",
        "net_args": {"model_config_dict": asdict(model_config_ff2048)},
    }
    train_job, _  = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_transparent_JJLR_sub4_small_bs200ac2_ff2048/bs12_gpu",
        datasets=train_data, train_args=train_args_ff2048, search_args=search_args_gpu, with_prior=False)
    train_job.rqmt["gpu_mem"] = 24
    
    
    train_args_bs300ac1 = copy.deepcopy(train_args)
    train_args_bs300ac1["config"]["batch_size"] = 300 * 16000
    train_args_bs300ac1["config"]["accum_grad_multiple_step"] = 1
    train_job, _  = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_transparent_JJLR_sub4_small_bs300ac1/bs12_gpu",
        datasets=train_data, train_args=train_args_bs300ac1, search_args=search_args_gpu, with_prior=False)
    train_job.rqmt["gpu_mem"] = 24
    
    
    # Do it large instead
    model_config_v5_enc384_dec512 = ModelConfig(
        frontend_config=copy.deepcopy(frontend_config),
        specaug_config=specaug_config,
        predictor_config=copy.deepcopy(predictor_config),
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=20,
        joiner_dim=512,
        joiner_activation="relu",
        joiner_dropout=0.1,
    )
    model_config_v5_enc384_dec512.predictor_config.lstm_hidden_dim = 512
    model_config_v5_enc384_dec512.predictor_config.lstm_dropout = 0.3
    model_config_v5_enc384_dec512.predictor_config.emebdding_dropout = 0.1
    model_config_v5_enc384_dec512.frontend_config.out_features = 384

    train_args = {
        **copy.deepcopy(train_args_adamw03_24gb_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7_transparent",
        "net_args": {"model_config_dict": asdict(model_config_v5_enc384_dec512)},
    }
    train_job, _  = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_transparent_JJLR_sub4_enc384_dec512/bs12_gpu",
        datasets=train_data, train_args=train_args, search_args=search_args_gpu, with_prior=False)
    train_job.rqmt["gpu_mem"] = 24
    

