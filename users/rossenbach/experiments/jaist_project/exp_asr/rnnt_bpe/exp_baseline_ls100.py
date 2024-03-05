from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast


from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.experiments.jaist_project.lm import get_4gram_binary_lm
from i6_experiments.users.rossenbach.experiments.jaist_project.data.bpe import build_bpe_training_datasets, TrainingDatasetSettings, get_text_lexicon
from i6_experiments.users.rossenbach.experiments.jaist_project.data.common import build_test_dataset
from i6_experiments.users.rossenbach.experiments.jaist_project.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT, KENLM_BINARY_PATH

from i6_experiments.users.rossenbach.experiments.jaist_project.pipeline import training, search, compute_prior

from i6_experiments.users.rossenbach.experiments.jaist_project.config import get_training_config, get_forward_config, get_prior_config
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import synthetic_ogg_zip_data


def conformer_rnnt_ls100():
    prefix_name = "experiments/jaist_project/asr/ls100_rnnt_bpe/"

    BPE_SIZE = 300

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )

    train_settings_syn_training = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=18,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_bpe_training_datasets(
        librispeech_key="train-clean-100",
        bpe_size=BPE_SIZE,
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev", "test"]:
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                preemphasis=train_settings.preemphasis,
                peak_normalization=train_settings.peak_normalization,
            )


    # ---------------------------------------------------------------------------------------------------------------- #


    def run_exp(ft_name, datasets, train_args, search_args=None, num_epochs=250, decoder="rnnt.decoder.experimental_rnnt_decoder"):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)


        returnn_search_config = get_forward_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)

        _, _, search_jobs = search(ft_name + "/last_%i" % num_epochs, returnn_search_config,
                                   train_job.out_checkpoints[num_epochs], test_dataset_tuples, RETURNN_EXE,
                                   MINI_RETURNN_ROOT)

        return train_job, search_jobs


    train_args_adamw03_accum2_jjlr = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
        np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 16000,  # no accum needed for JAIST
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
            "torch_amp_options":  {"dtype": "bfloat16"},
        },
        "debug": True,
    }
    
    #### New experiments with corrected FF-Dim

    from ...pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7_cfg import \
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
        out_features=384,
        activation=None,
    )
    predictor_config = PredictorConfig(
        symbol_embedding_dim=128,
        emebdding_dropout=0.2,
        num_lstm_layers=1,
        lstm_hidden_dim=256,
        lstm_dropout=0.2,
    )
    model_config_v5_sub6_512lstm = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        predictor_config=predictor_config,
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
        joiner_dim=320,
        joiner_activation="relu",
        joiner_dropout=0.1,
    )

    model_config_v5_sub6_512lstm_resume = copy.deepcopy(model_config_v5_sub6_512lstm)
    model_config_v5_sub6_512lstm_resume.specauc_start_epoch = 1

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
    }
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub4_start20_lstm256_amp16/bs12",
        datasets=train_data, train_args=train_args, search_args=search_args)

    # resume training
    train_args_resume = copy.deepcopy(train_args)
    train_args_resume["net_args"] = {"model_config_dict": asdict(model_config_v5_sub6_512lstm_resume)}
    train_args_resume["config"]["import_model_train_epoch1"] = train_job.out_checkpoints[250]

    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub4_start20_lstm256_amp16_continue/bs12",
        datasets=train_data, train_args=train_args_resume, search_args=search_args)

    # same with speed perturb
    train_args_speedpert = copy.deepcopy(train_args)
    train_args_speedpert["use_speed_perturbation"] = True

    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub4_start20_lstm256_amp16_speedpert/bs12",
        datasets=train_data, train_args=train_args_speedpert, search_args=search_args)
