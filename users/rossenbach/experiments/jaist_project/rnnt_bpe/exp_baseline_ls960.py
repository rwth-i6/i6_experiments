from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast


from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from ..lm import get_4gram_binary_lm
from ..data.bpe import build_bpe_training_datasets, TrainingDatasetSettings, get_text_lexicon
from ..data.common import build_test_dataset
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT, KENLM_BINARY_PATH

from ..pipeline import training, search, compute_prior

from ..config import get_training_config, get_forward_config, get_prior_config
from ..storage import ctc_models

def conformer_rnnt_ls960():
    prefix_name = "experiments/jaist_project/standalone_2024/rnnt_bpe/"

    BPE_SIZE = 5000

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=10,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_bpe_training_datasets(
        librispeech_key="train-other-960",
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
        },
        "debug": True,
    }
    
    #### New experiments with corrected FF-Dim

    from ..pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7_cfg import \
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
        pool1_kernel_size=(3, 1),
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )
    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        emebdding_dropout=0.2,
        num_lstm_layers=1,
        lstm_hidden_dim=512,
        lstm_dropout=0.1,
    )
    model_config_v5_sub6_512lstm = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        predictor_config=predictor_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=20,
        joiner_dim=640,
        joiner_activation="relu",
        joiner_dropout=0.1,
    )
    model_config_v5_sub6_512lstm_start1 = copy.deepcopy(model_config_v5_sub6_512lstm)
    model_config_v5_sub6_512lstm_start1.specauc_start_epoch = 1


    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
    }
    train_args["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_amp16/bs12",
        datasets=train_data, train_args=train_args, search_args=search_args)

    
    
    train_args_continue_ctc = copy.deepcopy(train_args)
    train_args_continue_ctc["config"]["batch_size"] = 120 * 16000
    train_args_continue_ctc["net_args"] = {"model_config_dict": asdict(model_config_v5_sub6_512lstm_start1)}
    train_args_continue_ctc["config"]["preload_from_files"] = {
        "encoder": {
            "filename": ctc_models["bpe5k_i6modelsLV1_LRv2_sub6_ep50"],
            "init_for_train": True,
            "ignore_missing": True,
        }
    }
    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start1_accum2_lstm512_amp16_init_from_ctc_50eps/bs12",
        datasets=train_data, train_args=train_args_continue_ctc, search_args=search_args)
    # train_job.hold()


    train_args_continue_ctc_LR5 = copy.deepcopy(train_args_continue_ctc)
    train_args_continue_ctc_LR5["config"]["learning_rates"] =  list(np.linspace(5e-5, 5e-4, 120)) + list(
        np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10))
    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start1_accum2_lstm512_LR5_amp16_init_from_ctc_50eps/bs12",
        datasets=train_data, train_args=train_args_continue_ctc_LR5, search_args=search_args)

    search_args_greedy = copy.deepcopy(search_args)
    search_args_greedy["beam_size"] = 1
    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start1_accum2_lstm512_LR5_amp16_init_from_ctc_50eps/bs1",
        datasets=train_data, train_args=train_args_continue_ctc_LR5, search_args=search_args_greedy)
    
    # continue even further
    train_args_continue_ctc_LR2 = copy.deepcopy(train_args_continue_ctc)
    train_args_continue_ctc_LR2["config"]["learning_rates"] =  list(np.linspace(5e-5, 2e-4,20)) + list(
        np.linspace(2e-4, 5e-5, 220)) + list(np.linspace(5e-5, 1e-7, 10))
    train_args_continue_ctc_LR2["config"]["preload_from_files"] = {
        "encoder": {
            "filename": train_job.out_checkpoints[250],
            "init_for_train": True,
            "ignore_missing": True,
        }
    }
    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start1_accum2_lstm512_LR5_amp16_init_from_ctc_50eps_rnnt_25eps/bs12",
        datasets=train_data, train_args=train_args_continue_ctc_LR2, search_args=search_args)

    train_job.hold()


    # train_job.hold()

    # Sub-4 from here
    

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
        out_features=512,
        activation=None,
    )

    model_config_v5_sub4_512lstm = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        predictor_config=predictor_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=20,
        joiner_dim=640,
        joiner_activation="relu",
        joiner_dropout=0.1,
    )


    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v7",
        "net_args": {"model_config_dict": asdict(model_config_v5_sub4_512lstm)},
    }
    train_args["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}
    train_args["config"]["batch_size"] = 100 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    search_args = {
        "beam_size": 12,
        "returnn_vocab": label_datastream.vocab,
    }
    # Had NaN in training
    # train_job, _ = run_exp(
    #     prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub4_start20_lstm512_amp16/bs12",
    #     datasets=train_data, train_args=train_args, search_args=search_args)
    #  train_job.hold()


    train_args_continue_ctc = copy.deepcopy(train_args)
    train_args_continue_ctc["config"]["preload_from_files"] = {
        "encoder": {
            "filename": ctc_models["bpe5k_i6modelsLV1_LRv2"],
            "init_for_train": True,
            "ignore_missing": True,
        }
    }
    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub4_start20_lstm512_amp16_init_from_ctc/bs12",
        datasets=train_data, train_args=train_args_continue_ctc, search_args=search_args)
    # train_job.hold()


    # with aux loss
    from ..pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v8_cfg import ModelConfig
    model_config_v8_sub4_512lstm = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        predictor_config=predictor_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=20,
        joiner_dim=640,
        joiner_activation="relu",
        joiner_dropout=0.1,
        ctc_output_loss=0.3,
    )


    train_args_ctc_aux = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v8",
        "net_args": {"model_config_dict": asdict(model_config_v8_sub4_512lstm)},
    }
    train_args_ctc_aux["config"]["torch_amp_options"] = {"dtype": "bfloat16"}
    train_args_ctc_aux["config"]["batch_size"] = 100 * 16000
    train_args_ctc_aux["config"]["accum_grad_multiple_step"] = 3
    train_args_ctc_aux["config"]["preload_from_files"] = {
        "encoder": {
            "filename": ctc_models["bpe5k_i6modelsLV1_LRv2"],
            "init_for_train": True,
            "ignore_missing": True,
            "var_name_mapping": {
                "encoder_ctc.weight": "final_linear.weight",
                "encoder_ctc.bias": "final_linear.bias"
            }
        }
    }
    train_job, _ = run_exp(
        prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v8_JJLR_sub4_start20_lstm512_amp16_init_from_ctc/bs12",
        datasets=train_data, train_args=train_args_ctc_aux, search_args=search_args)
    # train_job.hold()
