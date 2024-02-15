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


def run_ctc_bpe_synthetic_data():
    prefix_name = "experiments/jaist_project/asr/ls100_ctc_bpe/"

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
    dev_dataset_tuples = {}
    for devset in ["dev-clean", "dev-other"]:
            dev_dataset_tuples[devset] = build_test_dataset(
                dataset_key=devset,
                preemphasis=train_settings.preemphasis,
                peak_normalization=train_settings.peak_normalization,
            )
    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                preemphasis=train_settings.preemphasis,
                peak_normalization=train_settings.peak_normalization,
            )

    arpa_4gram_lm = get_4gram_binary_lm()

    # ---------------------------------------------------------------------------------------------------------------- #

    def run_exp(ft_name, datasets, train_args, search_args=None, with_prior=False, num_epochs=250, decoder="ctc.decoder.flashlight_bpe_ctc", eval_mode="dev"):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if with_prior:
            returnn_config = get_prior_config(training_datasets=datasets, **train_args)
            prior_file = compute_prior(
                ft_name,
                returnn_config,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(training_name + "/prior.txt", prior_file)
            search_args["prior_file"] = prior_file

        returnn_search_config = get_forward_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)

        assert eval_mode in ["dev", "test"]
        dataset_tuples = dev_dataset_tuples if eval_mode == "dev" else test_dataset_tuples
        _, _, search_jobs = search(ft_name + "/last_%i" % num_epochs, returnn_search_config,
                                   train_job.out_checkpoints[num_epochs], dataset_tuples, RETURNN_EXE,
                                   MINI_RETURNN_ROOT)

        return train_job, search_jobs
    
    
    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )
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
    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
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
        specauc_start_epoch=11,
    )
    model_config_resume = copy.deepcopy(model_config)
    model_config_resume.specauc_start_epoch = 1

    train_args_adamw03_accum2_jjlr = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
                np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 2 * 16000,  # no grad accum needed for JAIST Kagayaki
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
        },
        "debug": False,
    }

    default_search_args = {
        "lexicon": get_text_lexicon(librispeech_key="train-clean-100", bpe_size=BPE_SIZE),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 1024,
        "beam_size_token": 128,
        "arpa_lm": arpa_4gram_lm,
        "beam_threshold": 14,
    }

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config)},
    }
    train_args["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job, _ = run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_start11_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

            # test
            if lm_weight == 2.2 and prior_scale == 0.3:
                train_job, _ = run_exp(
                    prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_start11_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                        lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True, eval_mode="test")
            
    # resume training
    train_args_resume = copy.deepcopy(train_args)
    train_args_resume["net_args"] = {"model_config_dict": asdict(model_config_resume)}
    train_args_resume["config"]["import_model_train_epoch1"] = train_job.out_checkpoints[250]
    
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_start11_amp16_resume/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_resume, search_args=search_args, with_prior=True)
            
    # resume with test synthetic data
    syn_name = "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.3_syn_train-clean-360"
    syn_ogg_zip = synthetic_ogg_zip_data[syn_name]
    syn_train_data = build_bpe_training_datasets(
        librispeech_key="train-clean-100",
        bpe_size=BPE_SIZE,
        settings=train_settings_syn_training,
        real_data_weight=3,
        extra_zips=[syn_ogg_zip],
    )
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_start11_amp16_resume_syn/{syn_name}/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=syn_train_data, train_args=train_args_resume, search_args=search_args, with_prior=True)
