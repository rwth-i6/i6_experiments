from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import add_ctc_model



def bpe_ls960_1023_low_vocab_test():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_bpe_low_vocab"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig



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
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )
    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Normal Style
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
        out_features=512,
        activation=None,
    )



    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 240)) + list(
            np.linspace(5e-4, 5e-5, 240)) + list(np.linspace(5e-5, 1e-7, 20)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    global_train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "debug": False,
    }

    def tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, base_decoder_config, lm_scales, prior_scales):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module="ctc.decoder.flashlight_ctc_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values, mode="minimize")
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name, forward_config={}, asr_model=asr_model, decoder_module="ctc.decoder.flashlight_ctc_v1",
                decoder_args={"config": asdict(decoder_config)}, test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn
            )

    def greedy_search_helper(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: GreedyDecoderConfig
        ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )

    for BPE_SIZE in [128, 256, 512, 1024]:

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=False,
        )
        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

        dev_dataset_tuples = {}
        for testset in ["dev-clean", "dev-other"]:
            dev_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings,
            )

        test_dataset_tuples = {}
        for testset in ["test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings,
            )

        model_config = ModelConfig(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config,
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
            specauc_start_epoch=11,  # BPE does not converge otherwise
        )

        default_decoder_config_bpe = DecoderConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=BPE_SIZE),
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=1024,
            beam_size_token=16,  # makes it much faster
            arpa_lm=arpa_4gram_lm,
            beam_threshold=14,
        )
        
        train_args = copy.deepcopy(global_train_args)
        train_args["net_args"] = {"model_config_dict": asdict(model_config)}

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_50eps"
        train_job = training(training_name, train_data_bpe, train_args, num_epochs=500, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe, get_specific_checkpoint=500
        )
        add_ctc_model(f"ls960_ctc_bpe_{BPE_SIZE}." + network_module + ".512dim_sub4_24gbgpu_50eps_ckpt500", asr_model)
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])

        # Same with conv first
        network_module_conv_first = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
        train_args_conv_first = {
            "config": train_config_24gbgpu_amp,
            "network_module": network_module_conv_first,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
        }

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module_conv_first + ".512dim_sub4_24gbgpu_50eps"
        train_job = training(training_name, train_data_bpe, train_args_conv_first, num_epochs=500,
                             **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_conv_first, with_prior=True, datasets=train_data_bpe,
            get_specific_checkpoint=500
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0],
                                 prior_scales=[0.2, 0.3, 0.4])

        if BPE_SIZE == 128:
            decoder_config_bpe = copy.deepcopy(default_decoder_config_bpe)
            decoder_config_bpe.lm_weight = 1.8
            decoder_config_bpe.prior_scale = 0.2
            #investiage effect of batch size
            for max_batch_size in [1, 10, 240]:
                forward_config = {"max_seqs": max_batch_size, "seed": 2}
                search_name = training_name + "/tune_batchsize/search_batch_size_%i" % (max_batch_size)
                search_jobs, wers = search(
                                     search_name,
                                     forward_config=forward_config,
                                     asr_model=asr_model,
                                     decoder_module="ctc.decoder.flashlight_ctc_v1",
                                     decoder_args={"config": asdict(decoder_config_bpe)},
                                     test_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                     **default_returnn
                                 )
                for search_job in search_jobs:
                    search_job.rqmt["sbatch_args"] = "-A rescale_speed -p rescale_amd"


            # THIS RUN WAS WITH CORRECT RESCALE SETTINGS
            decoder_config_bpe_rescale = copy.deepcopy(default_decoder_config_bpe)
            decoder_config_bpe_rescale.lm_weight = 2.0
            decoder_config_bpe_rescale.prior_scale = 0.2
            
            forward_config = {"max_seqs": 1, "seed": 4}
            search_name = training_name + "/rescale_accurate/lm2.0_scale0.2_bs1024_bst16_bth_14"
            search_jobs, wers = search(
                                 search_name,
                                 forward_config=forward_config,
                                 asr_model=asr_model,
                                 decoder_module="ctc.decoder.flashlight_ctc_v1_rescale_measure",
                                 decoder_args={"config": asdict(decoder_config_bpe_rescale)},
                                 test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                                 **default_returnn
                             )
            for search_job in search_jobs:
                search_job.rqmt["sbatch_args"] = "-A rescale_speed -p rescale_amd"



            # increase search time to match phoneme, all with lm 1.8 and prior 0.2
            # Running all in AMD RTF mode
            for beam_size in [256, 512, 1024]:
                for beam_size_token in [4, 8, 12, 16]:
                    for beam_threshold in [8, 10, 12, 14]:
                        config = copy.deepcopy(decoder_config_bpe)
                        config.beam_size = beam_size
                        config.beam_size_token = beam_size_token
                        config.beam_threshold = beam_threshold

                        search_name = training_name + "/cpu_fast_search/search_lm_1.8_prior_0.2_bs_%i_bst_%i_bth_%i" % (beam_size, beam_size_token, beam_threshold)
                        search_jobs, wers = search(
                                search_name,
                                forward_config={"max_seqs": 1,},
                                asr_model=asr_model,
                                decoder_module="ctc.decoder.flashlight_ctc_v1",
                                decoder_args={"config": asdict(config)},
                                test_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                **default_returnn
                            )
                        search_name = training_name + "/rescale_amd_search/search_lm_1.8_prior_0.2_bs_%i_bst_%i_bth_%i" % (beam_size, beam_size_token, beam_threshold)
                        search_jobs, wers = search(
                                search_name,
                                forward_config={"max_seqs": 1, "seed": 2},
                                asr_model=asr_model,
                                decoder_module="ctc.decoder.flashlight_ctc_v1",
                                decoder_args={"config": asdict(config)},
                                test_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                **default_returnn
                            )
                        for search_job in search_jobs:
                            search_job.rqmt["sbatch_args"] = "-A rescale_speed -p rescale_amd"

        if BPE_SIZE == 128 or BPE_SIZE == 512:
            # Extra long training for the BPE 128 one
            train_args_conv_first_ep100 = copy.deepcopy(train_args_conv_first)
            train_args_conv_first_ep100["config"]["learning_rates"] = list(np.linspace(7e-6, 5e-4, 240)) + list(
                np.linspace(5e-4, 5e-5, 720)) + list(np.linspace(5e-5, 1e-7, 40))
            train_args_conv_first_ep100["config"]["gradient_clip"] = 1.0

            train_args_conv_first_ep100_sp = copy.deepcopy(train_args_conv_first_ep100)
            train_args_conv_first_ep100_sp["use_speed_perturbation"] = True

            train_args_conv_first_ep100_sp_late_peak = copy.deepcopy(train_args_conv_first_ep100_sp)
            train_args_conv_first_ep100_sp_late_peak["config"]["learning_rates"] = list(np.linspace(7e-6, 5e-4, 480)) + list(
                np.linspace(5e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40))
            
            train_args_conv_first_ep100_sp_more_drop = copy.deepcopy(train_args_conv_first_ep100_sp)
            model_config_more_drop = copy.deepcopy(model_config)
            model_config_more_drop.conv_dropout = 0.15
            model_config_more_drop.ff_dropout = 0.15
            model_config_more_drop.mhsa_dropout = 0.15
            model_config_more_drop.final_dropout = 0.15
            train_args_conv_first_ep100_sp_more_drop["net_args"] = {"model_config_dict": asdict(model_config_more_drop)}

            # Go closer to other experiments, full weight decay and full specaugment
            train_args_conv_first_ep100_sp_late_peak_full_reg = copy.deepcopy(train_args_conv_first_ep100_sp_late_peak)
            train_args_conv_first_ep100_sp_late_peak_full_reg["config"]["optimizer"]["weight_decay"] = 1e-2
            model_config_full_spec = copy.deepcopy(model_config)
            model_config_full_spec.specaug_config = specaug_config_full
            train_args_conv_first_ep100_sp_late_peak_full_reg["net_args"] = {"model_config_dict": asdict(model_config_full_spec)}


            # Go closer to other experiments, full weight decay and full specaugment
            train_args_conv_first_ep100_sp_late_peak_full_reg = copy.deepcopy(train_args_conv_first_ep100_sp_late_peak)
            train_args_conv_first_ep100_sp_late_peak_full_reg["config"]["optimizer"]["weight_decay"] = 1e-2
            model_config_full_spec = copy.deepcopy(model_config)
            model_config_full_spec.specaug_config = specaug_config_full
            train_args_conv_first_ep100_sp_late_peak_full_reg["net_args"] = {"model_config_dict": asdict(model_config_full_spec)}

            frontend_config_256 = copy.deepcopy(frontend_config)
            frontend_config_256.out_features = 256

            model_config_16x256 = ModelConfig(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config_256,
                specaug_config=specaug_config,
                label_target_size=vocab_size_without_blank,
                conformer_size=256,
                num_layers=16,
                num_heads=4,
                ff_dim=1024,
                att_weights_dropout=0.1,
                conv_dropout=0.1,
                ff_dropout=0.1,
                mhsa_dropout=0.1,
                conv_kernel_size=31,
                final_dropout=0.1,
                specauc_start_epoch=11,  # BPE does not converge otherwise
            )

            train_args_conv_first_ep100_sp_16x256 = copy.deepcopy(train_args_conv_first_ep100_sp)
            train_args_conv_first_ep100_sp_16x256["net_args"] = {"model_config_dict": asdict(model_config_16x256)}


            train_args_pairs = [
                (".512dim_sub4_24gbgpu_100eps", train_args_conv_first_ep100),
                (".512dim_sub4_24gbgpu_100eps_sp", train_args_conv_first_ep100_sp)
            ]

            if BPE_SIZE == 128:
                train_args_pairs += [
                    (".16x256dim_sub4_24gbgpu_100eps_sp", train_args_conv_first_ep100_sp_16x256),
                    (".512dim_sub4_24gbgpu_100eps_sp_late_peak", train_args_conv_first_ep100_sp_late_peak),
                    (".512dim_sub4_24gbgpu_100eps_sp_more_drop", train_args_conv_first_ep100_sp_more_drop),
                    (".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec", train_args_conv_first_ep100_sp_late_peak_full_reg)
                ]

            for name, train_args in train_args_pairs:
                training_name = prefix_name + "/" + str(
                    BPE_SIZE) + "/" + network_module_conv_first + name
                train_job = training(training_name, train_data_bpe, train_args, num_epochs=1000, **default_returnn)
                train_job.rqmt["gpu_mem"] = 24
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
                    get_specific_checkpoint=1000
                )
                add_ctc_model(f"ls960_ctc_bpe_{BPE_SIZE}." + network_module_conv_first + name + "_ckpt1000",
                              asr_model)
                tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model,
                                         default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0],
                                         prior_scales=[0.2, 0.3, 0.4])
                greedy_decoder_config = GreedyDecoderConfig(
                    returnn_vocab=label_datastream_bpe.vocab,
                )
                greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

