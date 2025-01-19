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
from ...storage import add_ctc_model, get_lm_model
from ...report import tune_and_evalue_report



def bpe_ls960_1023_low_vocab_test():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_bpe_low_vocab"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )
    
    train_settings_no_peak_norm = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=False,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )
    
    train_settings_no_preemhasis = DatasetSettings(
        preemphasis=None,  # TODO: Check if this is really useful
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
    from ...pytorch_networks.ctc.decoder.flashlight_ctc_lexical_free_v1 import DecoderConfig as LexicalFreeDecoderConfig


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

    def tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, base_decoder_config, lm_scales, prior_scales, decoder_module="ctc.decoder.flashlight_ctc_v1", debug=False, rescale_mode=False, run_test=True, use_gpu=False):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
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
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    debug=debug,
                    use_gpu=use_gpu,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))
                if rescale_mode:
                    assert use_gpu is False
                    for job in search_jobs:
                        job.rqmt["sbatch_args"] = ["-A", "rescale_speed", "-p", "rescale_amd"]
                if use_gpu:
                    for job in search_jobs:
                        job.rqmt["sbatch_args"] = ["-p", "gpu_test_24gb"]

        if run_test:
            for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
                pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values, mode="minimize")
                pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
                decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
                search_jobs, wers = search(
                    training_name, forward_config={}, asr_model=asr_model, decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)}, test_dataset_tuples={key: test_dataset_tuples[key]},
                    **default_returnn
                )
                report_values[key] = wers[training_name + "/" + key]

            tune_and_evalue_report(
                training_name=training_name,
                tune_parameters=tune_parameters,
                tuning_names=["LM", "Prior"],
                tune_values_clean=tune_values_clean,
                tune_values_other=tune_values_other,
                report_values=report_values
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

    def lexical_free_search(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: LexicalFreeDecoderConfig,
            debug=False,
    ):
        # remove prior if exists
        asr_model=copy.deepcopy(asr_model)
        asr_model.prior_file = None
        search_name = training_name + "/search_lexical_free"
        search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module="ctc.decoder.flashlight_ctc_lexical_free_v1",
            decoder_args={"config": asdict(decoder_config)},
            # test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            test_dataset_tuples={**dev_dataset_tuples},
            debug=debug,
            **default_returnn,
        )

    for BPE_SIZE in [0, 128, 256, 512, 1024]:

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=False,
        )
        train_data_bpe_no_peak_norm = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings_no_peak_norm,
            use_postfix=False,
        )
        train_data_bpe_no_preemph = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings_no_preemhasis,
            use_postfix=False,
        )
        
        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

        dev_dataset_tuples = {}
        dev_dataset_tuples_no_peaknorm = {}
        dev_dataset_tuples_no_preemph = {}
        for testset in ["dev-clean", "dev-other"]:
            dev_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings,
            )
            dev_dataset_tuples_no_peaknorm[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings_no_peak_norm,
            )
            dev_dataset_tuples_no_preemph[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings_no_preemhasis,
            )
            
        test_dataset_tuples = {}
        test_dataset_tuples_no_peaknorm = {}
        test_dataset_tuples_no_preemph = {}
        for testset in ["test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings,
            )
            test_dataset_tuples_no_peaknorm[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings_no_peak_norm,
            )
            test_dataset_tuples_no_preemph[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings_no_preemhasis,
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

        if BPE_SIZE != 0:
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

        if BPE_SIZE != 0:
            train_job = training(training_name, train_data_bpe, train_args_conv_first, num_epochs=500,
                                 **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_conv_first, with_prior=True, datasets=train_data_bpe,
                get_specific_checkpoint=500
            )
            tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0],
                                     prior_scales=[0.2, 0.3, 0.4])
            greedy_decoder_config = GreedyDecoderConfig(
                returnn_vocab=label_datastream_bpe.vocab,
            )
            greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

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

        if BPE_SIZE == 0 or BPE_SIZE == 128 or BPE_SIZE == 512:
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

            train_args_conv_first_ep100_sp_late_peak_full_reg_gradnorm = copy.deepcopy(train_args_conv_first_ep100_sp_late_peak_full_reg)
            train_args_conv_first_ep100_sp_late_peak_full_reg_gradnorm["config"].pop("gradient_clip")
            train_args_conv_first_ep100_sp_late_peak_full_reg_gradnorm["config"]["gradient_clip_norm"] = 1.0

            # train_args_conv_first_ep100_sp_late_peak_full_reg_grad_norm = copy.deepcopy(train_args_conv_first_ep100_sp_late_peak_full_reg)

            # Radam and plateau
            train_args_conv_first_ep100_sp_radam_plateau_full_reg = copy.deepcopy(train_args_conv_first_ep100_sp_late_peak_full_reg)
            train_args_conv_first_ep100_sp_radam_plateau_full_reg["config"]["optimizer"]["class"] = "radam"
            train_args_conv_first_ep100_sp_radam_plateau_full_reg["config"]["optimizer"]["decoupled_weight_decay"] = True
            train_args_conv_first_ep100_sp_radam_plateau_full_reg["config"]["learning_rates"] = list(np.linspace(1e-4, 5e-4, 240)) + list(
                np.linspace(5e-4, 5e-4, 240)) + list(np.linspace(5e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40))


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

            train_args_pairs = []
            if BPE_SIZE == 128 or BPE_SIZE == 512:
                train_args_pairs += [
                    (".512dim_sub4_24gbgpu_100eps", train_args_conv_first_ep100),
                    (".512dim_sub4_24gbgpu_100eps_sp", train_args_conv_first_ep100_sp)
                ]

            if BPE_SIZE == 128:
                train_args_pairs += [
                    (".16x256dim_sub4_24gbgpu_100eps_sp", train_args_conv_first_ep100_sp_16x256),
                    (".512dim_sub4_24gbgpu_100eps_sp_late_peak", train_args_conv_first_ep100_sp_late_peak),
                    (".512dim_sub4_24gbgpu_100eps_sp_more_drop", train_args_conv_first_ep100_sp_more_drop),
                    (".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec", train_args_conv_first_ep100_sp_late_peak_full_reg),
                    (".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm", train_args_conv_first_ep100_sp_late_peak_full_reg_gradnorm),
                    (".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_radam_plateau", train_args_conv_first_ep100_sp_radam_plateau_full_reg)
                ]

            if BPE_SIZE == 0:
                train_args_pairs += [
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

            if BPE_SIZE == 128:
                # one experiment additionally without peak normalization
                name = ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec"
                train_args = train_args_conv_first_ep100_sp_late_peak_full_reg
                training_name = prefix_name + "/" + str(
                    BPE_SIZE) + "/no_peaknorm_" + network_module_conv_first + name
                train_job = training(training_name, train_data_bpe_no_peak_norm, train_args, num_epochs=1000, **default_returnn)
                train_job.rqmt["gpu_mem"] = 24
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
                    get_specific_checkpoint=1000
                )
                tune_and_evaluate_helper(training_name, dev_dataset_tuples_no_peaknorm, test_dataset_tuples_no_peaknorm, asr_model,
                                         default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0],
                                         prior_scales=[0.2, 0.3, 0.4])
                greedy_decoder_config = GreedyDecoderConfig(
                    returnn_vocab=label_datastream_bpe.vocab,
                )
                greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
                
                # one experiment additionally without peak normalization
                name = ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec"
                train_args = train_args_conv_first_ep100_sp_late_peak_full_reg
                training_name = prefix_name + "/" + str(
                    BPE_SIZE) + "/no_preemph_" + network_module_conv_first + name
                train_job = training(training_name, train_data_bpe_no_preemph, train_args, num_epochs=1000, **default_returnn)
                train_job.rqmt["gpu_mem"] = 24
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
                    get_specific_checkpoint=1000
                )
                tune_and_evaluate_helper(training_name, dev_dataset_tuples_no_preemph, test_dataset_tuples_no_preemph, asr_model,
                                         default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0],
                                         prior_scales=[0.2, 0.3, 0.4])
                greedy_decoder_config = GreedyDecoderConfig(
                    returnn_vocab=label_datastream_bpe.vocab,
                )
                greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

                # one experiment with reduced batch size
                name = ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch"
                train_args = copy.deepcopy(train_args_conv_first_ep100_sp_late_peak_full_reg_gradnorm)
                train_args["config"]["batch_size"] = 240 * 16000

                training_name = prefix_name + "/" + str(
                    BPE_SIZE) + "/" + network_module_conv_first + name
                train_job = training(training_name, train_data_bpe, train_args, num_epochs=1000, **default_returnn)
                train_job.rqmt["gpu_mem"] = 24
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
                    get_specific_checkpoint=1000
                )
                tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model,
                                         default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0],
                                         prior_scales=[0.2, 0.3, 0.4])
                greedy_decoder_config = GreedyDecoderConfig(
                    returnn_vocab=label_datastream_bpe.vocab,
                )
                greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)


                # lexical free search test
                lexical_free_config = LexicalFreeDecoderConfig(
                    beam_size=256,
                    beam_size_token=8,
                    beam_threshold=14,
                    returnn_vocab=label_datastream_bpe.vocab,
                )
                lexical_free_search(training_name, asr_model=asr_model, decoder_config=lexical_free_config, debug=True)
                lexical_free_config = LexicalFreeDecoderConfig(
                    beam_size=256,
                    beam_size_token=8,
                    beam_threshold=50,
                    returnn_vocab=label_datastream_bpe.vocab,
                )
                lexical_free_search(training_name + "/th50", asr_model=asr_model, decoder_config=lexical_free_config, debug=True)

                # Neural LM test
                from ...pytorch_networks.ctc.decoder.flashlight_ctc_v2_neural_lm import DecoderConfig as DecoderConfigNeuralLM
                neural_lm = get_lm_model("bpe5k_2x2024_kazuki_lstmlm_3ep")
                decoder_config_neural_lm = DecoderConfigNeuralLM(
                    beam_size=default_decoder_config_bpe.beam_size,
                    beam_size_token=default_decoder_config_bpe.beam_size_token,
                    beam_threshold=default_decoder_config_bpe.beam_threshold,
                    lexicon=default_decoder_config_bpe.lexicon,
                    returnn_vocab=default_decoder_config_bpe.returnn_vocab,
                    lm_vocab=neural_lm.bpe_vocab,
                    lm_bpe_codes=neural_lm.bpe_codes,
                    lm_is_bpe=True,
                    lm_checkpoint=neural_lm.checkpoint,
                    lm_module=neural_lm.network_module,
                    lm_args=neural_lm.net_args
                )

                # Did not work so far, TODO: maybe try later again
                #tune_and_evaluate_helper(training_name + "_neurallm", dev_dataset_tuples, test_dataset_tuples, asr_model,
                #                         decoder_config_neural_lm, lm_scales=[1.8],
                #                         prior_scales=[0.2], decoder_module="ctc.decoder.flashlight_ctc_v2_neural_lm", debug=True)
                
                
                # Neural LM rescoring test
                from ...pytorch_networks.ctc.decoder.flashlight_ctc_v2_neural_lm_rescoring import DecoderConfig as DecoderConfigRescoringNeuralLM
                neural_lm = get_lm_model("bpe5k_2x2024_kazuki_lstmlm_3ep")
                decoder_config_neural_lm = DecoderConfigRescoringNeuralLM(
                    beam_size=default_decoder_config_bpe.beam_size,
                    beam_size_token=default_decoder_config_bpe.beam_size_token,
                    beam_threshold=default_decoder_config_bpe.beam_threshold,
                    lexicon=default_decoder_config_bpe.lexicon,
                    returnn_vocab=default_decoder_config_bpe.returnn_vocab,
                    lm_vocab=neural_lm.bpe_vocab,
                    lm_bpe_codes=neural_lm.bpe_codes,
                    lm_is_bpe=True,
                    lm_checkpoint=neural_lm.checkpoint,
                    lm_module=neural_lm.network_module,
                    lm_args=neural_lm.net_args,
                    arpa_lm=arpa_4gram_lm,
                    lm_rescore_scale=0.01,
                    lm_length_exponent=0.0,
                    n_best=20,
                )
                # CPU too slow for analysis
                # for scale in [0.0, 0.001, 0.01, 0.1]:
                #     for exponent in [0.0, 0.1, 0.5]:
                #         decoder_config = copy.deepcopy(decoder_config_neural_lm)
                #         decoder_config.lm_rescore_scale = scale
                #         decoder_config.lm_length_exponent = exponent
                #         tune_and_evaluate_helper(training_name + "_neurallm_rescoring/s%.3f_n20_e%.1f" % (scale, exponent), dev_dataset_tuples, test_dataset_tuples, asr_model,
                #                                  decoder_config, lm_scales=[1.8],
                #                                  prior_scales=[0.2], decoder_module="ctc.decoder.flashlight_ctc_v2_neural_lm_rescoring", debug=True, rescale_mode=True, run_test=False)

                # GPU Test
                test_tuples = [
                    (1.8, 0.015, 0.0, 20, 14),
                    (1.8, 0.015, 0.0, 50, 14),
                    (1.8, 0.015, 0.0, 50, 16),
                    (1.6, 0.015, 0.0, 20, 14),
                    (1.6, 0.010, 0.0, 100, 16),
                    (1.4, 0.010, 0.0, 100, 16),
                    (1.2, 0.005, 0.0, 100, 16),
                    (1.2, 0.010, 0.0, 100, 16),
                    (1.2, 0.015, 0.0, 100, 16),
                    (1.2, 0.020, 0.0, 100, 16),
                    (1.2, 0.50, 1.0, 100, 16),
                    (1.2, 0.0, 1.0, 100, 16),
                    (1.2, 0.1, 1.0, 100, 16),
                    (1.2, 1.0, 1.0, 100, 16),
                    (1.2, 1.5, 1.0, 100, 16),
                ]
                for t in test_tuples:
                    decoder_config = copy.deepcopy(decoder_config_neural_lm)
                    decoder_config.lm_rescore_scale = t[1]
                    decoder_config.lm_length_exponent = t[2]
                    decoder_config.n_best = t[3]
                    decoder_config.beam_threshold = t[4]
                    tune_and_evaluate_helper(
                        training_name + "_neurallm_rescoring/gpu_s%.3f_e%.1f_n%i_th%i" % (t[1], t[2], t[3], t[4]),
                        dev_dataset_tuples, test_dataset_tuples, asr_model,
                        decoder_config, lm_scales=[t[0]],
                        prior_scales=[0.2], decoder_module="ctc.decoder.flashlight_ctc_v2_neural_lm_rescoring",
                        debug=True, use_gpu=True, run_test=False)
