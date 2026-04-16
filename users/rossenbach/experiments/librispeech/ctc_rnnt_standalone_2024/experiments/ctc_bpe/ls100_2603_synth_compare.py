from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import synthetic_ogg_zip_data

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import add_ctc_model, get_lm_model, NeuralLM
from ...report import tune_and_evalue_report
from ... import PACKAGE


def bpe_ls100_2603_synth_compare():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls100_ctc_bpe"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=3,
        train_seq_ordering="laplace:.1000",
    )

    def get_training_settings_with_partition(partition):
        return DatasetSettings(
            preemphasis=0.97,  # TODO: Check if this is really useful
            peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
            # training
            train_partition_epoch=partition,
            train_seq_ordering="laplace:.1000",
        )
    
    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig
    
    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config, ConformerPosEmbConfig

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
        out_features=384,
        activation=None,
    )

    # Try to do like returnn frontend
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )


    def tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, base_decoder_config,
                                 lm_scales, prior_scales, unhashed_decoder_config=None,
                                 decoder_module="ctc.decoder.flashlight_ctc_v1", debug=False, use_gpu=False,
                                 extra_forward_config=None):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                if hasattr(decoder_config, "lm_scale"):
                    decoder_config.lm_scale = lm_weight
                else:
                    decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.2f_prior%.2f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config=extra_forward_config if extra_forward_config else {},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    unhashed_decoder_args={
                        "extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
                    test_dataset_tuples=dev_dataset_tuples,
                    debug=debug,
                    use_gpu=use_gpu,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values,
                                                                        mode="minimize")
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            if hasattr(decoder_config, "lm_scale"):
                decoder_config.lm_scale = pick_optimal_params_job.out_optimal_parameters[0]
            else:
                decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config=extra_forward_config if extra_forward_config else {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                unhashed_decoder_args={
                    "extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
                test_dataset_tuples={key: test_dataset_tuples[key]},
                use_gpu=use_gpu,
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
            decoder_config: GreedyDecoderConfig,
            dev_dataset_tuples,
            test_dataset_tuples,
            forward_config = None,
            use_dynamic_quant = False,
        ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={} if forward_config is None else forward_config,
            asr_model=asr_model,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3_dynamic_quant" if use_dynamic_quant else "ctc.decoder.greedy_bpe_ctc_v3",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )
        return search_jobs

    # for BPE_SIZE in [0, 128, 512]:
    for BPE_SIZE in [128, 512]:

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-clean-100",
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
            pos_emb_config=posemb_config,
            specaug_config=specaug_config_full,
            label_target_size=vocab_size_without_blank,
            conformer_size=384,
            num_layers=12,
            num_heads=8,
            ff_dim=1536,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            mhsa_with_bias=True,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11, #
            dropout_broadcast_axes=None,  # No dropout broadcast yet to properly compare
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
        )

        default_decoder_config_bpe = DecoderConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-clean-100", bpe_size=BPE_SIZE),
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=1024,
            beam_size_token=16,  # makes it much faster
            arpa_lm=arpa_4gram_lm,
            beam_threshold=14,
        )

        train_config_24gbgpu = {
            "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            # There is a higher LR, because this model is only 384 in dimension
            "learning_rates": list(np.linspace(7e-5, 7e-4, 140)) + list(
                np.linspace(7e-4, 7e-5, 140)) + list(np.linspace(7e-5, 1e-7, 20)),
            #############
            "batch_size": 240 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "torch_amp_options": {"dtype": "bfloat16"},
        }

        network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
        train_args = {
            "config": train_config_24gbgpu,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".384dim_sub4_24gbgpu_100eps_radam"
        train_job = training(training_name, train_data_bpe, train_args, num_epochs=300,
                             **default_returnn)
        train_job.rqmt["gpu_mem"] = 24

        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
            get_specific_checkpoint=300
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
                                 prior_scales=[0.1, 0.2, 0.3, 0.4])
        greedy_decoder_config = GreedyDecoderConfig(
            returnn_vocab=label_datastream_bpe.vocab,
        )
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config, dev_dataset_tuples=dev_dataset_tuples, test_dataset_tuples=test_dataset_tuples)


        if BPE_SIZE == 512:
            from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v5 import DecoderConfig as BeamSearchDecoderConfig, \
                DecoderExtraConfig as BeamSearchExtraConfig
            trafo_24x768 : NeuralLM = get_lm_model("ls100_bpe%i_trafo24x768_5ep" % BPE_SIZE)

            beam_search_config = BeamSearchDecoderConfig(
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=12,
                lm_model_args = trafo_24x768.net_args,
                lm_checkpoint = trafo_24x768.checkpoint,
                lm_module = "pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2.Model",
                lm_states_need_label_axis = True,
            )

            extra_config = BeamSearchExtraConfig(
                lm_package=PACKAGE
            )

            tune_and_evaluate_helper(training_name + "/trafo_lm_24x768", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_config, unhashed_decoder_config=extra_config, lm_scales=[0.0, 0.5, 1.0, 1.5, 2.0],
                                     prior_scales=[0.0, 0.2, 0.4], use_gpu=True, decoder_module="ctc.decoder.beam_search_bpe_ctc_v5", extra_forward_config={"max_seqs": 1})


            tune_and_evaluate_helper(
                training_name + "/trafo_lm_24x768_tune2", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_config, unhashed_decoder_config=extra_config,
                lm_scales=[1.0, 1.25, 1.5, 1.75, 2.0],
                prior_scales=[0.0, 0.3, 0.4, 0.5], use_gpu=True,
                decoder_module="ctc.decoder.beam_search_bpe_ctc_v5",
                extra_forward_config={"max_seqs": 1}
            )

            train_args_resume = copy.deepcopy(train_args)
            train_args_resume["config"][
                "import_model_train_epoch1"] = asr_model.checkpoint  # only get checkpoint, rest should be identical

            training_name = prefix_name + "/" + str(
                BPE_SIZE) + "/" + network_module + ".384dim_sub4_24gbgpu_100eps_radam_resume"
            train_job = training(training_name, train_data_bpe, train_args_resume, num_epochs=300, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_resume, with_prior=False, datasets=train_data_bpe,
                get_specific_checkpoint=300,
            )

            synth_keys = [
                #"ar_tts.tacotron2_decoding.tacotron2_decoding_v2_base320_fromglowbase256_400eps_gl32_syn_train-clean-360",
                "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360",
                #"nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-360",
                #"nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_train-clean-360",
                #"glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-360",
                # "glow_tts.glow_tts_v1_glow256align_400eps_oclr_nodrop_gl32_noise0.7_syn_train-clean-360",
                # "glow_tts.glow_tts_v1_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn_train-clean-360",
                #"grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-360",
                # ------------------------------------
                # "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360",
                # "glow_tts.glow_tts_v1_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn_train-clean-360",
            ]

            for synth_key in synth_keys:
                name = "/combine_3to1_synth_partition12/" + synth_key
                synth_ogg = synthetic_ogg_zip_data[synth_key]  # bliss and zip, so take zip
                train_data_bpe = build_bpe_training_datasets(
                    prefix=prefix_name,
                    librispeech_key="train-clean-100",
                    bpe_size=BPE_SIZE,
                    settings=get_training_settings_with_partition(12),
                    use_postfix=True,  # AED, use postfix
                    extra_train_ogg_zips=[synth_ogg],
                    data_repetition_factors=[3, 1],  ## 3x original + 1x synth
                )

                training_name = prefix_name + "/" + str(
                    BPE_SIZE) + "/" + network_module + ".384dim_sub4_24gbgpu_100eps_radam_resume" + name
                train_job = training(training_name, train_data_bpe, train_args_resume, num_epochs=300,
                                     **default_returnn)
                train_job.rqmt["gpu_mem"] = 24
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args_resume, with_prior=False, datasets=train_data_bpe,
                    get_specific_checkpoint=300,
                )