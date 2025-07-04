from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import get_ctc_model, add_rnnt_model
from ... import PACKAGE


def rnnt_bpe_ls960_0924_relposencoder():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_relposencoder_0924"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )



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

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.rnnt.decoder.experimental_rnnt_decoder import DecoderConfig, ExtraConfig

    def evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        unhashed_decoder_config: Optional[ExtraConfig] = None,
        beam_size: int = 1,
        use_gpu=False,
        decoder_module="rnnt.decoder.experimental_rnnt_decoder",
        debug=False,
        with_test=True,
        extra_forward_config={},
    ):
        """
        Example helper to execute tuning over lm_scales and prior scales.
        With the best values runs test-clean and test-other.

        This is just a reference helper and can (should) be freely changed, copied, modified etc...

        :param training_name: for alias and output names
        :param asr_model: ASR model to use
        :param base_decoder_config: any decoder config dataclass

        """
        decoder_config = copy.deepcopy(base_decoder_config)
        decoder_config.beam_size = beam_size
        search_name = training_name + "/search_bs%i" % beam_size
        dataset_tuples = {**dev_dataset_tuples, **test_dataset_tuples} if with_test else {**dev_dataset_tuples}
        search_jobs, wers = search(
            search_name,
            forward_config= {"seed": 2, **extra_forward_config} if use_gpu else {**extra_forward_config},
            asr_model=asr_model,
            decoder_module=decoder_module,
            decoder_args={"config": asdict(decoder_config)},
            unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
            test_dataset_tuples=dataset_tuples,
            use_gpu=use_gpu,
            debug=debug,
            **default_returnn,
        )

    from ...pytorch_networks.rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        PredictorConfig,
        ConformerPosEmbConfig,
    )

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
        max_dim_feat=16,  # normal style
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

    # Try to do like returnn frontend
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    
    for BPE_SIZE in [128, 512]:
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=True,  # RNN-T now, use postfix
        )
        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

        decoder_config_bpeany_greedy = DecoderConfig(
            beam_size=1,  # greedy as default
            returnn_vocab=label_datastream_bpe.vocab
        )

        from ...pytorch_networks.rnnt.decoder.experimental_rnnt_decoder_v2 import DecoderConfig as DecoderConfigV2
        decoder_config_bpeany_greedy_v2 = DecoderConfigV2(
            beam_size=1,  # greedy as default
            returnn_vocab=label_datastream_bpe.vocab
        )

        model_config_v5_sub6_512lstm = ModelConfig(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config,
            pos_emb_config=posemb_config,
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
            mhsa_with_bias=True,
            conv_kernel_size=31,
            final_dropout=0.1,
            specauc_start_epoch=11,
            joiner_dim=640,
            joiner_activation="relu",
            joiner_dropout=0.1,
            dropout_broadcast_axes=None,  # No dropout broadcast yet to properly compare
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=[11],
            aux_ctc_loss_scales=[0.3],
        )

        KEEP = [300, 400, 500, 600, 700, 800, 900, 950, 980]
        network_module = "rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
        train_config_24gbgpu_amp_radam = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates":list(np.linspace(5e-5, 5e-4, 240)) + list(
            np.linspace(5e-4, 5e-5, 720)) + list(np.linspace(5e-5, 1e-7, 40)),
            #############
            "batch_size": 240 * 16000,
            "gradient_clip_norm": 1.0,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "torch_amp_options": {"dtype": "bfloat16"},
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": KEEP
            }
        }
        train_args_radam = {
            "config": train_config_24gbgpu_amp_radam,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
            "include_native_ops": True,
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = prefix_name + "/" + str(
            BPE_SIZE) + "/" + network_module + ".512dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec11_sp_morel2"
        train_job = training(training_name, train_data_bpe,
                             train_args_radam,
                             num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for keep in KEEP:
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_radam,
                with_prior=False,
                datasets=train_data_bpe, get_specific_checkpoint=keep
            )
            evaluate_helper(
                training_name + "/keep_%i" % keep,
                asr_model,
                decoder_config_bpeany_greedy,
                use_gpu=True
            )
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_radam,
            with_prior=False,
            datasets=train_data_bpe, get_specific_checkpoint=1000
        )
        evaluate_helper(
            training_name + "/keep_%i" % 1000,
            asr_model,
            decoder_config_bpeany_greedy,
            use_gpu=True,
        )
        evaluate_helper(
            training_name + "/keep_%i" % 1000,
            asr_model,
            decoder_config_bpeany_greedy,
            beam_size=10,
            use_gpu=True,
        )
        evaluate_helper(
            training_name + "/keep_%i" % 1000,
            asr_model,
            decoder_config_bpeany_greedy,
            beam_size=4,
            use_gpu=True,
        )

        asr_model.lexicon = get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=BPE_SIZE)
        asr_model.returnn_vocab = label_datastream_bpe.vocab
        asr_model.settings = train_settings
        asr_model.label_datastream = label_datastream_bpe
        add_rnnt_model(network_module + f".bpe{BPE_SIZE}.512dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec11_sp_morel2", asr_model)

        from ...storage import get_lm_model, NeuralLM
        lstm_2x1024 : NeuralLM  = get_lm_model("bpe%i_2x2024_kazuki_lstmlm_3ep" % BPE_SIZE)

        from ...pytorch_networks.rnnt.decoder.experimental_rnnt_decoder_v3 import DecoderConfig as DecoderConfigV3
        from ...pytorch_networks.rnnt.decoder.experimental_rnnt_decoder_v3 import ExtraConfig as DecoderExtraConfigV3
        from i6_core.returnn.config import CodeWrapper
        decoder_config_bpeany_greedy_v3 = DecoderConfigV3(
            beam_size=1,  # greedy as default
            returnn_vocab=label_datastream_bpe.vocab,
            lm_model_args=lstm_2x1024.net_args,
            lm_checkpoint=lstm_2x1024.checkpoint,
            lm_module="pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v2.Model",
            lm_scale=0.2,
            zero_ilm_scale=0.1,
        )

        trafo_12x768 : NeuralLM = get_lm_model("bpe%i_trafo12x768_2ep" % BPE_SIZE)
        trafo_24x768 : NeuralLM = get_lm_model("bpe%i_trafo24x768_3ep" % BPE_SIZE)
        trafo_24x768_5ep : NeuralLM = get_lm_model("bpe%i_trafo24x768_5ep" % BPE_SIZE)
        trafo_32x768_5ep : NeuralLM = get_lm_model("bpe%i_trafo32x768_5ep" % BPE_SIZE)

        decoder_config_trafo = DecoderConfigV3(
            beam_size=1,  # greedy as default
            returnn_vocab=label_datastream_bpe.vocab,
            lm_model_args=trafo_12x768.net_args,
            lm_checkpoint=trafo_12x768.checkpoint,
            lm_module="pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2.Model",
            lm_scale=0.2,
            zero_ilm_scale=0.1,
        )
        decoder_config_trafo_24 = copy.deepcopy(decoder_config_trafo)
        decoder_config_trafo_24.lm_model_args=trafo_24x768.net_args
        decoder_config_trafo_24.lm_checkpoint=trafo_24x768.checkpoint

        decoder_config_trafo_24_5ep = copy.deepcopy(decoder_config_trafo)
        decoder_config_trafo_24_5ep.lm_model_args=trafo_24x768_5ep.net_args
        decoder_config_trafo_24_5ep.lm_checkpoint=trafo_24x768_5ep.checkpoint

        decoder_config_trafo_32_5ep = copy.deepcopy(decoder_config_trafo)
        decoder_config_trafo_32_5ep.lm_model_args=trafo_32x768_5ep.net_args
        decoder_config_trafo_32_5ep.lm_checkpoint=trafo_32x768_5ep.checkpoint

        decoder_unhashed_config_v3 = DecoderExtraConfigV3(
            lm_package=PACKAGE,
        )

        if BPE_SIZE == 128:
            evaluate_helper(
                training_name + "/keep_%i_decv3_cpu" % 1000,
                asr_model,
                decoder_config_bpeany_greedy_v3,
                unhashed_decoder_config=decoder_unhashed_config_v3,
                beam_size=10,
                use_gpu=False,
                decoder_module="rnnt.decoder.experimental_rnnt_decoder_v3",
                debug=True,
            )

        for lm_scale in [0.3, 0.35, 0.4, 0.45]:
            for prior_scale in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35]:
                decoder_settings = copy.deepcopy(decoder_config_bpeany_greedy_v3)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_bs10_lstmlm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=10,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v3",
                    debug=True,
                    with_test=False,
                    extra_forward_config={"batch_size": 200 * 16000}
                )
                decoder_settings = copy.deepcopy(decoder_config_bpeany_greedy_v3)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_bs4_lstmlm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=4,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v3",
                    debug=True,
                    with_test=False
                )

        for lm_scale in [0.4, 0.45, 0.5, 0.55, 0.60]:
            for prior_scale in [0.3, 0.35, 0.4, 0.45]:
                decoder_settings = copy.deepcopy(decoder_config_trafo)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_trafolm_12x768/trafolm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=4,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v4",
                    debug=True,
                    with_test=False,
                )
                decoder_settings = copy.deepcopy(decoder_config_trafo)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_trafolm_12x768/trafolm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=10,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v4",
                    debug=True,
                    with_test=False,
                    extra_forward_config={"batch_size": 200*16000}
                )
                # 24 layer TRAFO
                decoder_settings = copy.deepcopy(decoder_config_trafo_24)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_trafolm_24x768/trafolm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=4,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v4",
                    debug=True,
                    with_test=False,
                    extra_forward_config={"batch_size": 200 * 16000}
                )
                # 24 layer TRAFO (OOM)
                decoder_settings = copy.deepcopy(decoder_config_trafo_24)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_trafolm_24x768/trafolm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=10,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v4",
                    debug=True,
                    with_test=False,
                    extra_forward_config={"batch_size": 200 * 16000}
                )
                # 24 layer TRAFO 5EP
                decoder_settings = copy.deepcopy(decoder_config_trafo_24_5ep)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_trafolm_24x768_5ep/trafolm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=4,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v4",
                    debug=True,
                    with_test=False,
                    extra_forward_config={"batch_size": 200 * 16000}
                )
                # 24 layer TRAFO 5EP
                decoder_settings = copy.deepcopy(decoder_config_trafo_24_5ep)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_trafolm_24x768_5ep/trafolm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=10,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v4",
                    debug=True,
                    with_test=False,
                    extra_forward_config={"batch_size": 200 * 16000}
                )
                # 32 layer TRAFO 5EP
                decoder_settings = copy.deepcopy(decoder_config_trafo_32_5ep)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_trafolm_32x768_5ep/trafolm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=4,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v4",
                    debug=True,
                    with_test=False,
                    extra_forward_config={"batch_size": 200 * 16000}
                )
                # 32 layer TRAFO 5EP
                decoder_settings = copy.deepcopy(decoder_config_trafo_32_5ep)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_trafolm_32x768_5ep/trafolm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=10,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v4",
                    debug=True,
                    with_test=False,
                    extra_forward_config={"batch_size": 100 * 16000}
                )


        if BPE_SIZE == 128:
            decoder_settings = copy.deepcopy(decoder_config_bpeany_greedy_v3)
            decoder_settings.lm_scale = 0.3
            decoder_settings.zero_ilm_scale = 0.25
            evaluate_helper(
                training_name + "/keep_%i_bs10_lsttlm_0.3_0.25" % 1000,
                asr_model,
                decoder_settings,
                unhashed_decoder_config=decoder_unhashed_config_v3,
                beam_size=10,
                use_gpu=True,
                decoder_module="rnnt.decoder.experimental_rnnt_parallel_decoder_v3",
                debug=True,
                extra_forward_config={"batch_size": 200 * 16000},
            )
            
        # more tests -> dropout broadcasting does not work because of 4-dim pre-softmax tensor

        # model_config_v5_sub6_512lstm_predbroadcast = copy.deepcopy(model_config_v5_sub6_512lstm)
        # model_config_v5_sub6_512lstm_predbroadcast.dropout_broadcast_axes = "BT"
        # train_args_radam = {
        #     "config": train_config_24gbgpu_amp_radam,
        #     "network_module": network_module,
        #     "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm_predbroadcast)},
        #     "include_native_ops": True,
        #     "use_speed_perturbation": True,
        #     "debug": False,
        # }

        # training_name = prefix_name + "/" + str(
        #     BPE_SIZE) + "/" + network_module + ".512dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec11_sp_morel2_dropbroadcastbt"
        # train_job = training(training_name, train_data_bpe,
        #                      train_args_radam,
        #                      num_epochs=1000, **default_returnn)
        # train_job.rqmt["gpu_mem"] = 24
        # train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # asr_model = prepare_asr_model(
        #     training_name, train_job, train_args_radam,
        #     with_prior=False,
        #     datasets=train_data_bpe, get_specific_checkpoint=keep
        # )

        # evaluate_helper(
        #     training_name + "/keep_%i" % 1000,
        #     asr_model,
        #     decoder_config_bpeany_greedy,
        #     beam_size=4,
        #     use_gpu=True,
        # )
        
        # NEW LR scheduling
        network_module = "rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
        train_config_24gbgpu_amp_radam = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates":list(np.linspace(5e-5, 5e-4, 480)) + list(
            np.linspace(5e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40)),
            #############
            "batch_size": 240 * 16000,
            "gradient_clip_norm": 1.0,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "torch_amp_options": {"dtype": "bfloat16"},
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": KEEP
            }
        }
        train_args_radam = {
            "config": train_config_24gbgpu_amp_radam,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
            "include_native_ops": True,
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = prefix_name + "/" + str(
            BPE_SIZE) + "/" + network_module + ".512dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec11_sp_morel2_centerLR"
        train_job = training(training_name, train_data_bpe,
                             train_args_radam,
                             num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        # train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_radam,
            with_prior=False,
            datasets=train_data_bpe, get_specific_checkpoint=1000
        )
        evaluate_helper(
            training_name + "/keep_%i" % 1000,
            asr_model,
            decoder_config_bpeany_greedy,
            use_gpu=True,
        )
        
        
        for lm_scale in [0.45, 0.5, 0.55, 0.60]:
            for prior_scale in [0.3, 0.35, 0.4]:
                # 32 layer TRAFO 5EP
                decoder_settings = copy.deepcopy(decoder_config_trafo_32_5ep)
                decoder_settings.lm_scale = lm_scale
                decoder_settings.zero_ilm_scale = prior_scale
                evaluate_helper(
                    training_name + "/keep_%i_trafolm_32x768_5ep/trafolm_%.2f_%.2f" % (1000, lm_scale, prior_scale),
                    asr_model,
                    decoder_settings,
                    unhashed_decoder_config=decoder_unhashed_config_v3,
                    beam_size=10,
                    use_gpu=True,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder_v4",
                    debug=True,
                    with_test=False,
                    extra_forward_config={"batch_size": 100 * 16000}
                )

