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
from ...storage import get_ctc_model, get_lm_model, NeuralLM, add_aed_model


def aed_bpe_ls960_2504_base():
    ext_prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_aed/2504_low_bpe"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.2000",  # laplace subsampled by num workers
    )

    for BPE_SIZE in [5000, 10000]:
        prefix_name = ext_prefix_name + "/bpe_%i" % BPE_SIZE
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=True,  # AED, use postfix
        )
        label_datastream_bpe= cast(LabelDatastream, train_data_bpe.datastreams["labels"])
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

        default_returnn = {
            "returnn_exe": RETURNN_EXE,
            "returnn_root": MINI_RETURNN_ROOT,
        }

        from ...pytorch_networks.aed.decoder.greedy_prototype import DecoderConfig as GreedyDecoderConfig
        from ...pytorch_networks.aed.decoder.beam_search_single_v1 import DecoderConfig as BeamSearchDecoderConfig, BeamSearchOpts

        from ...pytorch_networks.aed.decoder.beam_search_single_v1_with_lm import \
            DecoderConfig as BeamSearchLmDecoderConfig

        bs_decoder_config = BeamSearchDecoderConfig(
            returnn_vocab=label_datastream_bpe.vocab,
            beam_search_opts=BeamSearchOpts(
                beam_size=12,
                length_normalization_exponent=1.0,
                length_reward=0,
                bos_label=0,
                eos_label=0,
                num_labels=label_datastream_bpe.vocab_size
            )
        )

        def greedy_search_helper(
                training_name: str,
                asr_model: ASRModel,
                decoder_config: GreedyDecoderConfig,
                seed: Optional[int] = None,
                use_gpu: bool = False,
            ):
            # remove prior if exists
            asr_model = copy.deepcopy(asr_model)
            asr_model.prior_file = None

            search_name = training_name + "/search_greedy"
            search_jobs, wers = search(
                search_name,
                forward_config={} if seed is None else {"seed": seed},
                asr_model=asr_model,
                decoder_module="aed.decoder.greedy_prototype",
                decoder_args={"config": asdict(decoder_config)},
                use_gpu=use_gpu,
                test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                **default_returnn,
            )

        def beam_search_prototype(
                training_name: str,
                asr_model: ASRModel,
                decoder_config: BeamSearchDecoderConfig,
                seed: Optional[int] = None,
                use_gpu: bool = False,
                decoder_module: str = "aed.decoder.beam_search_single_v1"
        ):
            # remove prior if exists
            asr_model = copy.deepcopy(asr_model)
            asr_model.prior_file = None

            search_name = training_name + "/search_bs"
            search_jobs, wers = search(
                search_name,
                forward_config={"max_seqs": 20} if seed is None else {"max_seqs": 20, "seed": seed},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                use_gpu=use_gpu,
                debug=True,
                test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                **default_returnn,
            )


        def beam_search_with_lm_prototype(
                training_name: str,
                asr_model: ASRModel,
                decoder_config: BeamSearchDecoderConfig,
                seed: Optional[int] = None,
                use_gpu: bool = False,
                dev_only=True,
                decoder_module: str = "aed.decoder.beam_search_single_v1_with_lm",
        ):
            # remove prior if exists
            asr_model = copy.deepcopy(asr_model)
            asr_model.prior_file = None

            search_name = training_name + "/search_bs"
            search_jobs, wers = search(
                search_name,
                forward_config={"max_seqs": 20} if seed is None else {"max_seqs": 20, "seed": seed},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                use_gpu=use_gpu,
                debug=True,
                test_dataset_tuples={**dev_dataset_tuples} if dev_only else {**dev_dataset_tuples, **test_dataset_tuples},
                **default_returnn,
            )


        from ...pytorch_networks.aed.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_LSTMDecoder_v2_cfg import (
            SpecaugConfig,
            VGG4LayerActFrontendV1Config_mod,
            LogMelFeatureExtractionV1Config,
            AttentionLSTMDecoderV1Config,
            AdditiveAttentionConfig,
            ConformerPosEmbConfig
        )

        from ...pytorch_networks.aed.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_LSTMDecoder_v2_cfg import ModelConfig

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
            max_dim_feat=16,  # Old style
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

        # Try to do like returnn frontend
        posemb_config = ConformerPosEmbConfig(
            learnable_pos_emb=False,
            rel_pos_clip=16,
            with_linear_pos=True,
            with_pos_bias=True,
            separate_pos_emb_per_head=True,
            pos_emb_dropout=0.0,
        )

        decoder_attention_zeineldeen_cfg = AdditiveAttentionConfig(
            attention_dim=1024, att_weights_dropout=0.0
        )
        decoder_zeineldeen_cfg = AttentionLSTMDecoderV1Config(
            encoder_dim=512,
            vocab_size=vocab_size_without_blank,
            target_embed_dim=640,
            target_embed_dropout=0.1,
            lstm_hidden_size=1024,
            zoneout_drop_h=0.05,
            zoneout_drop_c=0.15,
            attention_cfg=decoder_attention_zeineldeen_cfg,
            output_proj_dim=1024,
            output_dropout=0.3,
        )

        model_zeineldeen_config = ModelConfig(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config_full,
            decoder_config=decoder_zeineldeen_cfg,
            label_target_size=vocab_size_without_blank,
            pos_emb_config=posemb_config,
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
            ctc_softmax_dropout=0.1,
            encoder_out_dropout=0.1,
            specauc_start_epoch=21,  # CTC converges < 10, but AED needs longer, so 21 is a safe choice for now
            dropout_broadcast_axes=None,  # No dropout broadcast yet to properly compare
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=[11],  # 0 based
            aux_ctc_loss_scales=[0.7],
            label_smoothing=0.1,
            label_smoothing_start_epoch=31,
        )

        greedy_decoder_config = GreedyDecoderConfig(
            returnn_vocab=label_datastream_bpe.vocab,
        )

        KEEP = [300, 400, 500, 600, 700, 800, 900, 950, 980]
        train_config_24gbgpu_amp_july_baseline = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates": list(np.linspace(5e-5, 5e-5, 20)) + list(
            np.linspace(5e-4, 7e-4, 460)) + list(np.linspace(7e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40)),
            #############
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "gradient_clip_norm": 1.0,
            "torch_amp_options": {"dtype": "bfloat16"},
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": KEEP
            }
        }

        # longer training with label smoothing
        network_module = "aed.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_LSTMDecoder_v2_zero_forced_context"
        train_args = {
            "config": train_config_24gbgpu_amp_july_baseline,
            "post_config": {"num_workers_per_gpu": 4},
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_zeineldeen_config)},
            "debug": True,
        }
        
        training_name = prefix_name + "/" + network_module + ".512dim_sub6_work4_100eps"
        train_job = training(training_name, train_data_bpe, train_args, num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        #train_job.hold()
        #train_job.move_to_hpc = True

        for keep in KEEP + [1000]:
            asr_model = prepare_asr_model(
                training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe,
                get_specific_checkpoint=keep,
            )
            greedy_search_helper(
                training_name + "/keep_%i" % keep,
                asr_model=asr_model,
                decoder_config=greedy_decoder_config
            )

        beam_search_prototype(
            training_name,
            asr_model=asr_model,
            decoder_config=bs_decoder_config,
            use_gpu=True,
        )

        from ...pytorch_networks.aed.decoder.beam_search_single_v1_with_zero_ilm import \
            DecoderConfig as BeamSearchZeroLmDecoderConfig

        def lm_search(training_name, asr_model):
            neural_lm = get_lm_model("2x2024_kazuki_lstmlm_2ep")
            for beam_size in [12]:
                for lm_scale in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    for ilm_scale in [0.0, 0.1, 0.2, 0.3]:
                        asr_model_zero_forced = copy.deepcopy(asr_model)
                        asr_model_zero_forced.network_module = "aed.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_LSTMDecoder_v2_zero_forced_context"
                        beam_search_with_lm_prototype(
                            training_name + "/zeroilm_zero_force/lm%.2f_ilm%.2f_bs%i" % (lm_scale, ilm_scale, beam_size),
                            asr_model=asr_model_zero_forced,
                            decoder_config=BeamSearchZeroLmDecoderConfig(
                                returnn_vocab=label_datastream_bpe.vocab,
                                beam_search_opts=BeamSearchOpts(
                                    beam_size=beam_size,
                                    length_normalization_exponent=1.0,
                                    length_reward=0.0,
                                    bos_label=0,
                                    eos_label=0,
                                    num_labels=label_datastream_bpe.vocab_size,
                                ),
                                lm_module="lm.lstm.kazuki_lstm_zijian_variant_v2",
                                lm_args=neural_lm.net_args,
                                lm_checkpoint=neural_lm.checkpoint,
                                lm_scale=lm_scale,
                                zero_ilm_scale=ilm_scale,
                            ),
                            use_gpu=True,
                            decoder_module="aed.decoder.beam_search_single_v1_with_zero_ilm_v2"
                        )

        if BPE_SIZE == 10000:
            lm_search(training_name, asr_model)

        # with corrected LR
        if BPE_SIZE == 10000:
            train_args_correct_lr = copy.deepcopy(train_args)
            train_args_correct_lr["config"]["learning_rates"] = list(np.linspace(5e-5, 5e-5, 20)) + list(
                np.linspace(5e-5, 7e-4, 460)) + list(np.linspace(7e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40))
            training_name = prefix_name + "/" + network_module + ".512dim_sub6_work4_100eps_lrfix"
            train_job = training(training_name, train_data_bpe, train_args_correct_lr, num_epochs=1000, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48
            #train_job.hold()
            #train_job.move_to_hpc = True
            asr_model = prepare_asr_model(training_name, train_job, train_args_correct_lr, with_prior=False, datasets=train_data_bpe, get_specific_checkpoint=1000)
            beam_search_prototype(training_name, asr_model=asr_model, decoder_config=bs_decoder_config, use_gpu=True)
            lm_search(training_name, asr_model)

            model_args_no_encoder_drop = copy.deepcopy(model_zeineldeen_config)
            model_args_no_encoder_drop.encoder_out_dropout = 0.0
            model_args_no_encoder_drop.ctc_softmax_dropout = 0.0
            train_args_noencdrop = copy.deepcopy(train_args_correct_lr)
            train_args_noencdrop["net_args"] = {"model_config_dict": asdict(model_args_no_encoder_drop)}
            training_name = prefix_name + "/" + network_module + ".512dim_sub6_work4_100eps_noencdrop"
            train_job = training(training_name, train_data_bpe, train_args_noencdrop, num_epochs=1000, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48
            #train_job.hold()
            #train_job.move_to_hpc = True
            asr_model = prepare_asr_model(training_name, train_job, train_args_noencdrop, with_prior=False, datasets=train_data_bpe, get_specific_checkpoint=1000)
            beam_search_prototype(training_name, asr_model=asr_model, decoder_config=bs_decoder_config, use_gpu=True)
            lm_search(training_name, asr_model)

            # No decoder zoneout
            model_args_no_zoneout = copy.deepcopy(model_args_no_encoder_drop)
            model_args_no_zoneout.decoder_config.zoneout_drop_h = 0.0
            model_args_no_zoneout.decoder_config.zoneout_drop_c = 0.0
            train_args_no_zoneout = copy.deepcopy(train_args_correct_lr)
            train_args_no_zoneout["net_args"] = {"model_config_dict": asdict(model_args_no_zoneout)}
            training_name = prefix_name + "/" + network_module + ".512dim_sub6_work4_100eps_nozoneout"
            train_job = training(training_name, train_data_bpe, train_args_no_zoneout, num_epochs=1000, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48
            asr_model = prepare_asr_model(training_name, train_job, train_args_noencdrop, with_prior=False, datasets=train_data_bpe, get_specific_checkpoint=1000)
            beam_search_prototype(training_name, asr_model=asr_model, decoder_config=bs_decoder_config, use_gpu=True)
            lm_search(training_name, asr_model)



