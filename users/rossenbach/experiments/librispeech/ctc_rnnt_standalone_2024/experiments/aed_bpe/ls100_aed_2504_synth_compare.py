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
from ...storage import get_synthetic_data


def aed_bpe_ls100_2504_synth_compare():
    ext_prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls100_aed/2504_low_bpe"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=3,
        train_seq_ordering="laplace:.2000",  # laplace subsampled by num workers
    )
    
    def get_training_settings_with_partition(partition):
        return DatasetSettings(
            preemphasis=0.97,  # TODO: Check if this is really useful
            peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
            # training
            train_partition_epoch=partition,
            train_seq_ordering="laplace:.2000",
        )

    for BPE_SIZE in [2000]:
        prefix_name = ext_prefix_name + "/bpe_%i" % BPE_SIZE
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-clean-100",
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

        KEEP = [100, 150, 200, 250, 300]
        train_config_24gbgpu_amp_july_baseline = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates": list(np.linspace(5e-5, 5e-5, 10)) + list(
            np.linspace(5e-4, 7e-4, 140)) + list(np.linspace(7e-4, 5e-5, 140)) + list(np.linspace(5e-5, 1e-7, 10)),
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

        base_name = prefix_name + "/" + network_module
        training_name = base_name + ".512dim_sub6_work4_100eps"
        train_job = training(training_name, train_data_bpe, train_args, num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        #train_job.hold()
        #train_job.move_to_hpc = True

        for keep in KEEP:
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

        # Take over data from JAIST experiment, combined training
        from i6_experiments.users.rossenbach.experiments.jaist_project.storage import synthetic_ogg_zip_data

        train_args_resume = copy.deepcopy(train_args)
        train_args_resume["config"][
            "import_model_train_epoch1"] = asr_model.checkpoint  # only get checkpoint, rest should be identical
        
        training_name = base_name + ".512dim_sub6_work4_100eps_resume"
        train_job = training(training_name, train_data_bpe, train_args_resume, num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_resume, with_prior=False, datasets=train_data_bpe,
            get_specific_checkpoint=300,
        )
        beam_search_prototype(
            training_name,
            asr_model=asr_model,
            decoder_config=bs_decoder_config,
            use_gpu=True,
        )

        synth_keys = [
            "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_base320_fromglowbase256_400eps_gl32_syn_train-clean-360",
            "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360",
            "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-360",
            "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_train-clean-360",
            "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-360",
            # "glow_tts.glow_tts_v1_glow256align_400eps_oclr_nodrop_gl32_noise0.7_syn_train-clean-360",
            # "glow_tts.glow_tts_v1_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn_train-clean-360",
            "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-360",
            # ------------------------------------
            # "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360",
            # "glow_tts.glow_tts_v1_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn_train-clean-360",
        ]

        for synth_key in synth_keys:
            name = ".512dim_sub6_work4_100eps/combine_3to1_synth_partition12/" + synth_key
            synth_ogg = synthetic_ogg_zip_data[synth_key]  # bliss and zip, so take zip
            train_data_bpe = build_bpe_training_datasets(
                prefix=prefix_name,
                librispeech_key="train-clean-100",
                bpe_size=BPE_SIZE,
                settings=train_settings,
                use_postfix=True,  # AED, use postfix
                extra_train_ogg_zips=[synth_ogg],
                data_repetition_factors=[3, 1], ## 3x original + 1x synth
            )

            training_name = base_name + name
            train_job = training(training_name, train_data_bpe, train_args_resume, num_epochs=300, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_resume, with_prior=False, datasets=train_data_bpe,
                get_specific_checkpoint=300,
            )
            greedy_search_helper(
                training_name + "/keep_%i" % 300,
                asr_model=asr_model,
                decoder_config=greedy_decoder_config,
                use_gpu=True
            )
            beam_search_prototype(
                training_name,
                asr_model=asr_model,
                decoder_config=bs_decoder_config,
                use_gpu=True,
            )

        # do it again with correct partition

        for synth_key in synth_keys:
            name = ".512dim_sub6_work4_100eps/combine_3to1_synth_partition3/" + synth_key
            synth_ogg = synthetic_ogg_zip_data[synth_key]  # bliss and zip, so take zip
            train_data_bpe = build_bpe_training_datasets(
                prefix=prefix_name,
                librispeech_key="train-clean-100",
                bpe_size=BPE_SIZE,
                settings=get_training_settings_with_partition(12),
                use_postfix=True,  # AED, use postfix
                extra_train_ogg_zips=[synth_ogg],
                data_repetition_factors=[3, 1], ## 3x original + 1x synth
            )

            training_name = base_name + name
            train_job = training(training_name, train_data_bpe, train_args_resume, num_epochs=300, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48 # 24gb was full
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_resume, with_prior=False, datasets=train_data_bpe,
                get_specific_checkpoint=300,
            )
            greedy_search_helper(
                training_name + "/keep_%i" % 300,
                asr_model=asr_model,
                decoder_config=greedy_decoder_config,
                use_gpu=True
            )
            beam_search_prototype(
                training_name,
                asr_model=asr_model,
                decoder_config=bs_decoder_config,
                use_gpu=True,
            )

