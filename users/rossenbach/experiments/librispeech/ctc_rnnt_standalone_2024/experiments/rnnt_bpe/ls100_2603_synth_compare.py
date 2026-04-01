from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import synthetic_ogg_zip_data

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import get_ctc_model, add_rnnt_model
from ... import PACKAGE


def rnnt_bpe_ls100_synthcompare():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls100_rnnt_bpe"

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
        return search_jobs, wers

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

    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        emebdding_dropout=0.2,
        num_lstm_layers=1,
        lstm_hidden_dim=512,
        lstm_dropout=0.2,
    )


    for BPE_SIZE in [512]:
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-clean-100",
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

        model_config = ModelConfig(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config,
            pos_emb_config=posemb_config,
            predictor_config=predictor_config,
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
            specauc_start_epoch=11,
            joiner_dim=640,
            joiner_activation="relu",
            joiner_dropout=0.2,
            dropout_broadcast_axes=None,  # No dropout broadcast yet to properly compare
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=[11],
            aux_ctc_loss_scales=[0.3],
        )

        network_module = "rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
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

        train_args = {
            "config": train_config_24gbgpu,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "include_native_ops": True,
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = prefix_name + "/" + str(
            BPE_SIZE) + "/" + network_module + ".384dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec11_sp_morel2"
        train_job = training(training_name, train_data_bpe,
                             train_args,
                             num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24

        asr_model = prepare_asr_model(
            training_name, train_job, train_args,
            with_prior=False,
            datasets=train_data_bpe, get_specific_checkpoint=300
        )

        for beam_size in [1, 4, 8, 12]:
            evaluate_helper(
                training_name + "/keep_%i" % 300,
                asr_model,
                decoder_config_bpeany_greedy,
                beam_size=beam_size,
                use_gpu=True,
                extra_forward_config={"batch_size": 200 * 16000},
            )

        train_args_resume = copy.deepcopy(train_args)
        train_args_resume["config"][
            "import_model_train_epoch1"] = asr_model.checkpoint  # only get checkpoint, rest should be identical

        training_name = prefix_name + "/" + str(
            BPE_SIZE) + "/" + network_module + ".384dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec11_sp_morel2_resume"
        train_job = training(training_name, train_data_bpe, train_args_resume, num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_resume, with_prior=False, datasets=train_data_bpe,
            get_specific_checkpoint=300,
        )

        synth_keys = [
            # "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_base320_fromglowbase256_400eps_gl32_syn_train-clean-360",
            "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360",
            # "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-360",
            # "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_train-clean-360",
            # "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-360",
            # "glow_tts.glow_tts_v1_glow256align_400eps_oclr_nodrop_gl32_noise0.7_syn_train-clean-360",
            # "glow_tts.glow_tts_v1_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn_train-clean-360",
            # "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-360",
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
            train_job.rqmt["gpu_mem"] = 48
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_resume, with_prior=False, datasets=train_data_bpe,
                get_specific_checkpoint=300,
            )