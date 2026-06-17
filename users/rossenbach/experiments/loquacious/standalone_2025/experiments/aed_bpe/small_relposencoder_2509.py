from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset, build_short_dev_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...pipeline import training, prepare_asr_model, search, ASRModel, evaluate_all
from ...tune_eval import tune_and_evaluate_helper
from ...storage import get_lm_model, NeuralLM
from ... import PACKAGE


def aed_bpe_small():
    prefix_name = "experiments/loquacious/standalone_2025/rnnt_small_relposencoder_0925"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    short_dev_dataset_tuples = {
        "dev.short": build_short_dev_dataset(train_settings)
    }

    dev_dataset_tuples = {}
    for testset in ["dev.commonvoice", "dev.librispeech", "dev.voxpopuli", "dev.yodas"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test.commonvoice", "test.librispeech", "test.voxpopuli", "test.yodas"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.aed.decoder.beam_search_single_v1 import DecoderConfig as BeamSearchDecoderConfig, \
        BeamSearchOpts, ExtraConfig


    def evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: BeamSearchDecoderConfig,
        unhashed_decoder_config: Optional[ExtraConfig] = None,
        beam_size: int = 1,
        use_gpu=False,
        decoder_module="aed.decoder.beam_search_single_v1",
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
        decoder_config.beam_search_opts.beam_size = beam_size
        search_name = training_name + "/search_bs%i" % beam_size
        dev_search_jobs, dev_wers, dev_ctms = search(
            search_name,
            forward_config= {"seed": 2, **extra_forward_config} if use_gpu else {**extra_forward_config},
            asr_model=asr_model,
            decoder_module=decoder_module,
            decoder_args={"config": asdict(decoder_config)},
            unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
            test_dataset_tuples=dev_dataset_tuples,
            use_gpu=use_gpu,
            debug=debug,
            **default_returnn,
        )
        if with_test:
            test_search_jobs, test_wers, test_ctms = search(
                search_name,
                forward_config= {"seed": 2, **extra_forward_config} if use_gpu else {**extra_forward_config},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
                test_dataset_tuples=test_dataset_tuples,
                use_gpu=use_gpu,
                debug=debug,
                **default_returnn,
            )
        else:
            test_ctms = None
            test_search_jobs = {}
            test_wers = {}

        evaluate_all(search_name, dev_ctms, test_ctms)

        return {**dev_search_jobs, **test_search_jobs}, {**dev_wers, **test_wers}

    from ...pytorch_networks.aed.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_LSTMDecoder_v2_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        AttentionLSTMDecoderV1Config,
        AdditiveAttentionConfig,
        ConformerPosEmbConfig
    )
    from ...pytorch_networks.aed.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_LSTMDecoder_v2_cfg import \
        ModelConfig

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
    decoder_attention_zeineldeen_cfg = AdditiveAttentionConfig(
        attention_dim=1024, att_weights_dropout=0.0
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

    
    for BPE_SIZE in [1000]:
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            loquacious_key="train.small",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=True,  # RNN-T now, use postfix
        )
        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

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
            specaug_config=specaug_config,
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
            aux_ctc_loss_scales=[1.0],
            label_smoothing=0.1,
            label_smoothing_start_epoch=31,
        )



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

        network_module = "aed.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_LSTMDecoder_v2_zero_forced_context"
        train_config_24gbgpu_amp_radam = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates": list(np.linspace(7e-6, 5e-4, 240)) + list(
                np.linspace(5e-4, 5e-5, 240)) + list(np.linspace(5e-5, 1e-7, 20)),
            #############
            "batch_size": 240 * 16000,
            "gradient_clip_norm": 1.0,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "torch_amp_options": {"dtype": "bfloat16"},
        }
        train_args_radam = {
            "config": train_config_24gbgpu_amp_radam,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_zeineldeen_config)},
            "include_native_ops": True,
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = prefix_name + "/" + str(
            BPE_SIZE) + "/" + network_module + ".512dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec21_sp_morel2"
        train_job = training(training_name, train_data_bpe,
                             train_args_radam,
                             num_epochs=500, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        #train_job.hold()
        #train_job.move_to_hpc = True

        asr_model = prepare_asr_model(
            training_name, train_job, train_args_radam,
            with_prior=False,
            datasets=train_data_bpe, get_specific_checkpoint=500
        )

        for beam_size in [1, 4, 8, 12]:
            evaluate_helper(
                training_name + "/keep_%i" % 500,
                asr_model,
                bs_decoder_config,
                beam_size=beam_size,
                use_gpu=True,
            )

