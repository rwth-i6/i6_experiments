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
from ...storage import get_ctc_model, get_lm_model, NeuralLM


def aed_bpe_ls100_0824_base():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls100_aed_bpe_low_vocab"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=3,
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

    from ...pytorch_networks.aed.decoder.greedy_prototype import DecoderConfig as GreedyDecoderConfig
    from ...pytorch_networks.aed.decoder.beam_search_single_v1 import DecoderConfig as BeamSearchDecoderConfig, BeamSearchOpts

    from ...pytorch_networks.aed.decoder.beam_search_single_v1_with_lm import \
        DecoderConfig as BeamSearchLmDecoderConfig

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
    ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_bs"
        search_jobs, wers = search(
            search_name,
            forward_config={"max_seqs": 20} if seed is None else {"max_seqs": 20, "seed": seed},
            asr_model=asr_model,
            decoder_module="aed.decoder.beam_search_single_v1",
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
    ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_bs"
        search_jobs, wers = search(
            search_name,
            forward_config={"max_seqs": 20} if seed is None else {"max_seqs": 20, "seed": seed},
            asr_model=asr_model,
            decoder_module="aed.decoder.beam_search_single_v1_with_lm",
            decoder_args={"config": asdict(decoder_config)},
            use_gpu=use_gpu,
            debug=True,
            test_dataset_tuples={**dev_dataset_tuples} if dev_only else {**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )
    

    from ...pytorch_networks.aed.conformer_zoneout_prototype_0624.aed_prototype_v4_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        AttentionLSTMDecoderV1Config,
        AdditiveAttentionConfig
    )

    from ...pytorch_networks.aed.conformer_zoneout_prototype_0624.aed_prototype_v4_cfg import ModelConfig as ModelConfigV4

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
        out_features=384,
        activation=None,
    )

    decoder_attention_zeineldeen_cfg = AdditiveAttentionConfig(
        attention_dim=1024, att_weights_dropout=0.2
    )

    # no vocab size set yet
    decoder_zeineldeen_cfg_base = AttentionLSTMDecoderV1Config(
        encoder_dim=384,
        target_embed_dim=480,
        target_embed_dropout=0.2,
        lstm_hidden_size=768,
        zoneout_drop_h=0.05,
        zoneout_drop_c=0.15,
        attention_cfg=decoder_attention_zeineldeen_cfg,
        output_proj_dim=768,
        output_dropout=0.3,
        vocab_size=None,  # filled later
    )

    model_zeineldeen_config_v4_base = ModelConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        decoder_config=None,  # filled later
        label_target_size=None,  # filled later
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
        specauc_start_epoch=21,  # CTC converges < 10, but AED needs longer, so 21 is a safe choice for now
        ctc_loss_scale=0.7,  # quite strong but needed for convergence for now (0.7 worked, 0.3 did not, TODO: test 0.5)
        label_smoothing=0.05,
        label_smoothing_start_epoch=31,
    )

    for BPE_SIZE in [128, 512, 1024]:
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

        decoder_zeineldeen_cfg = copy.deepcopy(decoder_zeineldeen_cfg_base)
        decoder_zeineldeen_cfg.vocab_size = vocab_size_without_blank

        model_zeineldeen_config_v4 = copy.deepcopy(model_zeineldeen_config_v4_base)
        model_zeineldeen_config_v4.decoder_config = decoder_zeineldeen_cfg
        model_zeineldeen_config_v4.label_target_size = vocab_size_without_blank

        greedy_decoder_config = GreedyDecoderConfig(
            returnn_vocab=label_datastream_bpe.vocab,
        )

        train_config_11gbgpu_short_baseline = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates": list(np.linspace(5e-5, 5e-5, 20)) + list(
            np.linspace(5e-5, 5e-4, 130)) + list(np.linspace(5e-4, 5e-5, 130)) + list(np.linspace(5e-5, 1e-7, 20)),
            #############
            "batch_size": 240 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "gradient_clip_norm": 1.0,
            "use_speed_perturbation": True,
        }

        network_module = "aed.conformer_zoneout_prototype_0624.aed_prototype_v4"
        train_args = {
            "config": train_config_11gbgpu_short_baseline,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_zeineldeen_config_v4)},
            "debug": False,
        }

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".384dim_sub6_11gbgpu_100eps_radam"
        train_job = training(training_name, train_data_bpe, train_args, num_epochs=300, **default_returnn)
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe,
            get_specific_checkpoint=300
        )
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

        beam_search_prototype(
            training_name,
            asr_model=asr_model,
            decoder_config=BeamSearchDecoderConfig(
                returnn_vocab=label_datastream_bpe.vocab,
                beam_search_opts=BeamSearchOpts(
                    beam_size=12,
                    length_normalization_exponent=1.0,
                    length_reward=0,
                    bos_label=0,
                    eos_label=0,
                    num_labels=label_datastream_bpe.vocab_size
                )
            ),
            use_gpu=True,
        )
