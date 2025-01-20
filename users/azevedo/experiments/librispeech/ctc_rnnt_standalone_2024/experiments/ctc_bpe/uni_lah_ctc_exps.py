"""
WARNING: This baseline is highly outdated, updates coming soon
"""
import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import add_ctc_model



def bpe_ls960_1023_uni_lah(chunk_size: float, lookahead_size: int, kernel_size: int):
    prefix_name = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_low_bpe"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe128 = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=128,
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe128 = cast(LabelDatastream, train_data_bpe128.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe128.vocab_size

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

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig as DecoderConfigOffline
    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig
    # from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_lah_carryover_decoder import DecoderConfig as GreedyDecoderConfig

    def tune_and_evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        lm_scales: List[float],
        prior_scales: List[float],
        decoder_module: str = "ctc.decoder.flashlight_ctc_v1"
    ):
        """
        Example helper to execute tuning over lm_scales and prior scales.
        With the best values runs test-clean and test-other.

        This is just a reference helper and can (should) be freely changed, copied, modified etc...

        :param training_name: for alias and output names
        :param asr_model: ASR model to use
        :param base_decoder_config: any decoder config dataclass
        :param lm_scales: lm scales for tuning
        :param prior_scales: prior scales for tuning, same length as lm scales
        """
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
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn,
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config={},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn,
            )
    
    def greedy_search_helper(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: GreedyDecoderConfig,
            decoder_module: str,
    ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module=decoder_module,
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples=dev_dataset_tuples,
            **default_returnn,
        )

    default_decoder_config_bpe128 = DecoderConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=128),
        returnn_vocab=label_datastream_bpe128.vocab,
        beam_size=1024,  # Untuned
        beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,  # Untuned
        left_size=int(chunk_size * 16e3),
        right_size=int(chunk_size * 16e3),
        test_version=0.1,
    )

    offline_decoder_config_bpe128 = DecoderConfigOffline(
        lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=128),
        returnn_vocab=label_datastream_bpe128.vocab,
        beam_size=1024,  # Untuned
        beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,  # Untuned
    )

    

    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        ModelConfig as ModelConfigOffline
    )

    from ...pytorch_networks.ctc.conformer_1023.unified_lah_conformer_cfg import ModelConfig

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
    frontend_config_sub4 = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )

    #
    # Unified model
    #
    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
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
        conv_kernel_size=kernel_size,
        final_dropout=0.1,
        specauc_start_epoch=11,  # BPE does not converge otherwise
        chunk_size=chunk_size * 16e3,
        lookahead_size=lookahead_size,
        online_model_scale=0.5,
    )

    keep = [300, 400, 500, 600, 700, 800, 900, 950, 980, 1000]
    accum_grads = 1
    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 480))
        + list(np.linspace(5e-4, 5e-5, 480))
        + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 240 * 16000 // accum_grads,  # GPU MEM still very moderate, but larger batch did not help
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": accum_grads,
        "torch_amp_options": {"dtype": "bfloat16"},
        "gradient_clip_norm": 1.0,
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
            "keep": keep
        }
    }

    network_module = "ctc.conformer_1023.unified_lah_conformer"
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }

    training_name = (
        prefix_name + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_uni_lah" +
        "/" + str(chunk_size) + "/" + "conv%d" % kernel_size +
        "/" + "lah%d" % lookahead_size
    )
    train_job = training(training_name, train_data_bpe128, train_args, num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe128, get_specific_checkpoint=1000
    )
    #add_ctc_model("ls960_ctc_bpe_5k." + network_module + ".512dim_sub6_11gbgpu_50eps_ckpt500", asr_model)
    # online
    tune_and_evaluate_helper(
        training_name + "/online",
        asr_model,
        default_decoder_config_bpe128,
        lm_scales=[1.6, 1.8, 2.0],
        prior_scales=[0.2, 0.3, 0.4],
        decoder_module="ctc.decoder.flashlight_ctc_v2"
    )

    # offline
    default_decoder_config_bpe128_off = copy.deepcopy(default_decoder_config_bpe128)
    default_decoder_config_bpe128_off.left_size = None
    default_decoder_config_bpe128_off.right_size = None
    default_decoder_config_bpe128_off.test_version = 0.0
    tune_and_evaluate_helper(
        training_name + "/offline",
        asr_model,
        default_decoder_config_bpe128_off,
        lm_scales=[1.6, 1.8, 2.0],
        prior_scales=[0.2, 0.3, 0.4],
        decoder_module="ctc.decoder.flashlight_ctc_v2"
    )

    #
    # greedy
    #
    from ...pytorch_networks.ctc.conformer_1124.model_lah_carryover_cfg import ModelConfig as ModelConfigNew
    from ...pytorch_networks.rnnt.auxil.functional import TrainingStrategy
    model_config_new = ModelConfigNew(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
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
        conv_kernel_size=kernel_size,
        final_dropout=0.1,
        specauc_start_epoch=11,  # BPE does not converge otherwise

        chunk_size=model_config.chunk_size,
        lookahead_size=model_config.lookahead_size,
        online_model_scale=model_config.online_model_scale,
        carry_over_size=1,
        training_strategy=str(TrainingStrategy.STREAMING),
    )

    asr_model_new = copy.deepcopy(asr_model)
    asr_model_new.net_args = {"model_config_dict": asdict(model_config_new)}
    asr_model_new.network_module = "ctc.conformer_1124.model_streaming_lah_carryover"

    online_decoder_greedy = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe128.vocab,

        chunk_size=int(model_config.chunk_size),
        lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
        carry_over_size=1,
        test_version=0.0,
    )
    offline_decoder_greedy = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe128.vocab,

        chunk_size=None,
        lookahead_size=None,
        carry_over_size=None,
        test_version=0.0,
    )
    greedy_search_helper(
        training_name + "/online",
        asr_model_new,
        online_decoder_greedy,
        decoder_module="ctc.decoder.greedy_lah_carryover_decoder"
    )
    greedy_search_helper(
        training_name + "/offline",
        asr_model_new,
        offline_decoder_greedy,
        decoder_module="ctc.decoder.greedy_lah_carryover_decoder"
    )

    #
    # Baseline
    #
    model_config_offline = ModelConfigOffline(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
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
        conv_kernel_size=kernel_size,
        final_dropout=0.1,
        specauc_start_epoch=11,
    )

    train_args_offline = copy.deepcopy(train_args)
    train_args_offline["network_module"] = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args_offline["net_args"] = {"model_config_dict": asdict(model_config_offline)}

    training_name_offline = prefix_name + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_base"
    train_job = training(training_name_offline, train_data_bpe128, train_args_offline, num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model_offline = prepare_asr_model(
        training_name_offline, train_job, train_args_offline, with_prior=True, datasets=train_data_bpe128, get_specific_checkpoint=1000
    )
    
    tune_and_evaluate_helper(
        training_name_offline,
        asr_model_offline,
        offline_decoder_config_bpe128,
        lm_scales=[1.6, 1.8, 2.0],
        prior_scales=[0.2, 0.3, 0.4],
    )

    frontend_config_sub6 = VGG4LayerActFrontendV1Config_mod(
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
    model_config_offline = ModelConfigOffline(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
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
        conv_kernel_size=kernel_size,
        final_dropout=0.1,
        specauc_start_epoch=11,
    )

    train_args_offline = copy.deepcopy(train_args)
    train_args_offline["network_module"] = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args_offline["net_args"] = {"model_config_dict": asdict(model_config_offline)}

    training_name_offline = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_100eps_base"
    train_job = training(training_name_offline, train_data_bpe128, train_args_offline, num_epochs=1000,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model_offline = prepare_asr_model(
        training_name_offline, train_job, train_args_offline, with_prior=True, datasets=train_data_bpe128,
        get_specific_checkpoint=1000
    )

    tune_and_evaluate_helper(
        training_name_offline,
        asr_model_offline,
        offline_decoder_config_bpe128,
        lm_scales=[1.6, 1.8, 2.0],
        prior_scales=[0.2, 0.3, 0.4],
    )

    #
    # relpos baseline
    #


    from ...pytorch_networks.ctc.conformer_0125.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import (
        ConformerPosEmbConfig,
        ModelConfig as ModelConfigRelpos
    )

    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    model_config_relpos = ModelConfigRelpos(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
        pos_emb_config=posemb_config,
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
        mhsa_with_bias=True,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=11,
        dropout_broadcast_axes=None, # No dropout broadcast yet to properly compare
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
    )

    prefix_name_rel = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_bpe_relposencoder_0924"
    train_args_relpos = copy.deepcopy(train_args_offline)
    train_args_relpos["network_module"] = "ctc.conformer_0125.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    train_args_relpos["net_args"] = {"model_config_dict": asdict(model_config_relpos)}

    training_name_relpos = prefix_name_rel + "/" + train_args_relpos["network_module"] + ".512dim_sub6_48gbgpu_100eps_relpos_base"
    train_job = training(training_name_relpos, train_data_bpe128, train_args_relpos, num_epochs=1000,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    asr_model_relpos = prepare_asr_model(
        training_name_relpos, train_job, train_args_relpos, with_prior=True, datasets=train_data_bpe128,
        get_specific_checkpoint=1000
    )

    tune_and_evaluate_helper(
        training_name_relpos,
        asr_model_relpos,
        offline_decoder_config_bpe128,
        lm_scales=[1.6, 1.8, 2.0],
        prior_scales=[0.2, 0.3, 0.4],
    )