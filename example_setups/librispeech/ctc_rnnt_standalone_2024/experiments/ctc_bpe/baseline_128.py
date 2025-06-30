"""
Modern baseline for June 2025
"""
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
from ...pipeline import training, prepare_asr_model, search, ASRModel, NeuralLM
from ...report import tune_and_evalue_report
from ...storage import get_lm_model
from ... import PACKAGE


def bpe128_ls960_0924_base():
    prefix_name = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_bpe_128"

    BPE_SIZE = 128

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

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

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig as FlashlightDecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig
    from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v5 import DecoderConfig as BeamSearchDecoderConfig
    from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v5 import (
        DecoderExtraConfig as BeamSearchDecoderExtraConfig,
    )

    DecoderConfig = FlashlightDecoderConfig | GreedyDecoderConfig | BeamSearchDecoderConfig

    def tune_and_evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        lm_scales: List[float],
        prior_scales: List[float],
        unhashed_decoder_config: Optional[BeamSearchDecoderExtraConfig] = None,
        extra_forward_config=None,
        use_gpu=False,
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
        :param unhashed_decoder_config: decoder config without hashing, used for BeamSearch configs
        :param extra_forward_config: additional args to the ReturnnForwardJob, e.g. to modify batch size
        :param use_gpu: for GPU decoding
        """

        # Automatic selection of decoder module
        if isinstance(base_decoder_config, FlashlightDecoderConfig):
            decoder_module = "ctc.decoder.flashlight_ctc_v2"
        elif isinstance(base_decoder_config, BeamSearchDecoderConfig):
            decoder_module = "ctc.decoder.beam_search_bpe_ctc_v5"
            assert unhashed_decoder_config is not None
        else:
            assert False, "Invalid decoder config"

        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
        for lm_scale in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_scale = lm_scale
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.2f_prior%.2f" % (lm_scale, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config=extra_forward_config if extra_forward_config else {},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)}
                    if unhashed_decoder_config
                    else None,
                    use_gpu=use_gpu,
                    **default_returnn,
                )
                tune_parameters.append((lm_scale, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_scale = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config=extra_forward_config if extra_forward_config else {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)}
                if unhashed_decoder_config
                else None,
                use_gpu=use_gpu,
                **default_returnn,
            )
            report_values[key] = wers[training_name + "/" + key]

        tune_and_evalue_report(
            training_name=training_name,
            tune_parameters=tune_parameters,
            tuning_names=["LM", "Prior"],
            tune_values_clean=tune_values_clean,
            tune_values_other=tune_values_other,
            report_values=report_values,
        )

    def greedy_search_helper(training_name: str, asr_model: ASRModel, decoder_config: GreedyDecoderConfig):
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
            test_dataset_tuples=dev_dataset_tuples,
            **default_returnn,
        )

    default_flashlight_decoder_config = FlashlightDecoderConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=BPE_SIZE),
        returnn_vocab=label_datastream_bpe.vocab,
        beam_size=1024,  # Untuned
        beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,  # Untuned
    )

    default_greedy_config = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe.vocab,
    )

    trafo_32x768: NeuralLM = get_lm_model("bpe%i_trafo32x768_5ep" % BPE_SIZE)
    lstm_2x2048: NeuralLM = get_lm_model("bpe%i_2x2024_kazuki_lstmlm_3ep" % BPE_SIZE)

    lstmlm_beamsearch_decoder_bs10_config = BeamSearchDecoderConfig(
        returnn_vocab=label_datastream_bpe.vocab,
        beam_size=10,
        lm_model_args=lstm_2x2048.net_args,
        lm_checkpoint=lstm_2x2048.checkpoint,
        lm_module="pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v1_decoding.Model",
        lm_states_need_label_axis=False,
    )
    lstmlm_beamsearch_decoder_bs32_config = copy.deepcopy(lstmlm_beamsearch_decoder_bs10_config)
    lstmlm_beamsearch_decoder_bs32_config.beam_size = 32

    trafolm_beamsearch_decoder_config = BeamSearchDecoderConfig(
        returnn_vocab=label_datastream_bpe.vocab,
        beam_size=10,
        lm_model_args=trafo_32x768.net_args,
        lm_checkpoint=trafo_32x768.checkpoint,
        lm_module="pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v1_decoding.Model",
        lm_states_need_label_axis=True,
    )

    beamsearch_decoder_extra_config = BeamSearchDecoderExtraConfig(
        lm_package=PACKAGE,
    )

    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
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
        max_dim_feat=16,  # classic style
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

    # Try to do like returnn frontend
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
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
        dropout_broadcast_axes=None,  # No dropout broadcast yet to properly compare
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
    )

    # New test more closer to other setup
    train_config_24gbgpu_amp_radam = {
        "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
        "learning_rates": list(np.linspace(5e-5, 5e-4, 480))
        + list(np.linspace(5e-4, 5e-5, 480))
        + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 240 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
        "num_workers_per_gpu": 2,
    }

    network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    train_args_radam = {
        "config": train_config_24gbgpu_amp_radam,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "use_speed_perturbation": True,
        "debug": False,
    }

    training_name = (
        prefix_name
        + "/"
        + str(BPE_SIZE)
        + "/"
        + network_module
        + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_radam_lr5e-4"
    )
    train_job = training(training_name, train_data_bpe, train_args_radam, num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48

    asr_model = prepare_asr_model(
        training_name,
        train_job,
        train_args_radam,
        with_prior=True,
        datasets=train_data_bpe,
        get_specific_checkpoint=1000,
    )
    greedy_search_helper(training_name + "/greedy", asr_model, default_greedy_config)
    tune_and_evaluate_helper(
        training_name + "/flashlight_4gram",
        asr_model,
        default_flashlight_decoder_config,
        lm_scales=[1.6, 1.8, 2.0],
        prior_scales=[0.2, 0.3, 0.4],
    )
    tune_and_evaluate_helper(
        training_name + "/beamsearch_lstm_2x2048_bs10",
        asr_model,
        lstmlm_beamsearch_decoder_bs10_config,
        lm_scales=[0.6, 0.65, 0.7, 0.75, 0.8],
        prior_scales=[0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
        unhashed_decoder_config=beamsearch_decoder_extra_config,
        extra_forward_config={"batch_size": 200 * 16000},
        use_gpu=True,
    )
    tune_and_evaluate_helper(
        training_name + "/beamsearch_lstm_2x2048_bs32",
        asr_model,
        lstmlm_beamsearch_decoder_bs32_config,
        lm_scales=[0.6, 0.65, 0.7, 0.75, 0.8],
        prior_scales=[0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
        unhashed_decoder_config=beamsearch_decoder_extra_config,
        extra_forward_config={"batch_size": 200 * 16000},
        use_gpu=True,
    )
    tune_and_evaluate_helper(
        training_name + "/beamsearch_trafo_32x768",
        asr_model,
        trafolm_beamsearch_decoder_config,
        lm_scales=[0.7, 0.75, 0.8, 0.85, 0.9],
        prior_scales=[0.3, 0.35, 0.4],
        unhashed_decoder_config=beamsearch_decoder_extra_config,
        extra_forward_config={"batch_size": 200 * 16000},
        use_gpu=True,
    )
