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


def aed_bpe_ls960_1023_base():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_aed_0824_low_bpe"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe5000 = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=5000,
        settings=train_settings,
        use_postfix=True,  # RNN-T now, use postfix
    )
    label_datastream_bpe5000 = cast(LabelDatastream, train_data_bpe5000.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe5000.vocab_size

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
    from ...pytorch_networks.aed.decoder.ctc_greedy_only_v1 import DecoderConfig as CtcGreedyDecoderConfig
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
    

    from ...pytorch_networks.aed.conformer_zoneout_prototype_0624.aed_prototype_v1_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        AttentionLSTMDecoderV1Config,
        AdditiveAttentionConfig
    )

    from ...pytorch_networks.aed.conformer_zoneout_prototype_0624.aed_prototype_v2_cfg import ModelConfig as ModelConfigV2
    from ...pytorch_networks.aed.conformer_zoneout_prototype_0624.aed_prototype_v3_cfg import ModelConfig as ModelConfigV3
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
        out_features=512,
        activation=None,
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
    model_zeineldeen_config_v2 = ModelConfigV2(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        decoder_config=decoder_zeineldeen_cfg,
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
        specauc_start_epoch=21, # CTC converges < 10, but AED needs longer, so 21 is a safe choice for now
        ctc_loss_scale=0.7,  # quite strong but needed for convergence for now (0.7 worked, 0.3 did not, TODO: test 0.5)
    )

    model_zeineldeen_config_v3 = ModelConfigV3(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        decoder_config=decoder_zeineldeen_cfg,
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
        specauc_start_epoch=21,  # CTC converges < 10, but AED needs longer, so 21 is a safe choice for now
        ctc_loss_scale=0.7,  # quite strong but needed for convergence for now (0.7 worked, 0.3 did not, TODO: test 0.5)
        label_smoothing=0.1,
    )
    
    model_zeineldeen_config_v4 = ModelConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        decoder_config=decoder_zeineldeen_cfg,
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
        specauc_start_epoch=21,  # CTC converges < 10, but AED needs longer, so 21 is a safe choice for now
        ctc_loss_scale=0.7,  # quite strong but needed for convergence for now (0.7 worked, 0.3 did not, TODO: test 0.5)
        label_smoothing=0.1,
        label_smoothing_start_epoch=31,
    )

    greedy_decoder_config = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe5000.vocab,
    )

    ctc_greedy_decoder_config = CtcGreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe5000.vocab
    )

    train_config_24gbgpu_amp_july_baseline = {
        "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
        "learning_rates": list(np.linspace(5e-5, 5e-5, 20)) + list(
        np.linspace(5e-4, 5e-4, 460)) + list(np.linspace(5e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
        "use_speed_perturbation": True,
    }

    network_module = "aed.conformer_zoneout_prototype_0624.aed_prototype_v2"
    train_args = {
        "config": train_config_24gbgpu_amp_july_baseline,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_zeineldeen_config_v2)},
        "debug": True,
    }


    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_100eps_new_baseline"
    train_job = training(training_name, train_data_bpe5000, train_args, num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=1000
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
    beam_search_prototype(
        training_name,
        asr_model=asr_model,
        decoder_config=BeamSearchDecoderConfig(
            returnn_vocab=label_datastream_bpe5000.vocab,
            beam_search_opts=BeamSearchOpts(
                beam_size=12,
                length_normalization_exponent=1.0,
                length_reward=0,
                bos_label=0,
                eos_label=0,
                num_labels=label_datastream_bpe5000.vocab_size
            )
        ),
        use_gpu=True,
    )
    
    
    # longer training with label smoothing

    network_module = "aed.conformer_zoneout_prototype_0624.aed_prototype_v4"
    train_args = {
        "config": train_config_24gbgpu_amp_july_baseline,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_zeineldeen_config_v4)},
        "debug": True,
    }

    # long training with v4 and attention dropout

    model_zeineldeen_config_v4_attdrop = copy.deepcopy(model_zeineldeen_config_v4)
    model_zeineldeen_config_v4_attdrop.decoder_config.attention_cfg.att_weights_dropout = 0.1
    train_args_attdrop = copy.deepcopy(train_args)
    train_args_attdrop["net_args"] = {"model_config_dict": asdict(model_zeineldeen_config_v4_attdrop)}

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_100eps_new_baseline_attdrop"
    train_job = training(training_name, train_data_bpe5000, train_args_attdrop, num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args_attdrop, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=1000
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
    beam_search_prototype(
        training_name,
        asr_model=asr_model,
        decoder_config=BeamSearchDecoderConfig(
            returnn_vocab=label_datastream_bpe5000.vocab,
            beam_search_opts=BeamSearchOpts(
                beam_size=12,
                length_normalization_exponent=1.0,
                length_reward=0,
                bos_label=0,
                eos_label=0,
                num_labels=label_datastream_bpe5000.vocab_size
            )
        ),
        use_gpu=True,
    )

    neural_lm = get_lm_model("bpe5k_2x2024_kazuki_lstmlm_3ep")
    for beam_size in [12]:
        for lm_scale in [0.26, 0.28, 0.30, 0.32, 0.34]:
            beam_search_with_lm_prototype(
                training_name + "/lm3ep_%.2f_bs%i" % (lm_scale, beam_size),
                asr_model=asr_model,
                decoder_config=BeamSearchLmDecoderConfig(
                    returnn_vocab=label_datastream_bpe5000.vocab,
                    beam_search_opts=BeamSearchOpts(
                        beam_size=beam_size,
                        length_normalization_exponent=1.0,
                        length_reward=0.0,
                        bos_label=0,
                        eos_label=0,
                        num_labels=label_datastream_bpe5000.vocab_size,
                    ),
                    lm_module="lm.lstm.kazuki_lstm_zijian_variant_v2",
                    lm_args=neural_lm.net_args,
                    lm_checkpoint=neural_lm.checkpoint,
                    lm_scale=lm_scale,
                ),
                use_gpu=True,
            )
            
            




    # 11GB Runs from here

    train_config_11gbgpu_short_baseline = {
        "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
        "learning_rates": list(np.linspace(5e-5, 5e-5, 20)) + list(
        np.linspace(5e-4, 5e-4, 110)) + list(np.linspace(5e-4, 5e-5, 110)) + list(np.linspace(5e-5, 1e-7, 10)),
        #############
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "gradient_clip_norm": 1.0,
        "use_speed_perturbation": True,
    }

    network_module = "aed.conformer_zoneout_prototype_0624.aed_prototype_v2"
    train_args = {
        "config": train_config_11gbgpu_short_baseline,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_zeineldeen_config_v2)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_11gbgpu_25eps_new_baseline_run2"
    train_job = training(training_name, train_data_bpe5000, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

    beam_search_prototype(
        training_name,
        asr_model=asr_model,
        decoder_config=BeamSearchDecoderConfig(
            returnn_vocab=label_datastream_bpe5000.vocab,
            beam_search_opts=BeamSearchOpts(
                beam_size=12,
                length_normalization_exponent=1.0,
                length_reward=0,
                bos_label=0,
                eos_label=0,
                num_labels=label_datastream_bpe5000.vocab_size
            )
        ),
        use_gpu=True,
    )

    neural_lm = get_lm_model("bpe5k_2x2024_kazuki_lstmlm_3ep")
    for beam_size in [12]:
        for lm_scale in [0.26, 0.28, 0.30, 0.32, 0.34]:
            beam_search_with_lm_prototype(
                training_name + "/lm3ep_%.2f_bs%i" % (lm_scale, beam_size),
                asr_model=asr_model,
                decoder_config=BeamSearchLmDecoderConfig(
                    returnn_vocab=label_datastream_bpe5000.vocab,
                    beam_search_opts=BeamSearchOpts(
                        beam_size=beam_size,
                        length_normalization_exponent=1.0,
                        length_reward=0.0,
                        bos_label=0,
                        eos_label=0,
                        num_labels=label_datastream_bpe5000.vocab_size,
                    ),
                    lm_module="lm.lstm.kazuki_lstm_zijian_variant_v2",
                    lm_args=neural_lm.net_args,
                    lm_checkpoint=neural_lm.checkpoint,
                    lm_scale=lm_scale,
                ),
                use_gpu=True,
            )

    train_args_higher_peak = copy.deepcopy(train_args)
    train_args_higher_peak["config"]["learning_rates"] = list(np.linspace(5e-5, 5e-5, 20)) + list(
        np.linspace(5e-4, 7e-4, 110)) + list(np.linspace(7e-4, 5e-5, 110)) + list(np.linspace(5e-5, 1e-7, 10))

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_11gbgpu_25eps_new_baseline_peak7e-4"
    train_job = training(training_name, train_data_bpe5000, train_args_higher_peak, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
    
    
    network_module = "aed.conformer_zoneout_prototype_0624.aed_prototype_v3"
    train_args = {
        "config": train_config_11gbgpu_short_baseline,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_zeineldeen_config_v3)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_11gbgpu_25eps_new_baseline"
    train_job = training(training_name, train_data_bpe5000, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

    network_module = "aed.conformer_zoneout_prototype_0624.aed_prototype_v4"
    train_args = {
        "config": train_config_11gbgpu_short_baseline,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_zeineldeen_config_v4)},
        "debug": False,
    }

    train_args_minimal_lower_bs = copy.deepcopy(train_args)
    train_args_minimal_lower_bs["config"]["batch_size"] = 290 * 16000

    # training_name = prefix_name + "/" + network_module + ".512dim_sub6_11gbgpu_25eps_new_baseline"
    # train_job = training(training_name, train_data_bpe5000, train_args_minimal_lower_bs, num_epochs=250, **default_returnn)
    # train_job.rqmt["gpu_mem"] = 11
    # train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # asr_model = prepare_asr_model(
    #     training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000,
    #     get_specific_checkpoint=250
    # )
    # greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
    
    train_args_minimal_lower_bs_OCLR = copy.deepcopy(train_args_minimal_lower_bs)
    train_args_minimal_lower_bs_OCLR["config"]["learning_rates"] = list(np.linspace(5e-5, 5e-5, 20)) + list(
        np.linspace(5e-5, 7e-4, 100)) + list(np.linspace(7e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10))

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_11gbgpu_25eps_new_baseline_oclr"
    train_job = training(training_name, train_data_bpe5000, train_args_minimal_lower_bs_OCLR, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

    # Try again exactly like baseline, now with label smoothing 0.01
    for label_smoothing in [0.01, 0.02, 0.05, 0.1]:
        model_zeineldeen_config_v4_ls001 = copy.deepcopy(model_zeineldeen_config_v4)
        model_zeineldeen_config_v4_ls001.label_smoothing = label_smoothing

        train_args_labelsmoothing_001 = {
            "config": train_config_11gbgpu_short_baseline,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_zeineldeen_config_v4_ls001)},
            "debug": False,
        }

        training_name = prefix_name + "/" + network_module + ".512dim_sub6_11gbgpu_25eps_new_baseline_ls%.2f" % label_smoothing
        train_job = training(training_name, train_data_bpe5000, train_args_labelsmoothing_001, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 11
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        asr_model = prepare_asr_model(
            training_name, train_job, train_args_labelsmoothing_001, with_prior=False, datasets=train_data_bpe5000,
            get_specific_checkpoint=250
        )
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

        beam_search_prototype(
            training_name,
            asr_model=asr_model,
            decoder_config=BeamSearchDecoderConfig(
                returnn_vocab=label_datastream_bpe5000.vocab,
                beam_search_opts=BeamSearchOpts(
                    beam_size=12,
                    length_normalization_exponent=1.0,
                    length_reward=0,
                    bos_label=0,
                    eos_label=0,
                    num_labels=label_datastream_bpe5000.vocab_size
                )
            ),
            use_gpu=True,
        )

        neural_lm = get_lm_model("bpe5k_2x2024_kazuki_lstmlm_3ep")
        for beam_size in [12]:
            for lm_scale in [0.26, 0.28, 0.30, 0.32, 0.34]:
                beam_search_with_lm_prototype(
                    training_name + "/lm3ep_%.2f_bs%i" % (lm_scale, beam_size),
                    asr_model=asr_model,
                    decoder_config=BeamSearchLmDecoderConfig(
                        returnn_vocab=label_datastream_bpe5000.vocab,
                        beam_search_opts=BeamSearchOpts(
                            beam_size=beam_size,
                            length_normalization_exponent=1.0,
                            length_reward=0.0,
                            bos_label=0,
                            eos_label=0,
                            num_labels=label_datastream_bpe5000.vocab_size,
                        ),
                        lm_module="lm.lstm.kazuki_lstm_zijian_variant_v2",
                        lm_args=neural_lm.net_args,
                        lm_checkpoint=neural_lm.checkpoint,
                        lm_scale=lm_scale,
                    ),
                    use_gpu=True,
                )




