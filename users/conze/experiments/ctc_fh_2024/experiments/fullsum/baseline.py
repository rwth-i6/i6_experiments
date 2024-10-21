import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List
import textwrap

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_core.text.processing import WriteToTextFileJob
from i6_core.corpus.transform import MergeCorporaJob

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_phon_training_datasets, get_text_lexicon, get_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...report import tune_and_evalue_report


def ls960_fullsum_base():
    prefix_name = "ctc_fh_2024/ls960_fullsum"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    # TODO: lexicon/vocab doesn't include blank
    train_data = build_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
        add_eow_phonemes=True,
        add_silence=False,
        use_tags=True,
    )

    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

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

    def tune_and_evaluate_helper(
        training_name,
        asr_model,
        base_decoder_config,
        lm_scales,
        prior_scales,
        decoder_module="ctc.decoder.flashlight_ctc_v1",
    ):
        """
        TODO: run dev only flag
        """
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
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
            report_values[key] = wers[training_name + "/" + key]

        tune_and_evalue_report(
            training_name=training_name,
            tune_parameters=tune_parameters,
            tuning_names=["LM", "Prior"],
            tune_values_clean=tune_values_clean,
            tune_values_other=tune_values_other,
            report_values=report_values,
        )

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,  # beam-pruning-limit in RASR
        beam_size_token=12,  # similar to ALTAS, considers that many labels per frame, makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,  # beam-pruning in RASR
    )

    from ...pytorch_networks.fullsum.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
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
        out_features=512,
        activation=None,
    )

    # RASR config file for the FSA builder
    # Adapted from ...
    rasr_config_template = textwrap.dedent(
        """\
        [*]
        configuration.channel                   = output-channel
        dot.channel                             = nil
        encoding                                = UTF-8
        error.channel                           = output-channel, stderr
        log.channel                             = output-channel
        progress.channel                        = output-channel
        real-time-factor.channel                = output-channel
        statistics.channel                      = output-channel
        system-info.channel                     = output-channel
        time.channel                            = output-channel
        version.channel                         = output-channel
        warning.channel                         = output-channel, stderr
        allophone-state-graph-builder.topology  = ctc
        allow-for-silence-repetitions           = no
        LOGFILE                                 = nn-trainer.loss.log
        TASK                                    = 1
        action                                  = python-control
        python-control-loop-type                = python-control-loop
        extract-features                        = no

        [*.output-channel]
        append     = no
        compressed = no
        file       = fastbw.log
        unbuffered = no

        [*.model-combination.acoustic-model]
        fix-allophone-context-at-word-boundaries            = yes
        transducer-builder-filter-out-invalid-allophones    = yes

        [*.model-combination.acoustic-model.allophones]
        add-from-lexicon    = no
        add-all             = yes

        [*.model-combination.acoustic-model.hmm]
        states-per-phone    = 1
        state-repetitions   = 1
        across-word-model   = yes
        early-recombination = no

        [*.model-combination.acoustic-model.state-tying]
        type                 = monophone-eow

        [*.model-combination.acoustic-model.tdp]
        scale           = 1.0
        entry-m1.loop   = infinity
        entry-m2.loop   = infinity

        [*.model-combination.acoustic-model.tdp.*]
        loop    = 0.0
        forward = 0.0
        skip    = infinity
        exit    = 0.0

        [*.model-combination.acoustic-model.tdp.silence]
        loop    = 0.0
        forward = 0.0
        skip    = infinity
        exit    = 0.0

        [*.model-combination.lexicon]
        file                      = `cf {lexicon}`
        
        [*.corpus]
        file                           = {corpus}
        remove-corpus-name-prefix      = loss-corpus/
        warn-about-unexpected-elements = yes
        """
    )

    lexicon = get_lexicon(
        g2p_librispeech_key="train-other-960",
        with_g2p=True,
        add_eow_phonemes=True,
        add_silence=False,
    )

    all_corpora = get_bliss_corpus_dict()
    merged_corpus = MergeCorporaJob(
        [
            all_corpora["train-other-960"],
            all_corpora["dev-clean"],
            all_corpora["dev-other"],
        ],
        "loss-corpus",
    )

    rasr_config_str = DelayedFormat(
        rasr_config_template,
        **{
            # Lexicon is copy of /u/raissi/setups/librispeech/seq2seq_rasr/work/i6_experiments/users/berger/recipe/lexicon/modification/MakeBlankLexiconJob.3xj2btBvtWYR/output/lexicon.xml,
            # includes blank. TODO: Create via MakeBlankLexiconJob from lexicon
            "lexicon": "/work/smt4/benedikt.conze/temp-copies/temp_lexicon.xml",
            "corpus": merged_corpus.out_merged_corpus,
        },
    )

    create_rasr_config_file_job = WriteToTextFileJob(content=rasr_config_str, out_name=f"rasr.config")

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
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
        specauc_start_epoch=1,
        tdp_scale=0.0,  # Since the TDPs are 0 or inf here, this doesn't matter
        fsa_config_path=create_rasr_config_file_job.out_file,
    )

    train_config_11gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 120))
        + list(np.linspace(5e-4, 5e-5, 120))
        + list(np.linspace(5e-5, 1e-7, 10)),
        #############
        "batch_size": 150 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        # "torch_amp_options": {"dtype": "bfloat16"},  # No mixed-precision training on 11GB-GPUs
        "gradient_clip_norm": 1.0,
    }
    train_config_11gbgpu_amp_sp = copy.deepcopy(train_config_11gbgpu_amp)

    # Same with conv first
    network_module_conv_first = "fullsum.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
    train_args_conv_first = {
        "config": train_config_11gbgpu_amp,
        "network_module": network_module_conv_first,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "include_native_ops": True,
    }
    train_args_conv_first_sp = copy.deepcopy(train_args_conv_first)
    train_args_conv_first_sp["config"] = train_config_11gbgpu_amp_sp
    train_args_conv_first_sp["use_speed_perturbation"] = True

    name = ".512dim_sub4_11gbgpu_100eps_lp_fullspec_gradnorm_smallbatch_fullsum2"
    training_name = prefix_name + "/" + network_module_conv_first + name
    train_job = training(training_name, train_data, train_args_conv_first, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    asr_model = prepare_asr_model(
        training_name,
        train_job,
        train_args_conv_first,
        with_prior=True,
        datasets=train_data,
        get_specific_checkpoint=250,
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4]
    )

    # name = ".512dim_sub4_24gbgpu_100eps_lp_fullspec_gradnorm_smallbatch_sp_fullsum1"
    # training_name = prefix_name + "/" + network_module_conv_first + name
    # train_job = training(training_name, train_data, train_args_conv_first_sp, num_epochs=1000, **default_returnn)
    # train_job.rqmt["gpu_mem"] = 24
    # asr_model = prepare_asr_model(
    #    training_name, train_job, train_args_conv_first_sp, with_prior=True, datasets=train_data,
    #    get_specific_checkpoint=1000
    # )
    # tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0],
    #                         prior_scales=[0.2, 0.3, 0.4])

    # No improvement, just as example
    # asr_model_best4 = prepare_asr_model(
    #     training_name+ "/best4", train_job, train_args, with_prior=True, datasets=train_data, get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    # )
    # tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])
