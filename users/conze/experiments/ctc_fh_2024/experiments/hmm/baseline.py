import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List
import textwrap
from functools import partial

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_core.text.processing import WriteToTextFileJob
from i6_core.corpus.transform import MergeCorporaJob
import i6_core.rasr as rasr
from i6_core.am.config import acoustic_model_config
from i6_core.meta.system import CorpusObject

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
import i6_experiments.common.setups.rasr.config.lex_config as exp_rasr
from i6_experiments.users.raissi.setups.common.helpers.train.oclr import get_oclr_config
from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhonemeStateClasses,
    RasrStateTying,
)

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_phon_training_datasets, get_text_lexicon, get_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...report import tune_and_evalue_report


def ls960_hmm_base():
    prefix_name = "ctc_fh_2024/ls960_hmm"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data, bliss_corpora = build_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
        add_eow_phonemes=False,
        add_silence=True,
        use_tags=True,
        apply_lexicon=False,
        set_target_opts=False,
    )

    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size = label_datastream.vocab_size

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
        decoder_module="ctc.decoder.flashlight_ctc_v1",  # TODO: use different decoder
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

    from ...pytorch_networks.hmm.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
    )

    # Use same feature extraction as the CTC model
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

    lexicon = get_lexicon(
        g2p_librispeech_key="train-other-960",
        with_g2p=True,
        add_eow_phonemes=False,
        add_silence=True,
    )
    merged_corpus = MergeCorporaJob(
        [
            bliss_corpora["train-other-960"],
            bliss_corpora["dev-clean"],
            bliss_corpora["dev-other"],
        ],
        "loss-corpus",
    )

    # Create RASR config
    crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(crp)

    corpus_data = CorpusObject(
        corpus_file=merged_corpus.out_merged_corpus,
    )
    rasr.crp_set_corpus(crp, corpus_data)
    crp.corpus_config.remove_corpus_name_prefix = "loss-corpus/"

    lexicon_config = exp_rasr.LexiconRasrConfig(
        lex_path = lexicon.get_path(),
        normalize_pronunciation=False,
    )
    crp.lexicon_config = lexicon_config.get()

    crp.acoustic_model_config = acoustic_model_config(
        state_tying="monophone-dense",
        states_per_phone=1,
        state_repetitions=1,
        across_word_model=True,
        early_recombination=False,
        tdp_scale=1.0,
        tdp_transition=(0.6931471805599453, 0.6931471805599453, "infinity", 0.0),
        tdp_silence=(0.1743533871447778, 1.8325814637483102, "infinity", 0.0),
    )
    crp.acoustic_model_config.state_tying.use_boundary_classes = False
    crp.acoustic_model_config.state_tying.use_word_end_classes = True

    crp.acoustic_model_config.tdp.applicator_type = "corrected"

    crp.acoustic_model_config.fix_allophone_context_at_word_boundaries = True
    crp.acoustic_model_config.transducer_builder_filter_out_invalid_allophones = True

    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }
    config, post_config = rasr.build_config_from_mapping(
        crp,
        mapping,
    )
    config["*"].action = "python-control"
    config["*.allophone-state-graph-builder.orthographic-parser"].allow_for_silence_repetitions = False
    config["*.allophone-state-graph-builder.orthographic-parser"].normalize_lemma_sequence_scores = False

    create_rasr_config_job = rasr.WriteRasrConfigJob(config=config, post_config=post_config)

    # Get label info
    label_info = LabelInfo(
        n_contexts=label_datastream.vocab_size + 1,  # + empty context
        n_states_per_phone=1,
        phoneme_state_classes=PhonemeStateClasses.word_end,
        ph_emb_size=0,
        st_emb_size=0,
        state_tying=RasrStateTying.monophone,
        add_unknown_phoneme=True,  # Setting this to true means the n_contexts already includes the unknown phoneme
    )

    train_configs = {}
    am_scales = [0.3, 0.5]  #, 0.7]

    # TDP scale and fsa config adapted from:
    # /work/asr4/raissi/setups/librispeech/960-ls/2023-01--system_paper/work/i6_experiments/users/raissi/costum/returnn/rasr_returnn_bw/ReturnnRasrTrainingBWJob.j0QluKd9vhJ8
    # Create multiple configs with different AM scales
    ModelConfigTemplate = partial(ModelConfig,
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        label_target_size=label_info.get_n_of_dense_classes(),
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
        tdp_scale=0.1,
        fsa_config_path=create_rasr_config_job.out_config,
        normalization=1,  # unused
    )

    def get_tina_oclr_config(num_epochs, lrate):
        lrs = get_oclr_config(num_epochs, lrate)
        lrs["optimizer"] = {"class": "nadam", "epsilon": 1e-8}
        return lrs

    nick_lr_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "learning_rate_file": "lr.log",
        "learning_rates": list(np.linspace(7e-6, 5e-4, 120)) + list(
            np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10)),
    }

    for am_scale in am_scales:
        train_configs[f"nick_AM{am_scale}"] = (nick_lr_config, ModelConfigTemplate(am_scale=am_scale))
        for div_by_am in [True, False]:
            divisor = am_scale if div_by_am else 1.0
            for peak_lr in [4e-4, 1e-3]:
                train_configs[f"tina_AM{am_scale}_LR{peak_lr:.0e}{'_div' if div_by_am else ''}"] = (get_tina_oclr_config(250, peak_lr / divisor), ModelConfigTemplate(am_scale=am_scale))

    for config_name, (lr_config, model_config) in train_configs.items():
        train_config_11gbgpu_amp = {
            "batch_size": 150 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            # "torch_amp_options": {"dtype": "bfloat16"},  # No mixed-precision training on 11GB-GPUs
            #"gradient_clip_norm": 1.0,
            "gradient_clip": 20.0,
            "gradient_noise": 0.0,
        }
        train_config_11gbgpu_amp.update(lr_config)

        # Same with conv first
        network_module_conv_first = "hmm.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
        train_args_conv_first = {
            "config": train_config_11gbgpu_amp,
            "network_module": network_module_conv_first,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "include_native_ops": True,
        }

        name = f".512dim_sub4_11gbgpu_100eps_lp_fullspec_gradnorm_smallbatch_hmm_{config_name}"
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
    # TODO: re-enable tune_and_evaluate_helper with an HMM decoder
    #tune_and_evaluate_helper(
    #    training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4]
    #)

    #name = ".512dim_sub4_11gbgpu_100eps_lp_fullspec_gradnorm_smallbatch_sp_v2"
    #training_name = prefix_name + "/" + network_module_conv_first + name
    #train_job = training(training_name, train_data, train_args_conv_first_sp, num_epochs=250, **default_returnn)
    #train_job.rqmt["gpu_mem"] = 11
    #asr_model = prepare_asr_model(
    #    training_name,
    #    train_job,
    #    train_args_conv_first_sp,
    #    with_prior=True,
    #    datasets=train_data,
    #    get_specific_checkpoint=250,
    #)
    #tune_and_evaluate_helper(
    #    training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4]
    #)

    # No improvement, just as example
    # asr_model_best4 = prepare_asr_model(
    #     training_name+ "/best4", train_job, train_args, with_prior=True, datasets=train_data, get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    # )
    # tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])
