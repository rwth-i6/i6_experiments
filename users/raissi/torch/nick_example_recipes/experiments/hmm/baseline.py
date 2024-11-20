import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List
import textwrap
from functools import partial
from IPython import embed

from sisyphus.delayed_ops import DelayedFormat

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_core.text.processing import WriteToTextFileJob
from i6_core.corpus.transform import MergeCorporaJob
from i6_core.rasr.config import build_config_from_mapping

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.raissi.setups.common.decoder import RasrFeatureScorer
from i6_experiments.users.raissi.setups.common.helpers.train.oclr import get_oclr_config
from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhonemeStateClasses,
    RasrStateTying, PhoneticContext,
)
from i6_experiments.users.raissi.torch.decoder.TORCH_factored_hybrid_search import FeatureFlowType, ONNXDecodeIOMap
from ...config import get_decoding_config
#from recipe.returnn import ExportPyTorchModelToOnnxJobV2

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_phon_training_datasets, get_text_lexicon, get_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel, export_model_for_rasr_decoding
from ...report import tune_and_evalue_report


def ls960_hmm_conformer_monophone():
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
    #vocab_size = label_datastream.vocab_size not used in rasr based setups



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

    rasr_config_template = textwrap.dedent(
        """\
        [*]
        configuration.channel    = output-channel
        dot.channel              = nil
        encoding                 = UTF-8
        error.channel            = output-channel, stderr
        log.channel              = output-channel
        progress.channel         = output-channel
        real-time-factor.channel = output-channel
        statistics.channel       = output-channel
        system-info.channel      = output-channel
        time.channel             = output-channel
        version.channel          = output-channel
        warning.channel          = output-channel, stderr
        action                   = python-control

        [*.output-channel]
        append     = no
        compressed = no
        file       = fastbw.log
        unbuffered = no

        [*.allophone-state-graph-builder.orthographic-parser]
        allow-for-silence-repetitions   = no
        normalize-lemma-sequence-scores = no

        [*.model-combination.acoustic-model]
        fix-allophone-context-at-word-boundaries         = yes
        transducer-builder-filter-out-invalid-allophones = yes

        [*.model-combination.acoustic-model.allophones]
        add-all          = no
        add-from-lexicon = yes

        [*.model-combination.acoustic-model.hmm]
        across-word-model   = yes
        early-recombination = no
        state-repetitions   = 1
        states-per-phone    = 1

        [*.model-combination.acoustic-model.state-tying]
        type                 = monophone-dense
        use-boundary-classes = no
        use-word-end-classes = yes

        [*.model-combination.acoustic-model.tdp]
        applicator-type = corrected
        entry-m1.loop   = infinity
        entry-m2.loop   = infinity
        scale           = 1.0

        [*.model-combination.acoustic-model.tdp.*]
        exit    = 0.0
        forward = 0.6931471805599453
        loop    = 0.6931471805599453
        skip    = infinity

        [*.model-combination.acoustic-model.tdp.silence]
        exit    = 0.0
        forward = 1.8325814637483102
        loop    = 0.1743533871447778
        skip    = infinity

        [*.model-combination.lexicon]
        file                    = `cf {lexicon}`
        normalize-pronunciation = no

        [*.corpus]
        capitalize-transcriptions      = no
        file                           = {corpus}
        progress-indication            = global
        remove-corpus-name-prefix      = loss-corpus/
        warn-about-unexpected-elements = yes
        """
    )

    train_lexicon = get_lexicon(
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

    rasr_config_str = DelayedFormat(
        rasr_config_template,
        **{
            "lexicon": train_lexicon.get_path(),
            "corpus": merged_corpus.out_merged_corpus,
        },
    )
    create_rasr_config_file_job = WriteToTextFileJob(content=rasr_config_str, out_name=f"rasr.config")

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
    am_scales = [0.3] #, 0.4, 0.6]  #, 0.7]

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
        fsa_config_path=create_rasr_config_file_job.out_file,
        normalization=1,
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
        """
        for div_by_am in [True, False]:
            divisor = am_scale if div_by_am else 1.0
            for peak_lr in [4e-4, 1e-3]:
                train_configs[f"tina_AM{am_scale}_LR{peak_lr:.0e}{'_div' if div_by_am else ''}"] = (get_tina_oclr_config(250, peak_lr / divisor), ModelConfigTemplate(am_scale=am_scale))
        """

    for config_name, (lr_config, model_config) in train_configs.items():
        train_config_11gbgpu_amp = {
            "batch_size": 100 * 16000,
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
            #prior_kwargs={"device": "cpu"}
        )




        
        import i6_core.mm as mm
        import i6_experiments.users.raissi.experiments.librispeech.data_preparation.common.base_args as lbs_data_setups
        from i6_experiments.users.raissi.torch.decoder.TORCH_factored_hybrid_search import TORCHFactoredHybridDecoder
        from i6_experiments.users.raissi.utils.default_tools import u22_rasr_path_onnxtorch

        train_job = training(training_name, train_data, train_args_conv_first, num_epochs=250, **default_returnn)



        decoding_returnn_config = get_decoding_config(training_datasets=train_data, **train_args_conv_first)


        

        onnx_model = export_model_for_rasr_decoding(
            checkpoint=asr_model.checkpoint,
            returnn_config=decoding_returnn_config,
            returnn_root=

        )

        dummy_mixtures = mm.CreateDummyMixturesJob(
            label_info.get_n_of_dense_classes(),
            50,
        ).out_mixtures
        corpus_data = lbs_data_setups.get_corpus_data_inputs(corpus_key="train-other-960")
        rasr_init_args = lbs_data_setups.get_init_args()


        decoder = TORCHFactoredHybridDecoder(

        name=f"{training_name}_decode",
        rasr_binary_path=u22_rasr_path_onnxtorch,
        rasr_input_mapping=corpus_data.dev_data,
        rasr_init_args=rasr_init_args,
        context_type=PhoneticContext.monophone,
        feature_scorer_type=RasrFeatureScorer.onnx,
        feature_flow_type=FeatureFlowType.SAMPLE,
        #feature_path=None, #toDo
        model_path=None, #toDo
        io_map=ONNXDecodeIOMap.default(),
        mixtures=dummy_mixtures)

        #embed()
        """


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




        def tune_and_evaluate_helper(
        training_name,
        asr_model,
        base_decoder_config,
        lm_scales,
        prior_scales,
        decoder_module="ctc.decoder.flashlight_ctc_v1",  # TODO: use different decoder
    ):

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
    """
