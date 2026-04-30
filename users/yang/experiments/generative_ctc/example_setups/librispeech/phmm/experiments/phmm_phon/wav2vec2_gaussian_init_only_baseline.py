import copy
from dataclasses import asdict
from typing import cast

from sisyphus import tk
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.report import (
    tune_and_evalue_report,
)

from ...data.phmm_common import DatasetSettings, build_test_dataset
from ...data.phmm_phon import get_phmm_eow_lexicon, build_eow_phon_phmm_training_datasets
from ...init_checkpoint import InitializeGaussianCheckpointJob
from ...phmm_config import get_forward_config
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_lm import get_4gram_lm_rasr_config
from ...phmm_pipeline import ASRModel, search
from ...phmm_rasr import (
    AddSentenceBoundaryLemmataToPhmmLexiconJob,
    CreateLibrasrVenvJob,
    build_librasr_phmm_recognition_config,
)
from ...pytorch_networks.phmm.decoder.rasr_phmm_v1 import DecoderConfig as RasrDecoderConfig
from ...pytorch_networks.phmm.wav2vec2_hf_phmm_v2_cfg import ModelConfig


def eow_phon_phmm_ls960_wav2vec2_gaussian_init_only_baseline():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phmm_eow_phon_wav2vec2_gaussian_init_only"
    class_stats_state = tk.Path(
        "output/example_setups/librispeech/phmm_standalone_2024/ls960_phmm_eow_phon_wav2vec2_class_stats/class_stats_state.pt"
    )
    recipe_root = tk.Path("/u/zyang/setups/mini/recipe", hash_overwrite="LIBRISPEECH_LOCAL_RECIPE_TOPLEVEL")

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data = build_eow_phon_phmm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = int(label_datastream.vocab_size.get())

    dev_dataset_tuples = {
        testset: build_test_dataset(dataset_key=testset, settings=train_settings)
        for testset in ["dev-clean", "dev-other"]
    }
    test_dataset_tuples = {
        testset: build_test_dataset(dataset_key=testset, settings=train_settings)
        for testset in ["test-clean", "test-other"]
    }

    phmm_returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    default_returnn = {
        "returnn_exe": phmm_returnn_exe,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    phmm_lexicon = get_phmm_eow_lexicon(g2p_librispeech_key="train-other-960")
    tk.register_output(prefix_name + "/phmm_eow_phon_lexicon.xml.gz", phmm_lexicon)
    phmm_recog_lexicon = AddSentenceBoundaryLemmataToPhmmLexiconJob(phmm_lexicon).out_lexicon
    tk.register_output(prefix_name + "/phmm_eow_phon_recog_lexicon.xml.gz", phmm_recog_lexicon)

    def make_decoder_config(
        *,
        lm_scale,
        decode_layer_index: int,
        prior_scale: float = 0.0,
        logfile_suffix: str,
    ):
        recog_config = build_librasr_phmm_recognition_config(
            lexicon_path=phmm_recog_lexicon,
            lm_config=get_4gram_lm_rasr_config(lexicon_file=phmm_recog_lexicon, scale=lm_scale),
            logfile_suffix=logfile_suffix,
            max_beam_size=512,
            intermediate_max_beam_size=512,
        )
        return RasrDecoderConfig(
            rasr_config_file=recog_config,
            lexicon=phmm_recog_lexicon,
            decode_layer_index=decode_layer_index,
            prior_scale=prior_scale,
        )

    def tune_and_evaluate_lm_scale(
        tuning_name: str,
        asr_model: ASRModel,
        *,
        decode_layer_index: int,
        lm_scales,
        mem=32,
    ):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}

        for lm_scale in lm_scales:
            decoder_config = make_decoder_config(
                lm_scale=lm_scale,
                decode_layer_index=decode_layer_index,
                prior_scale=0.0,
                logfile_suffix=f"phmm_phon_recog_layer{decode_layer_index}_lm{lm_scale:g}",
            )
            search_name = tuning_name + f"/search_lm{lm_scale:g}"
            _search_jobs, wers = search(
                search_name,
                forward_config={"num_workers_per_gpu": 0, "batch_size": 200*16000},
                asr_model=copy.deepcopy(asr_model),
                decoder_module="phmm.decoder.rasr_phmm_v1",
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples=dev_dataset_tuples,
                mem_rqmt=mem,
                **default_returnn,
            )
            tune_parameters.append((lm_scale,))
            tune_values_clean.append(wers[search_name + "/dev-clean"])
            tune_values_other.append(wers[search_name + "/dev-other"])

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters,
                values=tune_values,
                mode="minimize",
            )
            pick_optimal_params_job.add_alias(tuning_name + f"/pick_best_{key}")
            decoder_config = make_decoder_config(
                lm_scale=pick_optimal_params_job.out_optimal_parameters[0],
                decode_layer_index=decode_layer_index,
                prior_scale=0.0,
                logfile_suffix=f"phmm_phon_recog_layer{decode_layer_index}_{key}_best",
            )
            test_search_name = tuning_name + f"/best_{key}"
            _search_jobs, wers = search(
                test_search_name,
                forward_config={"num_workers_per_gpu": 0, "batch_size": 200*16000},
                asr_model=copy.deepcopy(asr_model),
                decoder_module="phmm.decoder.rasr_phmm_v1",
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                mem_rqmt=mem,
                **default_returnn,
            )
            report_values[key] = wers[test_search_name + "/" + key]

        tune_and_evalue_report(
            training_name=tuning_name,
            tune_parameters=tune_parameters,
            tuning_names=["LM"],
            tune_values_clean=tune_values_clean,
            tune_values_other=tune_values_other,
            report_values=report_values,
        )

    network_module = "phmm.wav2vec2_hf_phmm_v2"
    model_config = ModelConfig(
        label_target_size=vocab_size_without_blank,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=False,
        final_dropout=0.1,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        activation_dropout=0.1,
        layerdrop=0.05,
        mask_time_prob=0.0,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        gradient_checkpointing=False,
        aux_loss_layers=[6, 9, 12],
        aux_loss_scales=[1.0, 1.0, 1.0],
        generative_model=False,
        gaussian_mixture_model=True,
        num_gaussian_mixtures=1,
        gaussian_precision_type="diagonal",
        gaussian_class_dependent_variance=False,
        gaussian_class_stats_path=class_stats_state,
        gaussian_init_from_class_stats=True,
        gaussian_init_class_means_only=True,
        gaussian_init_precision=1.0,
        freeze_gaussian_precision=True,
    )

    checkpoint_job = InitializeGaussianCheckpointJob(
        python_exe=phmm_returnn_exe,
        recipe_root=recipe_root,
        network_module=network_module,
        model_config_dict=asdict(model_config),
        format_version=3,
    )
    checkpoint_job.add_alias(prefix_name + "/init_checkpoint")
    tk.register_output(prefix_name + "/init_checkpoint.pt", checkpoint_job.out_checkpoint)

    init_logprobs_config = get_forward_config(
        network_module=network_module,
        config={
            "forward": dev_dataset_tuples["dev-other"][0].as_returnn_opts(),
            "batch_size": 1 * 16000,
        },
        net_args={"model_config_dict": asdict(model_config)},
        decoder="phmm.init_logprobs_v1",
        decoder_args={"config": {"output_filename": "init_logprobs_debug.txt"}},
        debug=False,
    )
    # init_logprobs_job = ReturnnForwardJobV2(
    #     model_checkpoint=checkpoint_job.out_checkpoint,
    #     returnn_config=init_logprobs_config,
    #     log_verbosity=5,
    #     mem_rqmt=8,
    #     time_rqmt=4,
    #     device="gpu",
    #     cpu_rqmt=2,
    #     returnn_python_exe=phmm_returnn_exe,
    #     returnn_root=MINI_RETURNN_ROOT,
    #     output_files=["init_logprobs_debug.txt"],
    # )
    # init_logprobs_job.add_alias(prefix_name + "/init_logprobs_debug/dev-other")
    # tk.register_output(
    #     prefix_name + "/init_logprobs_debug/dev-other/init_logprobs_debug.txt",
    #     init_logprobs_job.out_files["init_logprobs_debug.txt"],
    # )

    asr_model = ASRModel(
        checkpoint=checkpoint_job.out_checkpoint,
        net_args={"model_config_dict": asdict(model_config)},
        network_module=network_module,
        prior_file=None,
        prior_files=None,
        prefix_name=prefix_name,
    )

    for decode_layer_index in [9]:
        tune_and_evaluate_lm_scale(
            prefix_name + f"/decode_layer_{decode_layer_index}_lm_tuning",
            asr_model,
            decode_layer_index=decode_layer_index,
            lm_scales=(1.5,1.4,1.2), # best around 1.2-1.5,
            mem=32,
        )
# max beam 256, lm 1.5, dev-other 60.5

py = eow_phon_phmm_ls960_wav2vec2_gaussian_init_only_baseline
