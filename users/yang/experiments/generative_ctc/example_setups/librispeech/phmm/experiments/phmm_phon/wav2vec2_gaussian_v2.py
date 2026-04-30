import copy
from dataclasses import asdict
from typing import cast

import numpy as np
from sisyphus import tk
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.report import (
    tune_and_evalue_report,
)

from ...config import get_feature_variance_config
from ...data.phmm_common import DatasetSettings, build_test_dataset
from ...data.phmm_phon import build_eow_phon_phmm_training_datasets, get_phmm_eow_lexicon
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_lm import get_4gram_lm_rasr_config
from ...phmm_pipeline import ASRModel, compute_feature_variance, prepare_asr_model, search, training
from ...phmm_rasr import (
    AddSentenceBoundaryLemmataToPhmmLexiconJob,
    CreateLibrasrVenvJob,
    build_fsa_exporter_config,
    build_librasr_phmm_recognition_config,
)
from ...pytorch_networks.phmm.decoder.rasr_phmm_v1 import DecoderConfig as RasrDecoderConfig
from ...pytorch_networks.phmm.wav2vec2_hf_phmm_v2_cfg import ModelConfig


def eow_phon_phmm_ls960_wav2vec2_gaussian_v2():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phmm_eow_phon_wav2vec2_gaussian_v2"
    class_stats_state = tk.Path(
        "output/example_setups/librispeech/phmm_standalone_2024/ls960_phmm_eow_phon_wav2vec2_class_stats/class_stats_state.pt"
    )

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
    vocab_size_without_blank = label_datastream.vocab_size

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

    librispeech_corpus = get_bliss_corpus_dict(audio_format="ogg")["train-other-960"]
    fsa_exporter_config = build_fsa_exporter_config(
        lexicon_path=phmm_lexicon,
        corpus_path=librispeech_corpus,
    )

    def make_decoder_config(
        *,
        lm_scale,
        decode_layer_index: int,
        prior_scale: float = 0.3,
        logfile_suffix: str,
    ):
        recog_config = build_librasr_phmm_recognition_config(
            lexicon_path=phmm_recog_lexicon,
            lm_config=get_4gram_lm_rasr_config(lexicon_file=phmm_recog_lexicon, scale=lm_scale),
            logfile_suffix=logfile_suffix,
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
        prior_scale: float = 0.0,
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
                prior_scale=prior_scale,
                logfile_suffix=f"phmm_phon_recog_layer{decode_layer_index}_lm{lm_scale:g}",
            )
            search_name = tuning_name + f"/search_lm{lm_scale:g}"
            _search_jobs, wers = search(
                search_name,
                forward_config={"num_workers_per_gpu": 0},
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
                prior_scale=prior_scale,
                logfile_suffix=f"phmm_phon_recog_layer{decode_layer_index}_{key}_best",
            )
            test_search_name = tuning_name + f"/best_{key}"
            _search_jobs, wers = search(
                test_search_name,
                forward_config={"num_workers_per_gpu": 0},
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
    base_model_config = ModelConfig(
        label_target_size=vocab_size_without_blank,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=True,
        final_dropout=0.1,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        activation_dropout=0.1,
        layerdrop=0.05,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        gradient_checkpointing=False,
        aux_loss_layers=[6, 9, 12],
        aux_loss_scales=[1.0, 1.0, 1.0],
        generative_model=False,
        gaussian_mixture_model=True,
        gaussian_class_stats_path=class_stats_state,
        gaussian_init_from_class_stats=True,
    )

    def run_training_and_search(
        run_name_suffix: str,
        model_config: ModelConfig,
        *,
        lm_scales=(0.5, 1.0, 1.5),
        decoding_layers=(3, 6, 9, 12),
        gpu_mem=24,
        num_epochs=100,
        peak_lr=1e-4,
        init_lr=1e-5,
        batch_size=200,
        search_mem=32,
    ):
        epoch_1 = int(num_epochs * 0.45)
        epoch_2 = num_epochs - 2 * epoch_1
        training_name = prefix_name + "/" + network_module + run_name_suffix

        model_config_for_training = copy.deepcopy(model_config)
        if (
            model_config_for_training.gaussian_mixture_model
            and model_config_for_training.gaussian_init_from_variance_stats
            and not model_config_for_training.gaussian_init_from_class_stats
        ):
            if model_config_for_training.variance_path is None:
                variance_model_config = copy.deepcopy(model_config_for_training)
                variance_model_config.gaussian_init_from_variance_stats = False
                variance_output_files = (
                    [f"mean_{layer}.txt" for layer in variance_model_config.aux_loss_layers]
                    + [f"variance_{layer}.txt" for layer in variance_model_config.aux_loss_layers]
                    + ["variance_state.pt"]
                )
                variance_returnn_config = get_feature_variance_config(
                    forward_dataset=train_data.prior,
                    network_module=network_module,
                    config={"batch_size": batch_size * 16000},
                    net_args={"model_config_dict": asdict(variance_model_config)},
                    forward_args={},
                    debug=False,
                )
                variance_outputs, variance_job = compute_feature_variance(
                    training_name + "/feature_variance_init",
                    variance_returnn_config,
                    checkpoint=None,
                    output_files=variance_output_files,
                    **default_returnn,
                )
                variance_job.rqmt["gpu_mem"] = gpu_mem
                model_config_for_training.variance_path = variance_outputs["variance_state.pt"]

        train_config = {
            "optimizer": {"class": "adamw", "epsilon": 1e-08, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(init_lr, peak_lr, epoch_1))
            + list(np.linspace(peak_lr, init_lr, epoch_1))
            + list(np.linspace(init_lr, 1e-06, epoch_2)),
            "batch_size": batch_size * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
            "torch_amp_options": {"dtype": "bfloat16"},
            "gradient_clip_norm": 1.0,
            "num_workers_per_gpu": 2,
        }
        train_args = {
            "config": train_config,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config_for_training)},
            "train_step_args": {
                "fsa_exporter_config_path": fsa_exporter_config,
                "label_smoothing_scale": 0.0,
            },
            "include_native_ops": True,
            "use_speed_perturbation": True,
            "debug": False,
        }

        train_job = training(training_name, train_data, train_args, num_epochs=num_epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = gpu_mem

        asr_model = prepare_asr_model(
            training_name,
            train_job,
            train_args,
            with_prior=False,
            datasets=train_data,
            get_specific_checkpoint=num_epochs,
        )

        for decode_layer_index in decoding_layers:
            tune_and_evaluate_lm_scale(
                training_name + f"/recog_ep{num_epochs}/decode_layer_{decode_layer_index}_lm_tuning",
                asr_model,
                decode_layer_index=decode_layer_index,
                lm_scales=lm_scales,
                mem=search_mem,
            )

    # Old experiments intentionally disabled here. Only the three preloaded class-stats runs below remain active.

    run_training_and_search(
        ".gauss1_pooledvar_classmean_fixedprec_frozenenc_aux6912_lr1e-4",
        ModelConfig(
            **{
                **asdict(base_model_config),
                "num_gaussian_mixtures": 1,
                "gaussian_precision_type": "diagonal",
                "gaussian_class_dependent_variance": False,
                "gaussian_init_class_means_only": True,
                "freeze_gaussian_precision": True,
            }
        ),
    )

    run_training_and_search(
        ".gauss1_prec10_classmean_fixedprec_frozenenc_aux6912_lr1e-4",
        ModelConfig(
            **{
                **asdict(base_model_config),
                "num_gaussian_mixtures": 1,
                "gaussian_precision_type": "diagonal",
                "gaussian_class_dependent_variance": False,
                "gaussian_init_class_means_only": True,
                "gaussian_init_precision": 10.0,
                "freeze_gaussian_precision": True,
            }
        ),
    )

#    run_training_and_search(
#        ".gauss1_classvar_classmean_fixedprec_frozenenc_aux6912_lr1e-4",
#        ModelConfig(
#            **{
#                **asdict(base_model_config),
#                "num_gaussian_mixtures": 1,
#                "gaussian_precision_type": "diagonal",
#                "gaussian_class_dependent_variance": True,
#                "freeze_gaussian_precision": True,
#            }
#        ),
#    )
#
#    run_training_and_search(
#        ".gauss2_pooledvar_classmeanperturb_fixedprec_frozenenc_aux6912_lr1e-4",
#        ModelConfig(
#            **{
#                **asdict(base_model_config),
#                "num_gaussian_mixtures": 2,
#                "gaussian_precision_type": "diagonal",
#                "gaussian_class_dependent_variance": False,
#                "freeze_gaussian_precision": True,
#                "gaussian_class_stats_mixture_perturb_scale": 0.05,
#            }
#        ),
#    )


py = eow_phon_phmm_ls960_wav2vec2_gaussian_v2
