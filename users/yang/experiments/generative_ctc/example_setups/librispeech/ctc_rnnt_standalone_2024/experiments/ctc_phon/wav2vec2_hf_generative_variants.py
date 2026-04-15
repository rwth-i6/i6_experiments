import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...config import get_feature_variance_config, get_forced_align_config, get_pca_fit_config
from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_test_dataset_with_labels, build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import compute_feature_variance, compute_forced_alignments, compute_pca_state, prepare_asr_model, search, training
from ...report import tune_and_evalue_report


def eow_phon_ls960_wav2vec2_hf_generative_variants():
    prefix_name = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon_wav2vec2_hf_generative_models"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
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
    dev_dataset_tuples_with_labels = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples_with_labels[testset] = build_eow_phon_test_dataset_with_labels(
            dataset_key=testset,
            settings=train_settings,
            label_datastream=label_datastream,
            g2p_librispeech_key="train-other-960",
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.wav2vec2_hf_ctc_v2_cfg import ModelConfig

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
        ctc_loss_reduction="sum",
        gradient_checkpointing=False,
        aux_ctc_loss_layers=[3, 6, 9, -1],
        aux_ctc_loss_scales=[0.3, 0.3, 0.3, 1.0],
        generative_model=True,
        l2_norm=False,
        variance_normalize=False,
        pca_dim=None,
    )

    train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-08, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(1e-05, 3e-05, 40))
        + list(np.linspace(3e-05, 5e-05, 40))
        + list(np.linspace(5e-05, 1e-06, 120)),
        "batch_size": 250 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 2,
        "torch_amp_options": {"dtype": "bfloat16"},
        "gradient_clip_norm": 1.0,
        "num_workers_per_gpu": 2,
    }

    variance_forward_config = {
        "batch_size": 1000 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "torch_amp_options": {"dtype": "bfloat16"},
        "num_workers_per_gpu": 2,
    }
    pca_forward_config = {
        "batch_size": 1000 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "torch_amp_options": {"dtype": "bfloat16"},
        "num_workers_per_gpu": 2,
    }

    network_module = "ctc.wav2vec2_hf_ctc_v2"

    def tune_and_evaluate_lm_scale(
        tuning_name,
        asr_model,
        base_decoder_config,
        lm_scales,
        *,
        decoder_module="ctc.decoder.flashlight_ctc_v1",
    ):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}

        for lm_weight in lm_scales:
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = lm_weight
            search_name = tuning_name + "/search_lm%.1f" % lm_weight
            _search_jobs, wers = search(
                search_name,
                forward_config={"batch_size": 50 * 16000},
                asr_model=copy.deepcopy(asr_model),
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples=dev_dataset_tuples,
                use_gpu=False,
                **default_returnn,
            )
            tune_parameters.append((lm_weight,))
            tune_values_clean.append(wers[search_name + "/dev-clean"])
            tune_values_other.append(wers[search_name + "/dev-other"])

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters,
                values=tune_values,
                mode="minimize",
            )
            pick_optimal_params_job.add_alias(tuning_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            test_search_name = tuning_name + f"/best_{key}"
            _search_jobs, wers = search(
                test_search_name,
                forward_config={"batch_size": 50 * 16000},
                asr_model=copy.deepcopy(asr_model),
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                use_gpu=False,
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

    def run_training_and_search(
        run_name_suffix: str,
        model_config: ModelConfig,
        *,
        decoding: bool = True,
        decode_layers=(6,9),
        run_forced_align: bool = False,
        forced_align_layers=(6, 9),
        forced_align_write_ids: bool = True,
        forced_align_write_labels: bool = True,
    ):
        train_args = {
            "config": train_config,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = prefix_name + "/" + network_module + run_name_suffix

        train_job = training(training_name, train_data, train_args, num_epochs=200, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24

        asr_model = prepare_asr_model(
            training_name,
            train_job,
            train_args,
            with_prior=False,
            datasets=train_data,
            get_specific_checkpoint=200,
        )
        if decoding:
            for decode_layer_index in decode_layers:
                base_decoder_config = DecoderConfig(
                    lexicon=get_text_lexicon(),
                    returnn_vocab=label_datastream.vocab,
                    beam_size=1024,
                    beam_size_token=12,
                    arpa_lm=arpa_4gram_lm,
                    beam_threshold=14,
                    lm_weight=1.8,
                    prior_scale=0.0,
                    decode_layer_index=decode_layer_index,
                )

                tune_and_evaluate_lm_scale(
                    training_name + f"/decode_layer_{decode_layer_index}_lm_tuning",
                    asr_model,
                    base_decoder_config,
                    lm_scales=[2,3,10, 15, 20, 25,30],
                )

        if run_forced_align:
            for forced_align_layer_index in forced_align_layers:
                for dataset_key, (forced_align_dataset, _bliss) in dev_dataset_tuples_with_labels.items():
                    output_filename = f"forced_align_layer_{forced_align_layer_index}.txt"
                    forced_align_config = get_forced_align_config(
                        forward_dataset=forced_align_dataset,
                        network_module=network_module,
                        config={"batch_size": 50 * 16000},
                        net_args=asr_model.net_args,
                        forward_args={
                            "config": {
                                "decode_layer_index": forced_align_layer_index,
                                "output_filename": output_filename,
                                "returnn_vocab": label_datastream.vocab,
                                "write_ids": forced_align_write_ids,
                                "write_labels": forced_align_write_labels,
                            }
                        },
                        debug=False,
                    )
                    compute_forced_alignments(
                        training_name + f"/forced_align_layer_{forced_align_layer_index}/{dataset_key}",
                        forced_align_config,
                        checkpoint=asr_model.checkpoint,
                        output_files=[output_filename],
                        use_gpu=False,
                        **default_returnn,
                    )

        return training_name, train_job

    squared_diff_model_config = copy.deepcopy(base_model_config)
    squared_diff_name = ".wav2vec2_base_generative_sqdiff_aux3369final_lr5e-5_bs120x2_amp"
    run_training_and_search(
        squared_diff_name,
        squared_diff_model_config,
        decoding=False,
        run_forced_align=True,
    )

    pca512_model_config = copy.deepcopy(base_model_config)
    pca512_model_config.pca_dim = 512

    pca512_fit_model_config = copy.deepcopy(pca512_model_config)
    pca512_fit_model_config.update_pca_during_training = True
    pca512_fit_model_config.freeze_output_layers = True

    pca512_returnn_config = get_pca_fit_config(
        forward_dataset=train_data.prior,
        network_module=network_module,
        config=pca_forward_config,
        net_args={"model_config_dict": asdict(pca512_fit_model_config)},
        debug=False,
    )
    pca512_outputs, pca512_job = compute_pca_state(
        prefix_name + "/" + network_module + ".wav2vec2_base_generative_sqdiff_pca512_pca_fit",
        pca512_returnn_config,
        **default_returnn,
    )
    pca512_job.rqmt["gpu_mem"] = 24

    pca512_model_config.pca_state_path = pca512_outputs["pca_state.pt"]
    pca512_model_config.update_pca_during_training = False
    pca512_model_config.freeze_output_layers = False
    pca512_name = ".wav2vec2_base_generative_sqdiff_pca512_aux3369final_lr5e-5_bs120x2_amp"
    run_training_and_search(
        pca512_name,
        pca512_model_config,
    )

#    l2_sqdiff_model_config = copy.deepcopy(base_model_config)
#    l2_sqdiff_model_config.l2_norm = True
#    l2_sqdiff_name = ".wav2vec2_base_generative_l2_sqdiff_aux3369final_lr5e-5_bs120x2_amp"
#    run_training_and_search(
#        l2_sqdiff_name,
#        l2_sqdiff_model_config,
#    )
#
#    variance_model_config = copy.deepcopy(base_model_config)
#    variance_model_config.variance_normalize = True
#
#    variance_returnn_config = get_feature_variance_config(
#        forward_dataset=train_data.prior,
#        network_module=network_module,
#        config=variance_forward_config,
#        net_args={"model_config_dict": asdict(base_model_config)},
#        debug=False,
#    )
#
#    variance_output_files = (
#        [f"mean_{layer}.txt" for layer in base_model_config.aux_ctc_loss_layers]
#        + [f"variance_{layer}.txt" for layer in base_model_config.aux_ctc_loss_layers]
#        + ["variance_state.pt"]
#    )
#    variance_outputs, variance_job = compute_feature_variance(
#        prefix_name + "/" + network_module + ".wav2vec2_base_generative_sqdiff_varnorm_feature_variance",
#        variance_returnn_config,
#        checkpoint=None,
#        output_files=variance_output_files,
#        **default_returnn,
#    )
#    variance_job.rqmt["gpu_mem"] = 24
#
#    variance_model_config.variance_path = variance_outputs["variance_state.pt"]
#    variance_name = ".wav2vec2_base_generative_sqdiff_varnorm_aux3369final_lr5e-5_bs120x2_amp"
#    run_training_and_search(
#        variance_name,
#        variance_model_config,
#    )


py = eow_phon_ls960_wav2vec2_hf_generative_variants
