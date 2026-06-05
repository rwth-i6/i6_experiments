import copy
from dataclasses import asdict
from typing import cast

import numpy as np
from sisyphus import tk
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.data.hdf_seq_whitelist import (
    ExtractSeqListFromHDFJob,
)
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.report import (
    tune_and_evalue_report,
)

from ...data.phmm_common import (
    DatasetSettings,
    TrainingDatasets,
    build_oggzip_dataset_with_optional_hdf,
    build_test_dataset,
    get_audio_raw_datastream,
)
from ...data.phmm_phon import build_eow_phon_phmm_training_datasets, get_phmm_eow_lexicon, get_phmm_eow_vocab_datastream
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_lm import get_4gram_lm_rasr_config
from ...phmm_pipeline import ASRModel, prepare_asr_model, search, training
from ...phmm_rasr import (
    AddSentenceBoundaryLemmataToPhmmLexiconJob,
    CreateLibrasrVenvJob,
    build_fsa_exporter_config,
    build_librasr_phmm_recognition_config,
)
from ...pytorch_networks.phmm.decoder.rasr_phmm_v1 import DecoderConfig as RasrDecoderConfig
from ...pytorch_networks.phmm.wav2vec2_hf_phmm_generative_cfg import ModelConfig


def _build_generative_training_datasets(*, prefix_name: str, librispeech_key: str, settings: DatasetSettings) -> TrainingDatasets:
    from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict

    label_datastream = get_phmm_eow_vocab_datastream(prefix=prefix_name, g2p_librispeech_key=librispeech_key)
    alignment_hdf = [tk.Path(f"output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf") for i in range(1, 201)]
    alignment_seq_whitelist = ExtractSeqListFromHDFJob(alignment_hdf).out_seq_list
    split_job = ShuffleAndSplitSegmentsJob(
        segment_file=alignment_seq_whitelist,
        split={"train": 0.99, "cv": 0.01},
        shuffle=True,
    )
    split_job.add_alias(prefix_name + f"/{librispeech_key}/train_cv_99_1_split")
    train_segment_file = split_job.out_segments["train"]
    cv_segment_file = split_job.out_segments["cv"]
    alignment_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=label_datastream.vocab,
        vocab_size=label_datastream.vocab_size,
        unk_label=label_datastream.unk_label,
    )

    ogg_zip_dict = get_ogg_zip_dict(prefix_name, returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[librispeech_key]
    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    train_dataset, datastreams = build_oggzip_dataset_with_optional_hdf(
        ogg_files=train_ogg,
        audio_datastream=audio_datastream,
        label_datastream=None,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="data",
        partition_epoch=settings.train_partition_epoch,
        segment_file=train_segment_file,
        seq_ordering=settings.train_seq_ordering,
        additional_options=settings.train_additional_options,
    )
    eval_dataset, _ = build_oggzip_dataset_with_optional_hdf(
        ogg_files=train_ogg,
        audio_datastream=audio_datastream,
        label_datastream=None,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="data",
        partition_epoch=1,
        segment_file=cv_segment_file,
        seq_ordering="sorted_reverse",
        additional_options=None,
    )
    prior_dataset, _ = build_oggzip_dataset_with_optional_hdf(
        ogg_files=train_ogg,
        audio_datastream=audio_datastream,
        label_datastream=None,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="data",
        partition_epoch=1,
        segment_file=train_segment_file,
        seq_ordering="sorted_reverse",
        additional_options=None,
    )

    return TrainingDatasets(
        train=train_dataset,
        cv=eval_dataset,
        devtrain=eval_dataset,
        datastreams={**datastreams, "labels": label_datastream},
        prior=prior_dataset,
    )


def eow_phon_phmm_ls960_wav2vec2_generative2():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phmm_eow_phon_wav2vec2_generative2"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data = _build_generative_training_datasets(
        prefix_name=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    fullsum_train_data = build_eow_phon_phmm_training_datasets(
        prefix=prefix_name + "/fullsum_dataset",
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
    librispeech_corpus = get_bliss_corpus_dict(audio_format="ogg")["train-other-960"]
    fsa_exporter_config = build_fsa_exporter_config(
        lexicon_path=phmm_lexicon,
        corpus_path=librispeech_corpus,
    )

    def make_decoder_config(*, lm_scale, decode_layer_index: int, prior_scale: float = 0.0, logfile_suffix: str):
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
                prior_scale=0.0,
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

    network_module = "phmm.wav2vec2_hf_phmm_generative"
    base_model_config = ModelConfig(
        label_target_size=vocab_size_without_blank,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=False,
        freeze_encoder=False,
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
        aux_loss_layers=[9, 12],
        aux_loss_scales=[0.3, 1.0],
        freeze_output_layers=False,
        viterbi_training=True,
        sampling_type="batch",
        sampling_ratio=0.3,
        share_samples=False,
        ratio_corrector=1.0,
        input_time_batch_norm=True,
        input_residual_linear=True,
    )

    def run_training_and_search(
        run_name_suffix: str,
        model_config: ModelConfig,
        *,
        lm_scales=(1,1.2,0.8,1.5),
        decoding_layers=(12,),
        gpu_mem=24,
        num_epochs=200,
        peak_lr=1e-4,
        init_lr=1e-5,
        batch_size=100,
        accum_grad_multiple_step=2,
        search_mem=32,
        eval_epochs=(40, 80),
        datasets=None,
        train_step_args=None,
        include_native_ops=False,
    ):
        epoch_1 = int(num_epochs * 0.45)
        epoch_2 = num_epochs - 2 * epoch_1
        training_name = prefix_name + "/" + network_module + run_name_suffix
        decode_epochs = sorted(set(list(eval_epochs) + [num_epochs]))

        train_config = {
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": decode_epochs,
            },
            "optimizer": {"class": "adamw", "epsilon": 1e-08, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(init_lr, peak_lr, epoch_1))
            + list(np.linspace(peak_lr, init_lr, epoch_1))
            + list(np.linspace(init_lr, 1e-06, epoch_2)),
            "batch_size": batch_size * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": accum_grad_multiple_step,
            "torch_amp_options": {"dtype": "bfloat16"},
            "gradient_clip_norm": 1.0,
            "num_workers_per_gpu": 2,
        }
        if not include_native_ops:
            train_args = {
                "config": train_config,
                "network_module": network_module,
                "net_args": {"model_config_dict": asdict(model_config)},
                "train_step_args": {},
                "include_native_ops": False,
                "use_speed_perturbation": True,
                "debug": False,
            }
        else:
            train_args = {
                "config": train_config,
                "network_module": network_module,
                "net_args": {"model_config_dict": asdict(model_config)},
                "train_step_args": train_step_args or {},
                "include_native_ops": True,
                "use_speed_perturbation": True,
                "debug": False,
            }

        run_datasets = datasets or train_data
        train_job = training(training_name, run_datasets, train_args, num_epochs=num_epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = gpu_mem

        for checkpoint_epoch in decode_epochs:
            asr_model = prepare_asr_model(
                training_name,
                train_job,
                train_args,
                with_prior=False,
                datasets=run_datasets,
                get_specific_checkpoint=checkpoint_epoch,
            )

            for decode_layer_index in decoding_layers:
                tune_and_evaluate_lm_scale(
                    training_name + f"/recog_ep{checkpoint_epoch}/decode_layer_{decode_layer_index}_lm_tuning",
                    asr_model,
                    decode_layer_index=decode_layer_index,
                    lm_scales=lm_scales,
                    mem=search_mem,
                )

    frozen_encoder_base_model_config = ModelConfig(
        **{
            **asdict(base_model_config),
            "aux_loss_layers": [9],
            "aux_loss_scales": [1.0],
            "freeze_feature_encoder": True,
            "freeze_encoder": True,
        }
    )

    trainable_encoder_base_model_config = ModelConfig(
        **{
            **asdict(base_model_config),
            "aux_loss_layers": [9],
            "aux_loss_scales": [1.0],
            "freeze_feature_encoder": False,
            "freeze_encoder": False,
        }
    )

    viterbi_frozen_encoder_kernel4_stride2_model_config = ModelConfig(
        **{
            **asdict(frozen_encoder_base_model_config),
            "generator_kernel": 4,
            "generator_stride": 2,
            "viterbi_training": True,
        }
    )
    run_training_and_search(
        ".kernel4_stride2_bnres_frozenenc_aux9_viterbi",
        viterbi_frozen_encoder_kernel4_stride2_model_config,
        num_epochs=200,
        decoding_layers=(9,),
        accum_grad_multiple_step=2,
    )

    fullsum_frozen_encoder_kernel4_stride2_model_config = ModelConfig(
        **{
            **asdict(frozen_encoder_base_model_config),
            "generator_kernel": 4,
            "generator_stride": 2,
            "viterbi_training": False,
        }
    )
    run_training_and_search(
        ".kernel4_stride2_bnres_frozenenc_aux9_fullsum",
        fullsum_frozen_encoder_kernel4_stride2_model_config,
        num_epochs=200,
        decoding_layers=(9,),
        batch_size=150,
        accum_grad_multiple_step=4,
        datasets=fullsum_train_data,
        train_step_args={
            "fsa_exporter_config_path": fsa_exporter_config,
            "label_smoothing_scale": 0.0,
        },
        include_native_ops=True,
    )

    fullsum_frozen_encoder_kernel9_stride3_model_config = ModelConfig(
        **{
            **asdict(frozen_encoder_base_model_config),
            "generator_kernel": 9,
            "generator_stride": 3,
            "viterbi_training": False,
        }
    )
    run_training_and_search(
        ".kernel9_stride3_bnres_frozenenc_aux9_fullsum",
        fullsum_frozen_encoder_kernel9_stride3_model_config,
        num_epochs=200,
        decoding_layers=(9,),
        batch_size=150,
        accum_grad_multiple_step=4,
        datasets=fullsum_train_data,
        train_step_args={
            "fsa_exporter_config_path": fsa_exporter_config,
            "label_smoothing_scale": 0.0,
        },
        include_native_ops=True,
    )

    # viterbi_frozen_encoder_kernel1_stride1_model_config = ModelConfig(
    #     **{
    #         **asdict(frozen_encoder_base_model_config),
    #         "generator_kernel": 1,
    #         "generator_stride": 1,
    #         "viterbi_training": True,
    #     }
    # )
    # run_training_and_search(
    #     ".kernel1_stride1_bnres_frozenenc_aux9_viterbi",
    #     viterbi_frozen_encoder_kernel1_stride1_model_config,
    #     num_epochs=200,
    #     decoding_layers=(9,),
    #     accum_grad_multiple_step=2,
    # )

    fullsum_frozen_encoder_kernel1_stride1_model_config = ModelConfig(
        **{
            **asdict(frozen_encoder_base_model_config),
            "generator_kernel": 1,
            "generator_stride": 1,
            "viterbi_training": False,
        }
    )
    run_training_and_search(
        ".kernel1_stride1_bnres_frozenenc_aux9_fullsum",
        fullsum_frozen_encoder_kernel1_stride1_model_config,
        num_epochs=200,
        decoding_layers=(9,),
        batch_size=100,
        accum_grad_multiple_step=4,
        datasets=fullsum_train_data,
        train_step_args={
            "fsa_exporter_config_path": fsa_exporter_config,
            "label_smoothing_scale": 0.0,
        },
        include_native_ops=True,
    )


    fullsum_frozen_encoder_kernel4_stride2_share_sample_model_config = ModelConfig(
        **{
            **asdict(frozen_encoder_base_model_config),
            "generator_kernel": 4,
            "generator_stride": 2,
            "viterbi_training": False,
            "share_samples": True,
        }
    )
    run_training_and_search(
        ".kernel4_stride2_bnres_share_samples_frozenenc_aux9_fullsum",
        fullsum_frozen_encoder_kernel4_stride2_share_sample_model_config,
        num_epochs=200,
        decoding_layers=(9,),
        batch_size=150,
        accum_grad_multiple_step=4,
        datasets=fullsum_train_data,
        train_step_args={
            "fsa_exporter_config_path": fsa_exporter_config,
            "label_smoothing_scale": 0.0,
        },
        include_native_ops=True,
    )

    fullsum_frozen_encoder_kernel4_stride2_share_sample_r05_model_config = ModelConfig(
        **{
            **asdict(frozen_encoder_base_model_config),
            "generator_kernel": 4,
            "generator_stride": 2,
            "viterbi_training": False,
            "share_samples": True,
            "sampling_ratio": 0.5,
        }
    )
    run_training_and_search(
        ".kernel4_stride2_bnres_share_samples_r0.5_frozenenc_aux9_fullsum",
        fullsum_frozen_encoder_kernel4_stride2_share_sample_r05_model_config,
        num_epochs=200,
        decoding_layers=(9,),
        batch_size=150,
        accum_grad_multiple_step=4,
        datasets=fullsum_train_data,
        train_step_args={
            "fsa_exporter_config_path": fsa_exporter_config,
            "label_smoothing_scale": 0.0,
        },
        include_native_ops=True,
    )

    fullsum_frozen_encoder_kernel4_stride2_corrector1_5_model_config = ModelConfig(
        **{
            **asdict(frozen_encoder_base_model_config),
            "generator_kernel": 4,
            "generator_stride": 2,
            "viterbi_training": False,
            "sampling_ratio": 0.2,
            "ratio_corrector":1.5,

        }
    )
    run_training_and_search(
        ".kernel4_stride2_bnres_sample_0.2_ratio_correct1.5_frozenenc_aux9_fullsum",
        fullsum_frozen_encoder_kernel4_stride2_corrector1_5_model_config,
        num_epochs=200,
        decoding_layers=(9,),
        batch_size=150,
        accum_grad_multiple_step=4,
        datasets=fullsum_train_data,
        train_step_args={
            "fsa_exporter_config_path": fsa_exporter_config,
            "label_smoothing_scale": 0.0,
        },
        include_native_ops=True,
    )

    fullsum_trainable_encoder_kernel4_stride2_share_sample_r05_model_config = ModelConfig(
        **{
            **asdict(trainable_encoder_base_model_config),
            "generator_kernel": 4,
            "generator_stride": 2,
            "viterbi_training": False,
            "share_samples": True,
            "sampling_ratio": 0.5,
        }
    )
    run_training_and_search(
        ".kernel4_stride2_bnres_share_samples_r0.5_trainenc_aux9_fullsum",
        fullsum_trainable_encoder_kernel4_stride2_share_sample_r05_model_config,
        num_epochs=200,
        lm_scales=[1.2,1,0.8,0.6,0.4],
        decoding_layers=(9,),
        batch_size=150,
        accum_grad_multiple_step=4,
        datasets=fullsum_train_data,
        train_step_args={
            "fsa_exporter_config_path": fsa_exporter_config,
            "label_smoothing_scale": 0.0,
        },
        include_native_ops=True,
    )


py = eow_phon_phmm_ls960_wav2vec2_generative2
