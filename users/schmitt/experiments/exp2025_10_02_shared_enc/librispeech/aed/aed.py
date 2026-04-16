import copy
from dataclasses import asdict
import numpy as np
from typing import cast
from sisyphus import tk
from functools import partial

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.librispeech.aed import model_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import learning_rate_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import optimizer_configs
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datastreams.base import FeatureDatastream
from i6_experiments.users.schmitt.hdf import GetHdfDatasetStatisticsJob
from i6_experiments.users.schmitt.text.normalize import NormalizeLBSLMDataJob
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset, get_subepoch_dataset
from i6_experiments.users.schmitt.datasets.distrib_files import DistributedFilesDataset
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset
from i6_experiments.users.zeyer.returnn.alternate_batching import alternate_batching


from i6_core.text.processing import HeadJob, PipelineJob
from i6_core.tools.download import DownloadJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.returnn import CodeWrapper

from ..data.common import DatasetSettings, build_test_dataset, remove_audio_from_oggzip, TrainingDatasets
from ..data.spm import build_spm_training_datasets
from ..data.wav2vec import run_meta_experiments
from ..data.create_LMs import run_meta_experiments as run_meta_experiments_v2
from ..data.phon import build_eow_phon_training_datasets, get_eow_bliss_and_zip
from ..data.text import get_phonemized_lm_data, get_corpus_text, get_dev_text
from ..pipeline import training
from .tune_eval import build_base_report, eval_model
from ...default_tools import RETURNN_EXE, RETURNN_ROOT, MINI_RETURNN_ROOT
from ...report import generate_report
from ...recognition.aed.beam_search import DecoderConfig


def text_only_baseline():
    prefix_name = "experiments/librispeech/aed/ls960/baselines/text_only"

    raw_lm_corpus = DownloadJob(
        url="https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm_corpus/librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt"
    ).out_file
    tk.register_output("data/librispeech/lm/lbs_lm_minus_librivox.raw", raw_lm_corpus)

    wav2letter_repo = CloneGitRepositoryJob(
        "https://github.com/flashlight/wav2letter",
        commit="e5a4b62d87f15fde6a963d9ac174c8db8eb67fbc",
        checkout_folder_name="wav2letter",
    ).out_repository
    normalized_lm_data = NormalizeLBSLMDataJob(
        wav2letter_root=wav2letter_repo,
        wav2letter_python_exe=tk.Path("/work/asr4/schmitt/venvs/wav2letter_lm_corpus/bin/python3"),
        librispeech_lm_corpus=raw_lm_corpus,
    ).out_corpus_norm

    phoneme_train_hdfs, phoneme_vocab, lexicon_file, phoneme_file = get_phonemized_lm_data(
        alias="train-960h-filtered",
        text_file=normalized_lm_data,
        dump_hdf_concurrent=40,
    )
    phoneme_devtrain_hdfs, _, _, _ = get_phonemized_lm_data(
        text_file=normalized_lm_data,
        dump_hdf_concurrent=1,
        fixed_random_subset=3000,
    )
    phoneme_dev_hdfs, _, _, _ = get_phonemized_lm_data(
        alias="dev",
        # text_file=get_corpus_text("dev-other"),
        text_file=get_dev_text(),
        dump_hdf_concurrent=1,
        fixed_random_subset=3000,
        lexicon_file=lexicon_file,
        phoneme_file=phoneme_file,
    )

    phoneme_stats_job = GetHdfDatasetStatisticsJob(hdf_files=phoneme_train_hdfs)
    phoneme_stats_job.add_alias("data/librispeech/phoneme/ls960/phoneme_stats")
    tk.register_output(phoneme_stats_job.get_one_alias(), phoneme_stats_job.out_statistics)

    train_data = TrainingDatasets(
        train=DistributedFilesDataset(
            files=phoneme_train_hdfs,
            partition_epoch=40,
            get_subepoch_dataset=get_subepoch_dataset
        ),
        cv=HdfDataset(
            files=phoneme_dev_hdfs
        ),
        devtrain=HdfDataset(
            files=phoneme_devtrain_hdfs
        ),
        datastreams={
            "labels": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        }
    )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": RETURNN_ROOT,
    }

    epochs = 80
    batch_size = 10_000

    for model_alias, model_config, mask_prob, min_span, max_span, itc in [
        ("text_v1", copy.deepcopy(model_configs.text_v1), 0.3, 1, 3, False),
    ]:
        model_config["text_out_dim"] = train_data.datastreams["labels"].vocab_size

        network_module = (
            "pytorch_networks.conformer_aed_v1"
        )
        train_config = {
            **optimizer_configs.v1,
            **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep=epochs,),
            #############
            "batch_size": batch_size,
            "max_seq_length": {"data": 700},
            "accum_grad_multiple_step": 1,
            "gradient_clip_global_norm": 5.0,
            "__num_gpus": 1,  # 4, TODO: only for debugging: set to 1 GPU
            "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
        }
        # batch size, adamw, speed pert, gradient clip,
        train_args = {
            "config": train_config,
            "post_config": {
                "torch_log_memory_usage": True
            },
            "network_module": network_module,
            "train_step_module": "training.aed_ctc_train_step",
            "net_args": model_config,
            "train_args": {
                "ce_loss_scale": 0.5,
                "masked_ce_loss_scale": 1.0,
                "label_masking_opts": {
                    "mask_prob": mask_prob,
                    "min_span": min_span,
                    "max_span": max_span,
                }
            },
            "debug": True,
        }
        results = {}
        training_name = (
            prefix_name
            + "/"
            + network_module
            + f"/{model_alias}/"
            + f"bs{batch_size}_ep{epochs}_mask{mask_prob}-spans{min_span}-{max_span}"
        )
        train_job = training(
            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
        )

        if itc:
            train_job.hold()
            train_job.move_to_hpc = True
            train_job.rqmt["time_rqmt"] = 36


def audio_only_baseline_v1():
    prefix_name = "experiments/librispeech/aed/ls960/baselines/audio_only"

    wav2vec_segmented_features_train_hdf, clusters_train = run_meta_experiments(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        vad_concurrent=10,
        max_abs_value=1e4,
    )
    wav2vec_segmented_features_devtrain_hdf, _ = run_meta_experiments(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=1,
        vad_concurrent=10,
        fixed_random_subset=2832,  # number of seqs in dev_other  # 3000,
    )
    wav2vec_segmented_features_dev_other_hdf, _ = run_meta_experiments(
        librispeech_key="dev-other",
        existing_clusters=clusters_train,
    )

    audio_stats_job = GetHdfDatasetStatisticsJob(hdf_files=wav2vec_segmented_features_dev_other_hdf)
    audio_stats_job.add_alias("data/librispeech/wav2vec/dev-other/audio_stats")
    tk.register_output(audio_stats_job.get_one_alias(), audio_stats_job.out_statistics)

    audio_stats_job = GetHdfDatasetStatisticsJob(hdf_files=wav2vec_segmented_features_train_hdf)
    audio_stats_job.add_alias("data/librispeech/wav2vec/train/audio_stats")
    tk.register_output(audio_stats_job.get_one_alias(), audio_stats_job.out_statistics)

    train_data = TrainingDatasets(
        train=DistributedFilesDataset(
            files=wav2vec_segmented_features_train_hdf,
            partition_epoch=10,
            get_subepoch_dataset=get_subepoch_dataset
        ),
        cv=HdfDataset(
            files=wav2vec_segmented_features_dev_other_hdf
        ),
        devtrain=HdfDataset(
            files=wav2vec_segmented_features_devtrain_hdf
        ),
        datastreams={
            "features": FeatureDatastream(
                available_for_inference=True,
                feature_size=512,
            ),
        }
    )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": RETURNN_ROOT,
    }

    report = {}
    epochs = 80
    batch_size = 10_000
    default_decoder_config = DecoderConfig()

    for model_alias, model_config, mask_prob, min_span, max_span, epochs, audio_loss in [
        ("audio_v1/baseline", copy.deepcopy(model_configs.audio_v1), 0.3, 1, 3, 80, "mse"),
        ("audio_v1/baseline", copy.deepcopy(model_configs.audio_v1), 0.3, 1, 3, 800, "mse"),
        ("audio_v1/baseline", copy.deepcopy(model_configs.audio_v1), 0.3, 1, 3, 800, "l1"),
        ("audio_v2/baseline", copy.deepcopy(model_configs.audio_v2), 0.3, 1, 3, 800, "l1"),
        ("audio_v2/baseline", copy.deepcopy(model_configs.audio_v2), 0.3, 4, 8, 800, "l1"),
    ]:
        model_config["text_out_dim"] = None

        network_module = (
            "pytorch_networks.conformer_aed_v1"
        )
        train_config = {
            **optimizer_configs.v1,
            **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep=epochs,),
            #############
            "batch_size": batch_size,
            "max_seq_length": {"data": 700},
            "accum_grad_multiple_step": 1,
            "gradient_clip_global_norm": 5.0,
            "__num_gpus": 1,  # 4, TODO: only for debugging: set to 1 GPU
            "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
        }
        # batch size, adamw, speed pert, gradient clip,
        train_args = {
            "config": train_config,
            "post_config": {
                "torch_log_memory_usage": True
            },
            "network_module": network_module,
            "train_step_module": "training.aed_ctc_train_step",
            "net_args": model_config,
            "train_args": {
                "mse_loss_scale": 0.5,
                "masked_mse_loss_scale": 1.0,
                "eos_ce_loss_scale": 1.0,
                "feature_masking_opts": {
                    "mask_prob": mask_prob,
                    "min_span": min_span,
                    "max_span": max_span,
                },
            },
            "debug": True,
        }
        if audio_loss != "mse":
            train_args["train_args"]["audio_loss"] = audio_loss

        results = {}
        training_name = (
            prefix_name
            + "/"
            + network_module
            + f"/{model_alias}/"
            + f"bs{batch_size}_ep{epochs}_mask{mask_prob}-spans{min_span}-{max_span}_{audio_loss}-loss"
        )
        train_job = training(
            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
        )

    for model_alias, model_config, epochs, peak_lr, accum_grad in [
        # v2
        ("audio_v2/prob-0.3-1-3_l1", copy.deepcopy(model_configs.audio_v2), 800, 1e-4, 1),
        ("audio_v2/prob-0.3-1-3_l1", copy.deepcopy(model_configs.audio_v2), 800, 5e-4, 1),
        ("audio_v2/prob-0.3-1-3_l1", copy.deepcopy(model_configs.audio_v2), 800, 2e-4, 3),
        # v3
        ("audio_v3/prob-0.3-1-3_l1", copy.deepcopy(model_configs.audio_v3), 800, 1e-4, 1),
    ]:
        model_config["text_out_dim"] = None

        network_module = (
            "pytorch_networks.conformer_aed_v1"
        )
        train_config = {
            **optimizer_configs.v1,
            **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep=epochs, peak_lr=peak_lr),
            #############
            "batch_size": batch_size,
            "max_seq_length": {"data": 700},
            "accum_grad_multiple_step": accum_grad,
            "gradient_clip_global_norm": 5.0,
            "__num_gpus": 1,  # 4, TODO: only for debugging: set to 1 GPU
            "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
        }
        # batch size, adamw, speed pert, gradient clip,
        train_args = {
            "config": train_config,
            "post_config": {
                "torch_log_memory_usage": True
            },
            "network_module": network_module,
            "train_step_module": "training.aed_ctc_train_step",
            "net_args": model_config,
            "train_args": {
                "mse_loss_scale": 0.5,
                "masked_mse_loss_scale": 1.0,
                "eos_ce_loss_scale": 1.0,
                "feature_masking_opts": {
                    "mask_prob": 0.3,
                    "min_span": 1,
                    "max_span": 3,
                },
            },
            "debug": True,
        }
        if audio_loss != "mse":
            train_args["train_args"]["audio_loss"] = "l1"

        results = {}
        training_name = (
            prefix_name
            + "/"
            + network_module
            + f"/{model_alias}/"
            + f"bs{batch_size}_ep{epochs}_peak-lr-{peak_lr}_accum-{accum_grad}"
        )
        train_job = training(
            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
        )


def text_audio_baseline():
    prefix_name = "experiments/librispeech/aed/ls960/baselines/text_audio"

    wav2vec_segmented_features_train_hdf, clusters_train, _ = run_meta_experiments(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        vad_concurrent=10,
        max_abs_value=1e4,
    )
    wav2vec_segmented_features_devtrain_hdf, _, _ = run_meta_experiments(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=1,
        vad_concurrent=10,
        fixed_random_subset=2832,  # number of seqs in dev_other  # 3000,
    )
    wav2vec_segmented_features_dev_other_hdf, _, _ = run_meta_experiments(
        librispeech_key="dev-other",
        existing_clusters=clusters_train,
    )

    raw_lm_corpus = DownloadJob(
        url="https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm_corpus/librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt"
    ).out_file
    tk.register_output("data/librispeech/lm/lbs_lm_minus_librivox.raw", raw_lm_corpus)

    wav2letter_repo = CloneGitRepositoryJob(
        "https://github.com/flashlight/wav2letter",
        commit="e5a4b62d87f15fde6a963d9ac174c8db8eb67fbc",
        checkout_folder_name="wav2letter",
    ).out_repository
    normalized_lm_data = NormalizeLBSLMDataJob(
        wav2letter_root=wav2letter_repo,
        wav2letter_python_exe=tk.Path("/work/asr4/schmitt/venvs/wav2letter_lm_corpus/bin/python3"),
        librispeech_lm_corpus=raw_lm_corpus,
    ).out_corpus_norm

    phoneme_train_hdfs, phoneme_vocab, lexicon_file, phoneme_file = get_phonemized_lm_data(
        alias="train-960h-filtered",
        text_file=normalized_lm_data,
        dump_hdf_concurrent=40,
    )
    phoneme_devtrain_hdfs, _, _, _ = get_phonemized_lm_data(
        text_file=normalized_lm_data,
        dump_hdf_concurrent=1,
        fixed_random_subset=3000,
    )
    phoneme_dev_hdfs, _, _, _ = get_phonemized_lm_data(
        alias="dev",
        # text_file=get_corpus_text("dev-other"),
        text_file=get_dev_text(),
        dump_hdf_concurrent=1,
        fixed_random_subset=3000,
        lexicon_file=lexicon_file,
        phoneme_file=phoneme_file,
        # vocab_size=100?
    )

    audio_stats_job = GetHdfDatasetStatisticsJob(hdf_files=wav2vec_segmented_features_train_hdf)
    audio_stats_job.add_alias("data/librispeech/wav2vec/ls960/audio_stats")
    tk.register_output(audio_stats_job.get_one_alias(), audio_stats_job.out_statistics)

    phoneme_stats_job = GetHdfDatasetStatisticsJob(hdf_files=phoneme_train_hdfs)
    phoneme_stats_job.add_alias("data/librispeech/phoneme/ls960/phoneme_stats")
    tk.register_output(phoneme_stats_job.get_one_alias(), phoneme_stats_job.out_statistics)

    train_data = TrainingDatasets(
        train=CombinedDataset(
            datasets={
                "phonemes": DistributedFilesDataset(
                    files=phoneme_train_hdfs,
                    partition_epoch=40,
                    get_subepoch_dataset=get_subepoch_dataset
                ),
                "audio_features": DistributedFilesDataset(
                    # upsample audio data to match phoneme data size
                    # phoneme data has 58x more frames than audio data
                    # set partition_epoch 8x smaller and use 7x replication -> 56x
                    files=wav2vec_segmented_features_train_hdf * 7,
                    partition_epoch=5,
                    get_subepoch_dataset=get_subepoch_dataset
                ),
            },
            data_map={
                ("phonemes", "data"): "phon_indices",
                ("audio_features", "data"): "data",
            },
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        cv=CombinedDataset(
            datasets={
                "phonemes": HdfDataset(
                    files=phoneme_dev_hdfs
                ),
                "audio_features": HdfDataset(
                    files=wav2vec_segmented_features_dev_other_hdf
                ),
            },
            data_map={
                ("phonemes", "data"): "phon_indices",
                ("audio_features", "data"): "data",
            },
            seq_ordering="sorted",
            partition_epoch=1,
        ),
        devtrain=CombinedDataset(
            datasets={
                "phonemes": HdfDataset(
                    files=phoneme_devtrain_hdfs
                ),
                "audio_features": HdfDataset(
                    files=wav2vec_segmented_features_devtrain_hdf
                ),
            },
            data_map={
                ("phonemes", "data"): "phon_indices",
                ("audio_features", "data"): "data",
            },
            seq_ordering="sorted",
            partition_epoch=1,
        ),
        datastreams={
            "features": FeatureDatastream(
                available_for_inference=True,
                feature_size=512,
            ),
            "labels": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        }
    )

    # dev_dataset_tuples = {}
    # for testset in ["dev-clean", "dev-other"]:
    #     dev_dataset_tuples[testset] = build_test_dataset(
    #         dataset_key=testset,
    #         settings=train_settings,
    #     )
    #
    # test_dataset_tuples = {}
    # for testset in ["test-clean", "test-other"]:
    #     test_dataset_tuples[testset] = build_test_dataset(
    #         dataset_key=testset,
    #         settings=train_settings,
    #     )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": RETURNN_ROOT,
    }

    report = {}
    epochs = 80
    batch_size = {"data": 10_000, "phon_indices": 10_000}
    default_decoder_config = DecoderConfig()

    for model_alias, model_config, mask_prob, min_span, max_span, itc, batch_size in [
        ("text_audio_v1", copy.deepcopy(model_configs.audio_text_v1), 0.3, 1, 3, False, 10_000),
        ("text_audio_v1", copy.deepcopy(model_configs.audio_text_v1), 0.3, 1, 3, False, 9_000),
        # ("text_audio_v1", copy.deepcopy(model_configs.audio_text_v1), 0.3, 1, 3, True, 90_000),
    ]:
        model_config["text_out_dim"] = train_data.datastreams["labels"].vocab_size

        network_module = (
            "pytorch_networks.conformer_aed_v1"
        )
        train_config = {
            **optimizer_configs.v1,
            **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep=epochs,),
            #############
            "batch_size": {"data": batch_size, "phon_indices": batch_size},
            "torch_batching": CodeWrapper("alternate_batching"),
            "max_seq_length": {"data": 700, "phon_indices": 700},
            "accum_grad_multiple_step": 2,
            "gradient_clip_global_norm": 5.0,
            "__num_gpus": 1,  # 4, TODO: only for debugging: set to 1 GPU
            "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
        }

        # batch size, adamw, speed pert, gradient clip,
        train_args = {
            "config": train_config,
            "post_config": {
                "torch_log_memory_usage": True,
                "tensorboard_opts": {
                    # uneven so that both text and audio losses get logged (alternated batching)
                    "log_every_n_train_steps": 101,
                }
            },
            "python_prolog": [
                "from i6_experiments.users.zeyer.returnn.alternate_batching import alternate_batching"
            ],
            "network_module": network_module,
            "train_step_module": "training.aed_ctc_train_step",
            "net_args": model_config,
            "train_args": {
                "ce_loss_scale": 0.5,
                "masked_ce_loss_scale": 1.0,
                "mse_loss_scale": 0.5,
                "masked_mse_loss_scale": 1.0,
                "eos_ce_loss_scale": 1.0,
                "label_masking_opts": {
                    "mask_prob": mask_prob,
                    "min_span": min_span,
                    "max_span": max_span,
                },
                "feature_masking_opts": {
                    "mask_prob": mask_prob,
                    "min_span": min_span,
                    "max_span": max_span,
                },
            },
            "debug": True,
        }
        results = {}
        training_name = (
            prefix_name
            + "/"
            + network_module
            + f"/{model_alias}/"
            + f"bs{batch_size}_ep{epochs}_mask{mask_prob}-spans{min_span}-{max_span}"
        )
        train_job = training(
            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
        )

        if itc:
            train_job.hold()
            train_job.move_to_hpc = True
