import copy
from sisyphus import tk

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.librispeech.aed import model_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import learning_rate_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import optimizer_configs
from i6_experiments.common.setups.returnn.datastreams.base import FeatureDatastream
from i6_experiments.users.schmitt.hdf import GetHdfDatasetStatisticsJob
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset, get_subepoch_dataset
from i6_experiments.users.schmitt.datasets.distrib_files import DistributedFilesDataset

from ..data.common import DatasetSettings, TrainingDatasets
from ..data.wav2vec import run_meta_experiments
from ..pipeline import training
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
# from ...recognition.aed.beam_search import DecoderConfig


def py():
    prefix_name = "experiments/librispeech/aed/ls960/baselines/audio_only"

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

    audio_stats_job = GetHdfDatasetStatisticsJob(hdf_files=wav2vec_segmented_features_dev_other_hdf)
    audio_stats_job.add_alias("data/librispeech/wav2vec/dev-other/audio_feature_stats")
    tk.register_output(audio_stats_job.get_one_alias(), audio_stats_job.out_statistics)

    audio_stats_job = GetHdfDatasetStatisticsJob(hdf_files=wav2vec_segmented_features_train_hdf)
    audio_stats_job.add_alias("data/librispeech/wav2vec/train/audio_feature_stats")
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
    batch_size = 10_000

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

        training_name = (
            prefix_name
            + "/"
            + network_module
            + f"/{model_alias}/"
            + f"bs{batch_size}_ep{epochs}_mask{mask_prob}-spans{min_span}-{max_span}_{audio_loss}-loss"
        )
        training(
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

        training_name = (
            prefix_name
            + "/"
            + network_module
            + f"/{model_alias}/"
            + f"bs{batch_size}_ep{epochs}_peak-lr-{peak_lr}_accum-{accum_grad}"
        )
        training(
            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
        )
