from dataclasses import asdict
from typing import cast

import numpy as np
from sisyphus import tk

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_experiments.common.setups.returnn.datastreams.base import FeatureDatastream
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ....hsmm.data.common import build_training_datasets_with_hdf
from ...config import get_forward_config
from ...data.common import DatasetSettings
from ...data.phon import build_eow_phon_training_datasets, get_eow_bliss_and_zip
from ...default_tools import MINI_RETURNN_ROOT, RETURNN_EXE
from ...pipeline import training
from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_cfg import (
    LogMelFeatureExtractionV1Config,
    ModelConfig,
    SpecaugConfig,
    VGG4LayerActFrontendV1ConfigMod,
)
from .gmm_warmup_generative import (
    GMM_ALIGNMENT_LABEL_MAP,
    _build_gmm_alignment_training_datasets,
)


TOTAL_NUM_EPOCHS = 300
GMM_WARMUP_EPOCHS = 50
CTC_TARGET_REFRESH_EPOCHS = 10
CTC_TARGET_FILENAME = "ctc_soft_targets.hdf"


def _dump_ctc_soft_targets(
    *,
    name: str,
    dataset,
    checkpoint,
    network_module: str,
    net_args,
    num_classes: int,
    time_rqmt: int,
):
    forward_config = get_forward_config(
        network_module=network_module,
        net_args=net_args,
        config={
            "forward": dataset.as_returnn_opts(),
            "batch_size": 240 * 16000,
            "max_seqs": 200,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
        decoder=(
            "ctc.conformer_1023."
            "i6modelsV1_VGG4LayerActFrontendV1_v6_generative_ctc_soft_target_forward"
        ),
        decoder_args={
            "config": {
                "output_filename": CTC_TARGET_FILENAME,
                "num_classes": num_classes,
                "storage_dtype": "float16",
            }
        },
    )
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=forward_config,
        log_verbosity=5,
        mem_rqmt=24,
        time_rqmt=time_rqmt,
        device="gpu",
        cpu_rqmt=6,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=[CTC_TARGET_FILENAME],
    )
    forward_job.rqmt["gpu_mem"] = 24
    forward_job.add_alias(name + "/forward")
    tk.register_output(name + "/" + CTC_TARGET_FILENAME, forward_job.out_files[CTC_TARGET_FILENAME])
    return forward_job.out_files[CTC_TARGET_FILENAME]


def eow_phon_ls960_1023_gmm_warmup_iterative_frozen_ctc_nce():
    prefix_name = (
        "users/barkoczi/experiments/gen_ctc/"
        "ls960_ctc_eow_phon_gmm_warmup_iterative_frozen_ctc_nce"
    )
    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )
    ctc_train_data = build_eow_phon_training_datasets(
        prefix=prefix_name + "/ctc_targets",
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, ctc_train_data.datastreams["labels"])
    gmm_train_data = _build_gmm_alignment_training_datasets(
        prefix_name=prefix_name + "/gmm_hard_targets",
        settings=train_settings,
        label_datastream=label_datastream,
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
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1ConfigMod(
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
    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=label_datastream.vocab_size,
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
        sampling_type="batch",
        sampling_ratio=0.1,
        share_samples=False,
        ratio_corrector=1.0,
    )
    model_args = {"model_config_dict": asdict(model_config)}

    full_learning_rates = (
        list(np.linspace(7e-6, 5e-4, 144))
        + list(np.linspace(5e-4, 5e-5, 144))
        + list(np.linspace(5e-5, 1e-7, 12))
    )
    common_train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "batch_size": 240 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
        "gradient_clip_norm": 1.0,
    }

    gmm_network_module = (
        "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_gmm"
    )
    gmm_train_args = {
        "config": {
            **common_train_config,
            "learning_rates": full_learning_rates[:GMM_WARMUP_EPOCHS],
            "cleanup_old_models": {
                "keep_last_n": 1,
                "keep_best_n": 1,
                "keep": [GMM_WARMUP_EPOCHS],
            },
        },
        "network_module": gmm_network_module,
        "net_args": {
            **model_args,
            "alignment_label_map": GMM_ALIGNMENT_LABEL_MAP,
            "target_vocab": label_datastream.vocab,
            "alignment_subsampling_factor": 4,
        },
        "use_speed_perturbation": False,
        "debug": False,
    }
    gmm_training_name = (
        prefix_name
        + "/"
        + gmm_network_module
        + ".512dim_sub4_24gbgpu_5full-epochs_gmm-hard-targets-gennce"
    )
    gmm_train_job = training(
        gmm_training_name,
        gmm_train_data,
        gmm_train_args,
        num_epochs=GMM_WARMUP_EPOCHS,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )
    gmm_train_job.rqmt["gpu_mem"] = 24

    if (TOTAL_NUM_EPOCHS - GMM_WARMUP_EPOCHS) % CTC_TARGET_REFRESH_EPOCHS != 0:
        raise ValueError("The post-warmup epoch budget must be divisible by the target refresh interval")

    train_ogg = get_eow_bliss_and_zip(
        librispeech_key="train-other-960",
        g2p_librispeech_key="train-other-960",
        remove_unk_seqs=False,
    )[1]
    dev_clean_ogg = get_eow_bliss_and_zip(
        librispeech_key="dev-clean",
        g2p_librispeech_key="train-other-960",
        remove_unk_seqs=True,
    )[1]
    dev_other_ogg = get_eow_bliss_and_zip(
        librispeech_key="dev-other",
        g2p_librispeech_key="train-other-960",
        remove_unk_seqs=True,
    )[1]
    soft_target_datastream = FeatureDatastream(
        available_for_inference=False,
        feature_size=label_datastream.vocab_size + 1,
    )

    score_network_module = (
        "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first"
    )
    offline_train_network_module = (
        "ctc.conformer_1023."
        "i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first_offline_ctc"
    )
    checkpoint = gmm_train_job.out_checkpoints[GMM_WARMUP_EPOCHS]
    num_blocks = (TOTAL_NUM_EPOCHS - GMM_WARMUP_EPOCHS) // CTC_TARGET_REFRESH_EPOCHS
    block_jobs = []
    for block_idx in range(num_blocks):
        source_global_epoch = GMM_WARMUP_EPOCHS + block_idx * CTC_TARGET_REFRESH_EPOCHS
        target_start_epoch = source_global_epoch + 1
        target_end_epoch = source_global_epoch + CTC_TARGET_REFRESH_EPOCHS
        refresh_name = prefix_name + f"/refresh_{block_idx + 1:02d}_from_global_ep{source_global_epoch:03d}"
        train_hdf = _dump_ctc_soft_targets(
            name=refresh_name + "/train",
            dataset=ctc_train_data.prior,
            checkpoint=checkpoint,
            network_module=score_network_module,
            net_args=model_args,
            num_classes=label_datastream.vocab_size + 1,
            time_rqmt=168,
        )
        cv_hdf = _dump_ctc_soft_targets(
            name=refresh_name + "/cv",
            dataset=ctc_train_data.cv,
            checkpoint=checkpoint,
            network_module=score_network_module,
            net_args=model_args,
            num_classes=label_datastream.vocab_size + 1,
            time_rqmt=24,
        )
        frozen_target_data = build_training_datasets_with_hdf(
            train_ogg=train_ogg,
            dev_clean_ogg=dev_clean_ogg,
            dev_other_ogg=dev_other_ogg,
            label_datastream=label_datastream,
            settings=train_settings,
            hdf_file=train_hdf,
            train_hdf=train_hdf,
            cv_hdf=cv_hdf,
            devtrain_hdf=train_hdf,
            prior_hdf=train_hdf,
            hdf_datastream=soft_target_datastream,
            hdf_stream_name="ctc_soft_targets",
        )
        block_train_args = {
            "config": {
                **common_train_config,
                # RETURNN resumes at source_global_epoch + 1 and restores the adjacent
                # .opt.pt file, preserving AdamW state across target refreshes.
                "learning_rates": full_learning_rates[:target_end_epoch],
                "load": checkpoint,
                "start_epoch": source_global_epoch,
                "cleanup_old_models": {
                    "keep_last_n": 1,
                    "keep_best_n": 1,
                    "keep": [target_end_epoch],
                },
            },
            "network_module": offline_train_network_module,
            "net_args": model_args,
            # Frozen frame scores no longer match audio after speed perturbation.
            "use_speed_perturbation": False,
            "debug": False,
        }
        block_training_name = (
            prefix_name
            + "/"
            + offline_train_network_module
            + f".global_ep{target_start_epoch:03d}_{target_end_epoch:03d}"
            + f"_frozen_ctc_refresh_{block_idx + 1:02d}"
        )
        block_job = training(
            block_training_name,
            frozen_target_data,
            block_train_args,
            num_epochs=target_end_epoch,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        block_job.rqmt["gpu_mem"] = 24
        block_jobs.append(block_job)
        checkpoint = block_job.out_checkpoints[target_end_epoch]

    return block_jobs


py = eow_phon_ls960_1023_gmm_warmup_iterative_frozen_ctc_nce
