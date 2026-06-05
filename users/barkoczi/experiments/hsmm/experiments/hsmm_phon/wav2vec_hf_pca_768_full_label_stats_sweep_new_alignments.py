import copy
from dataclasses import asdict

from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...config import get_forward_config, get_pca_fit_config, get_pca_label_stats_config
from ...data.common import DatasetSettings
from ...data.phon import build_eow_phon_training_datasets_95_5_split, get_eow_vocab_datastream
from ...default_tools import MINI_RETURNN_ROOT, RETURNN_EXE
from ...pipeline import compute_hmm_frame_accuracy, compute_pca_label_stats, compute_pca_state
from ...pytorch_networks.hsmm.wav2vec2_hf_pca_dump_cfg import ModelConfig


def _dataset_opts_without_cache_manager(dataset):
    opts = copy.deepcopy(dataset.as_returnn_opts())

    def visit(value):
        if isinstance(value, dict):
            if value.get("use_cache_manager") is True:
                value["use_cache_manager"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(opts)
    return opts


def _stats_output_files(target_layer: int, feature_name: str):
    return [
        f"layer_{target_layer}_{feature_name}_pooled_variance.txt",
        f"layer_{target_layer}_{feature_name}_pooled_variance.npy",
        f"layer_{target_layer}_{feature_name}_pooled_variance.pt",
        f"layer_{target_layer}_{feature_name}_phoneme_feature_means.txt",
        f"layer_{target_layer}_{feature_name}_phoneme_feature_means.npy",
        f"layer_{target_layer}_{feature_name}_phoneme_feature_means.pt",
    ]


def _stats_forward_args(target_layer: int, feature_name: str, label_datastream, *, pca_dims=None):
    config = {
        "target_layer": target_layer,
        "returnn_vocab": label_datastream.vocab,
        "log_interval_batches": 10,
        "alignment_subsample_factor": 2,
    }
    if pca_dims is None:
        config.update(
            {
                "pooled_variance_filename": f"layer_{target_layer}_{feature_name}_pooled_variance.txt",
                "pooled_variance_npy_filename": f"layer_{target_layer}_{feature_name}_pooled_variance.npy",
                "pooled_variance_pt_filename": f"layer_{target_layer}_{feature_name}_pooled_variance.pt",
                "phoneme_means_filename": f"layer_{target_layer}_{feature_name}_phoneme_feature_means.txt",
                "phoneme_means_npy_filename": f"layer_{target_layer}_{feature_name}_phoneme_feature_means.npy",
                "phoneme_means_pt_filename": f"layer_{target_layer}_{feature_name}_phoneme_feature_means.pt",
            }
        )
    else:
        config.update(
            {
                "pca_dims": pca_dims,
                "pooled_variance_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_pooled_variance.txt",
                "pooled_variance_npy_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_pooled_variance.npy",
                "pooled_variance_pt_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_pooled_variance.pt",
                "phoneme_means_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_phoneme_feature_means.txt",
                "phoneme_means_npy_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_phoneme_feature_means.npy",
                "phoneme_means_pt_filename_template": f"layer_{target_layer}_pca_{{pca_dim}}_phoneme_feature_means.pt",
            }
        )
    return {"config": config}


def _run_accuracy_jobs(
    *,
    layer_prefix: str,
    target_layer: int,
    feature_name: str,
    train_data,
    network_module: str,
    model_config: ModelConfig,
    stats_outputs,
):
    pooled_variance_pt_filename = f"layer_{target_layer}_{feature_name}_pooled_variance.pt"
    phoneme_means_pt_filename = f"layer_{target_layer}_{feature_name}_phoneme_feature_means.pt"

    for decode_mode in ["framewise", "viterbi"]:
        accuracy_output_filename = f"layer_{target_layer}_{feature_name}_{decode_mode}_frame_accuracy.txt"
        accuracy_output_json_filename = f"layer_{target_layer}_{feature_name}_{decode_mode}_frame_accuracy.json"
        accuracy_returnn_config = get_forward_config(
            network_module=network_module,
            config={
                "forward": _dataset_opts_without_cache_manager(train_data.devtrain),
                "batch_size": 800 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "max_seqs": 240,
                "num_workers_per_gpu": 1,
            },
            net_args={"model_config_dict": asdict(model_config)},
            decoder="hsmm.wav2vec2_hmm_frame_accuracy",
            decoder_args={
                "config": {
                    "phoneme_means_pt": stats_outputs[phoneme_means_pt_filename],
                    "pooled_variance_pt": stats_outputs[pooled_variance_pt_filename],
                    "output_filename": accuracy_output_filename,
                    "output_json_filename": accuracy_output_json_filename,
                    "decode_mode": decode_mode,
                    "target_layer": target_layer,
                    "alignment_subsample_factor": 2,
                    "use_label_priors": True,
                    "self_loop_prob": 0.95,
                    "log_interval_batches": 10,
                }
            },
            debug=False,
        )
        _accuracy_outputs, accuracy_job = compute_hmm_frame_accuracy(
            prefix_name=layer_prefix + f"/{feature_name}/{decode_mode}_accuracy",
            returnn_config=accuracy_returnn_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            output_files=[accuracy_output_filename, accuracy_output_json_filename],
            mem_rqmt=24,
            use_gpu=True,
        )
        accuracy_job.rqmt["gpu_mem"] = 24


def compute_wav2vec2_large_pca_768_and_full_label_statistics_sweep():
    root_prefix = "example_setups/librispeech/feature_dump/ls960_wav2vec2_large_pca_768_full_label_stats_sweep_new_alignments"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=1,
        train_seq_ordering="sorted_reverse",
    )

    librispeech_key = "train-other-960"

    label_datastream = get_eow_vocab_datastream(
        prefix=root_prefix,
        g2p_librispeech_key=librispeech_key,
    )

    alignment_hdf = [
        tk.Path(f"/u/zyang/setups/mini/output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf")
        for i in range(1, 201)
    ]

    alignment_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=label_datastream.vocab,
        vocab_size=label_datastream.vocab_size,
        unk_label=label_datastream.unk_label,
    )

    train_data = build_eow_phon_training_datasets_95_5_split(
        prefix=root_prefix,
        librispeech_key=librispeech_key,
        settings=train_settings,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="data",
    )

    network_module = "hsmm.wav2vec2_hf_pca_dump"
    target_layers = [14, 15, 16]

    for target_layer in target_layers:
        layer_prefix = f"{root_prefix}/wav2vec2_large_layer_{target_layer}"

        common_config_kwargs = {
            "label_target_size": label_datastream.vocab_size,
            "hf_model_name": "facebook/wav2vec2-large",
            "pretrained": True,
            "freeze_feature_encoder": True,
            "freeze_encoder": True,
            "apply_spec_augment": False,
            "aux_ctc_loss_layers": [target_layer],
        }

        pca_model_config = ModelConfig(
            **common_config_kwargs,
            pca_dim=768,
        )
        pca_model_config.update_pca_during_training = True
        pca_returnn_config = get_pca_fit_config(
            forward_dataset=train_data.train,
            network_module=network_module,
            config={
                "forward": _dataset_opts_without_cache_manager(train_data.train),
                "batch_size": 400 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "max_seqs": 240,
                "num_workers_per_gpu": 1,
            },
            net_args={"model_config_dict": asdict(pca_model_config)},
            forward_args={"config": {"log_interval_batches": 10}},
            debug=False,
        )
        pca_outputs, pca_job = compute_pca_state(
            layer_prefix + "/fit_pca_768",
            pca_returnn_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            use_gpu=True,
        )
        pca_job.rqmt["gpu_mem"] = 24

        pca_stats_model_config = copy.deepcopy(pca_model_config)
        pca_stats_model_config.pca_state_path = pca_outputs["pca_state.pt"]
        pca_stats_model_config.update_pca_during_training = False
        pca_stats_returnn_config = get_pca_label_stats_config(
            forward_dataset=train_data.train,
            network_module=network_module,
            config={
                "forward": _dataset_opts_without_cache_manager(train_data.train),
                "batch_size": 800 * 16000,
                "max_seqs": 240,
                "num_workers_per_gpu": 1,
            },
            net_args={"model_config_dict": asdict(pca_stats_model_config)},
            forward_args=_stats_forward_args(
                target_layer,
                "pca_768",
                label_datastream,
                pca_dims=[768],
            ),
            debug=False,
        )
        pca_stats_outputs, pca_stats_job = compute_pca_label_stats(
            prefix_name=layer_prefix + "/pca_768_label_stats",
            returnn_config=pca_stats_returnn_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            output_files=_stats_output_files(target_layer, "pca_768"),
            mem_rqmt=24,
            use_gpu=True,
        )
        pca_stats_job.rqmt["gpu_mem"] = 24

        _run_accuracy_jobs(
            layer_prefix=layer_prefix,
            target_layer=target_layer,
            feature_name="pca_768",
            train_data=train_data,
            network_module=network_module,
            model_config=pca_stats_model_config,
            stats_outputs=pca_stats_outputs,
        )

        full_model_config = ModelConfig(
            **common_config_kwargs,
            pca_dim=None,
        )
        full_model_config.update_pca_during_training = False
        full_stats_returnn_config = get_pca_label_stats_config(
            forward_dataset=train_data.train,
            network_module=network_module,
            config={
                "forward": _dataset_opts_without_cache_manager(train_data.train),
                "batch_size": 800 * 16000,
                "max_seqs": 240,
                "num_workers_per_gpu": 1,
            },
            net_args={"model_config_dict": asdict(full_model_config)},
            forward_args=_stats_forward_args(
                target_layer,
                "full",
                label_datastream,
                pca_dims=None,
            ),
            debug=False,
        )
        full_stats_outputs, full_stats_job = compute_pca_label_stats(
            prefix_name=layer_prefix + "/full_label_stats",
            returnn_config=full_stats_returnn_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            output_files=_stats_output_files(target_layer, "full"),
            mem_rqmt=24,
            use_gpu=True,
        )
        full_stats_job.rqmt["gpu_mem"] = 24

        _run_accuracy_jobs(
            layer_prefix=layer_prefix,
            target_layer=target_layer,
            feature_name="full",
            train_data=train_data,
            network_module=network_module,
            model_config=full_model_config,
            stats_outputs=full_stats_outputs,
        )


py = compute_wav2vec2_large_pca_768_and_full_label_statistics_sweep
