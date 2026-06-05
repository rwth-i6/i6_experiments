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


def compute_wav2vec2_pca_label_statistics():
    prefix_name = "example_setups/librispeech/feature_dump/ls960_wav2vec2_pca512_label_stats"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=1,
        train_seq_ordering="sorted_reverse",
    )

    librispeech_key = "train-other-960"

    label_datastream = get_eow_vocab_datastream(
        prefix=prefix_name,
        g2p_librispeech_key=librispeech_key,
    )

    alignment_hdf = [
        tk.Path(f"/home/joerg/dev/a/i6_setups/gen/alignments/alignment_{i}.hdf")
        for i in range(1, 201)
    ]

    alignment_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=label_datastream.vocab,
        vocab_size=label_datastream.vocab_size,
        unk_label=label_datastream.unk_label,
    )

    train_data = build_eow_phon_training_datasets_95_5_split(
        prefix=prefix_name,
        librispeech_key=librispeech_key,
        settings=train_settings,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="classes",
    )

    base_model_config = ModelConfig(
        label_target_size=label_datastream.vocab_size,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=False,
        pca_dim=512,
        aux_ctc_loss_layers=[3, 6, 9, -1],
    )

    network_module = "hsmm.wav2vec2_hf_pca_dump"
    pca_model_config = copy.deepcopy(base_model_config)
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
        prefix_name + "/fit_pca",
        pca_returnn_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        use_gpu=True,
    )
    pca_job.rqmt["gpu_mem"] = 24

    stats_model_config = copy.deepcopy(base_model_config)
    stats_model_config.pca_state_path = pca_outputs["pca_state.pt"]
    stats_model_config.update_pca_during_training = False

    pooled_variance_filename = "pooled_variance.txt"
    pooled_variance_npy_filename = "pooled_variance.npy"
    pooled_variance_pt_filename = "pooled_variance.pt"
    phoneme_means_filename = "phoneme_feature_means.txt"
    phoneme_means_npy_filename = "phoneme_feature_means.npy"
    phoneme_means_pt_filename = "phoneme_feature_means.pt"

    stats_returnn_config = get_pca_label_stats_config(
        forward_dataset=train_data.train,
        network_module=network_module,
        config={
            "forward": _dataset_opts_without_cache_manager(train_data.train),
            "batch_size": 400 * 16000,
            "max_seqs": 240,
            "num_workers_per_gpu": 1,
        },
        net_args={"model_config_dict": asdict(stats_model_config)},
        forward_args={
            "config": {
                "pooled_variance_filename": pooled_variance_filename,
                "pooled_variance_npy_filename": pooled_variance_npy_filename,
                "pooled_variance_pt_filename": pooled_variance_pt_filename,
                "phoneme_means_filename": phoneme_means_filename,
                "phoneme_means_npy_filename": phoneme_means_npy_filename,
                "phoneme_means_pt_filename": phoneme_means_pt_filename,
                "target_layer": -1,
                "returnn_vocab": label_datastream.vocab,
                "log_interval_batches": 10,
                "alignment_subsample_factor": 2,
            }
        },
        debug=False,
    )

    _stats_outputs, stats_job = compute_pca_label_stats(
        prefix_name=prefix_name + "/label_stats",
        returnn_config=stats_returnn_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=[
            pooled_variance_filename,
            pooled_variance_npy_filename,
            pooled_variance_pt_filename,
            phoneme_means_filename,
            phoneme_means_npy_filename,
            phoneme_means_pt_filename,
        ],
        mem_rqmt=24,
        use_gpu=True,
    )
    stats_job.rqmt["gpu_mem"] = 24

    for decode_mode in ["framewise", "viterbi"]:
        accuracy_output_filename = f"{decode_mode}_frame_accuracy.txt"
        accuracy_output_json_filename = f"{decode_mode}_frame_accuracy.json"
        accuracy_returnn_config = get_forward_config(
            network_module=network_module,
            config={
                "forward": _dataset_opts_without_cache_manager(train_data.train),
                "batch_size": 400 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "max_seqs": 240,
                "num_workers_per_gpu": 1,
            },
            net_args={"model_config_dict": asdict(stats_model_config)},
            decoder="hsmm.wav2vec2_hmm_frame_accuracy",
            decoder_args={
                "config": {
                    "phoneme_means_pt": _stats_outputs[phoneme_means_pt_filename],
                    "pooled_variance_pt": _stats_outputs[pooled_variance_pt_filename],
                    "output_filename": accuracy_output_filename,
                    "output_json_filename": accuracy_output_json_filename,
                    "decode_mode": decode_mode,
                    "target_layer": -1,
                    "alignment_subsample_factor": 2,
                    "use_label_priors": True,
                    "self_loop_prob": 0.95,
                    "log_interval_batches": 10,
                }
            },
            debug=False,
        )
        _accuracy_outputs, accuracy_job = compute_hmm_frame_accuracy(
            prefix_name=prefix_name + f"/{decode_mode}_accuracy",
            returnn_config=accuracy_returnn_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            output_files=[accuracy_output_filename, accuracy_output_json_filename],
            mem_rqmt=24,
            use_gpu=True,
        )
        accuracy_job.rqmt["gpu_mem"] = 24


py = compute_wav2vec2_pca_label_statistics
