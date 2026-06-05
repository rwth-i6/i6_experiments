from dataclasses import asdict

from sisyphus import tk

from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...config import get_forward_config
from ...data.common import DatasetSettings
from ...data.phon import build_eow_phon_training_datasets_95_5_split, get_eow_vocab_datastream
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...pytorch_networks.hsmm.wav2vec2_hf_pca_dump_cfg import ModelConfig


def _make_feature_alignment_debug_job(*, prefix_name: str, forward_dataset, model_config: ModelConfig, output_filename: str):
    network_module = "hsmm.wav2vec2_hf_pca_dump"
    decoder_module = "hsmm.wav2vec2_feature_alignment_text_debug"

    returnn_config = get_forward_config(
        network_module=network_module,
        config={
            "forward": forward_dataset.as_returnn_opts(),
            "batch_size": 120 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "num_workers_per_gpu": 2,
        },
        net_args={"model_config_dict": asdict(model_config)},
        decoder=decoder_module,
        decoder_args={
            "config": {
                "output_filename": output_filename,
                "target_layer": -1,
                "max_seqs_to_dump": 8,
                "max_frames_to_dump": 12,
                "feature_precision": 4,
                "alignment_subsample_factor": 2,
            }
        },
        debug=False,
    )

    forward_job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=24,
        time_rqmt=8,
        device="gpu",
        cpu_rqmt=4,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=[output_filename],
    )
    forward_job.add_alias(prefix_name + "/forward_job")
    tk.register_output(prefix_name + f"/{output_filename}", forward_job.out_files[output_filename])


def debug_wav2vec2_feature_alignment_pairs():
    prefix_name = "example_setups/librispeech/feature_dump/ls960_wav2vec2_feature_alignment_debug"

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
        tk.Path(f"/work/asr3/zyang/share/joerg/generative/hsmm/alignments/lbs_mono_phone/train_960/alignment_{i}.hdf")
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

    model_config = ModelConfig(
        label_target_size=label_datastream.vocab_size,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=False,
        pca_dim=None,
        aux_ctc_loss_layers=[-1],
    )

    split_to_dataset = {
        "train": train_data.train,
        "devtrain": train_data.devtrain,
    }

    for split_name, dataset in split_to_dataset.items():
        _make_feature_alignment_debug_job(
            prefix_name=prefix_name + f"/{split_name}",
            forward_dataset=dataset,
            model_config=model_config,
            output_filename=f"{split_name}_feature_alignment_debug.txt",
        )


py = debug_wav2vec2_feature_alignment_pairs
