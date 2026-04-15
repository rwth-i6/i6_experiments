from sisyphus import tk

from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...config import get_forward_config
from ...data.common import DatasetSettings, build_oggzip_dataset_with_optional_hdf, get_audio_raw_datastream
from ...data.hdf_seq_whitelist import ExtractSeqListFromHDFJob
from ...data.phon import (
    build_eow_phon_training_datasets_95_5_split,
    get_eow_bliss_and_zip,
    build_eow_phon_test_dataset_with_optional_hdf,
    get_eow_vocab_datastream,
)
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT


def _make_dataset_debug_forward_job(*, prefix_name: str, forward_dataset):
    network_module = "ctc.dataset_debug_v1"
    returnn_config = get_forward_config(
        network_module=network_module,
        config={"forward": forward_dataset.as_returnn_opts(), "batch_size": 50 * 16000},
        net_args={},
        decoder=network_module,
        decoder_args={"config": {"output_filename": "dataset_debug.txt", "alignment_key": "alignments"}},
        debug=False,
    )

    forward_job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=8,
        time_rqmt=4,
        device="cpu",
        cpu_rqmt=2,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=["dataset_debug.txt"],
    )
    forward_job.add_alias(prefix_name + "/forward_job")
    tk.register_output(prefix_name + "/dataset_debug.txt", forward_job.out_files["dataset_debug.txt"])


def eow_phon_dataset_with_optional_hdf_debug():
    prefix_name = (
        "example_setups/librispeech/ctc_rnnt_standalone_2024/"
        "ls960_ctc_eow_phon_dataset_with_optional_hdf_debug"
    )

    settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=1,
        train_seq_ordering="sorted_reverse",
    )

    librispeech_key = "train-other-960"
    dataset_key = "dev-clean"

    label_datastream = get_eow_vocab_datastream(
        prefix=prefix_name,
        g2p_librispeech_key=librispeech_key,
    )

    alignment_hdf = [
        tk.Path(f"/work/asr3/zyang/share/joerg/generative/hsmm/alignments/lbs_mono_phone/dev_clean/alignment_{i}.hdf")
        for i in range(1, 11)
    ]
    alignment_seq_whitelist = ExtractSeqListFromHDFJob(alignment_hdf).out_seq_list

    alignment_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=label_datastream.vocab,
        vocab_size=label_datastream.vocab_size,
        unk_label=label_datastream.unk_label,
    )

    forward_dataset, _bliss = build_eow_phon_test_dataset_with_optional_hdf(
        dataset_key=dataset_key,
        settings=settings,
        label_datastream=label_datastream,
        g2p_librispeech_key=librispeech_key,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="classes",
        segment_file=alignment_seq_whitelist,
    )

    _make_dataset_debug_forward_job(prefix_name=prefix_name, forward_dataset=forward_dataset)


def eow_phon_training_dataset_with_optional_hdf_debug():
    prefix_name = (
        "example_setups/librispeech/ctc_rnnt_standalone_2024/"
        "ls960_ctc_eow_phon_training_dataset_with_optional_hdf_debug"
    )

    settings = DatasetSettings(
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

    # Replace these with the actual training alignment HDFs when available.
    alignment_hdf = [
        tk.Path(f"/work/asr3/zyang/share/joerg/generative/hsmm/alignments/lbs_mono_phone/train_960/alignment_{i}.hdf") for i in range (1,201)
    ]
    alignment_seq_whitelist = ExtractSeqListFromHDFJob(alignment_hdf).out_seq_list

    alignment_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=label_datastream.vocab,
        vocab_size=label_datastream.vocab_size,
        unk_label=label_datastream.unk_label,
    )

    _train_bliss, train_ogg = get_eow_bliss_and_zip(
        librispeech_key=librispeech_key,
        g2p_librispeech_key=librispeech_key,
        remove_unk_seqs=False,
    )
    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)
    forward_dataset, _datastreams = build_oggzip_dataset_with_optional_hdf(
        ogg_files=train_ogg,
        audio_datastream=audio_datastream,
        label_datastream=label_datastream,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="classes",
        partition_epoch=1,
        segment_file=alignment_seq_whitelist,
        seq_ordering="sorted_reverse",
    )

    _make_dataset_debug_forward_job(prefix_name=prefix_name, forward_dataset=forward_dataset)


def eow_phon_training_dataset_95_5_split_with_optional_hdf_debug():
    prefix_name = (
        "example_setups/librispeech/ctc_rnnt_standalone_2024/"
        "ls960_ctc_eow_phon_training_dataset_95_5_split_with_optional_hdf_debug"
    )

    settings = DatasetSettings(
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

    training_datasets = build_eow_phon_training_datasets_95_5_split(
        prefix=prefix_name,
        librispeech_key=librispeech_key,
        settings=settings,
        hdf_file=alignment_hdf,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="classes",
    )

    _make_dataset_debug_forward_job(prefix_name=prefix_name + "/train", forward_dataset=training_datasets.train)
    _make_dataset_debug_forward_job(prefix_name=prefix_name + "/devtrain", forward_dataset=training_datasets.devtrain)


py = eow_phon_training_dataset_95_5_split_with_optional_hdf_debug
