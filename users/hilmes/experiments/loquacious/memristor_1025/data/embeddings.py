
from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.text.processing import HeadJob, PipelineJob

from i6_experiments.common.datasets.loquacious.corpus import get_ogg_zip_dict, get_bliss_corpus_dict

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datasets.audio import OggZipDataset
from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset
from i6_experiments.common.setups.returnn.datasets.base import MetaDataset


from i6_experiments.users.rossenbach.datasets.loquacious import PrepareLoquaciousTrainSmallSpeakerLabelsJob
from .common import DatasetSettings, TrainingDatasets, get_audio_raw_datastream
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE


def build_training_datasets(
    prefix: str,
    loquacious_key: str,
    settings: DatasetSettings,
) -> TrainingDatasets:
    """

    :param settings: settings object for the RETURNN data pipeline
    """
    ogg_zip_dict = get_ogg_zip_dict(prefix, returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    bliss_dict = get_bliss_corpus_dict()
    train_ogg = ogg_zip_dict[loquacious_key]

    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    speaker_label_job = PrepareLoquaciousTrainSmallSpeakerLabelsJob(
        hf_home_dir="/work/common/asr/loquacious/huggingface"
    )

    speaker_datastream = LabelDatastream(
        available_for_inference=True,
        vocab_size=speaker_label_job.out_num_speakers,
        vocab=speaker_label_job.out_speaker_dict,
    )
    datastreams = {
        "raw_audio": audio_datastream,
        "speaker_labels": speaker_datastream,
    }
    
    joint_speaker_dataset = HDFDataset(
        files=[speaker_label_job.out_speaker_hdf]
    )

    data_map = {
        "raw_audio": ("zip_dataset", "data"),
        "speaker_labels": ("speaker_dataset", "data")}

    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    def make_meta(dataset: OggZipDataset):
        return MetaDataset(
            data_map=data_map,
            datasets={
                "zip_dataset": dataset,
                "speaker_dataset": joint_speaker_dataset,
            },
            seq_order_control_dataset="zip_dataset"
        )


    train_small_segments = SegmentCorpusJob(bliss_dict["train.small"], 1).out_single_segment_files[1]
    shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
        segment_file=train_small_segments,
        split={"train": 0.98, "dev": 0.02},
        shuffle=True
    )
    train_segments = shuffle_segment_file_job.out_segments["train"]
    dev_segments = shuffle_segment_file_job.out_segments["dev"]
    dev_train_segments = HeadJob(train_segments, num_lines=1000).out

    train_ogg_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=None,
        segment_file=train_segments,
        partition_epoch=settings.train_partition_epoch,
        seq_ordering=settings.train_seq_ordering,
        **settings.train_additional_options
    )

    train_dataset = make_meta(train_ogg_dataset)

    dev_ogg_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=None,
        segment_file=dev_segments,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )

    dev_dataset = make_meta(dev_ogg_dataset)

    dev_train_ogg_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=None,
        segment_file=dev_train_segments,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )

    dev_train_dataset = make_meta(dev_train_ogg_dataset)

    return TrainingDatasets(
        train=train_dataset,
        cv=dev_dataset,
        devtrain=dev_train_dataset,
        datastreams=datastreams,
        prior=None,
    )
