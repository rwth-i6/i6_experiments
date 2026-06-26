"""
Dataset helpers for the BPE-based training
"""
from sisyphus import tk

from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.bpe import CreateBPELexiconJob

from i6_experiments.common.datasets.loquacious.corpus import get_ogg_zip_dict
from i6_experiments.common.setups.returnn.datasets import OggZipDataset

from .common import DatasetSettings, get_audio_raw_datastream, make_multi_proc
from ...default_tools import RETURNN_ROOT, RETURNN_EXE


def build_audio_only_training_dataset(
    prefix: str,
    loquacious_key: str,
    settings: DatasetSettings,
) -> OggZipDataset:
    """

    :param loquacious_key: which librispeech corpus to use for bpe training
    :param settings: configuration object for the dataset pipeline
    """

    ogg_zip_dict = get_ogg_zip_dict(prefix, returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[loquacious_key]

    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    training_audio_opts = audio_datastream.as_returnn_audio_opts()

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=None,
        partition_epoch=settings.train_partition_epoch,
        seq_ordering=settings.train_seq_ordering,
        additional_options=settings.train_additional_options,
    )
    train_dataset = make_multi_proc(train_zip_dataset, num_workers=settings.num_workers)

    return train_dataset
