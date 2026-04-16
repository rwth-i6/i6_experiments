
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any

from i6_core.returnn import CodeWrapper
from i6_experiments.common.datasets.librispeech import get_subword_nmt_bpe, get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import (
    AudioRawDatastream,
    ReturnnAudioRawOptions,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream

from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset

from ..default_tools import RETURNN_EXE, RETURNN_ROOT


@lru_cache()
def get_bpe_datastream(bpe_size, is_recog):
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param int bpe_size: size for the bpe labels
    :param bool is_recog: removes the UNK label when not in training
    :return:
    """
    # build dataset
    bpe_settings = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=bpe_size, unk_label="<unk>")
    bpe_targets = BpeDatastream(available_for_inference=False, bpe_settings=bpe_settings, use_unk_label=is_recog)
    return bpe_targets


@lru_cache()
def get_audio_raw_datastream(preemphasis=None):
    audio_datastream = AudioRawDatastream(
        available_for_inference=True, options=ReturnnAudioRawOptions(peak_normalization=True, preemphasis=preemphasis)
    )
    return audio_datastream


@lru_cache()
def build_extern_test_dataset_tuple(bliss, bpe_size=10000, use_raw_features=True, preemphasis=None):

    from i6_core.returnn.oggzip import BlissToOggZipJob

    ogg_zip = BlissToOggZipJob(
        bliss_corpus=bliss,
        no_conversion=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    ).out_ogg_zip

    from i6_core.corpus.convert import CorpusToTextDictJob

    test_reference_dict_file = CorpusToTextDictJob(bliss).out_dictionary

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=True)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream(preemphasis)  # TODO: experimenting
    else:
        raise NotImplementedError

    data_map = {"audio_features": ("zip_dataset", "data"), "bpe_labels": ("zip_dataset", "classes")}

    test_zip_dataset = OggZipDataset(
        files=[ogg_zip],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, test_reference_dict_file, bliss