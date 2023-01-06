from sisyphus import tk

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any

from i6_core.returnn import CodeWrapper
from i6_experiments.common.datasets.librispeech import get_subword_nmt_bpe, get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import AudioRawDatastream, \
    ReturnnAudioRawOptions
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream

from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset


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
    bpe_settings = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=bpe_size, unk_label='<unk>')
    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog
    )
    return bpe_targets


@lru_cache()
def get_audio_raw_datastream():
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=True)
    )
    return audio_datastream


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    extern_data: Dict[str, Dict[str, Any]]

@tk.block()
def build_training_datasets(
        returnn_python_exe, returnn_root, output_path,
        bpe_size=10000,
        partition_epoch=20,
        use_curicculum=True,
        seq_ordering="laplace:.1000",
        link_speed_perturbation=False,
        use_raw_features=False
    ):
    """

    :param returnn_python_exe:
    :param returnn_root:
    :param output_path:
    :param int bpe_size:
    :param int bpe_partition_epoch:
    :param str seq_ordering:
    :return:
    """
    ogg_zip_dict = get_ogg_zip_dict("corpora")
    train_clean_960_ogg = ogg_zip_dict['train-other-960']
    dev_clean_ogg = ogg_zip_dict['dev-clean']
    dev_other_ogg = ogg_zip_dict['dev-other']

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=False)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream()
    else:
        raise NotImplementedError

    extern_data = {
        'audio_features': audio_datastream.as_returnn_extern_data_opts(),
        'bpe_labels': train_bpe_datastream.as_returnn_extern_data_opts()
    }

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}


    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    if link_speed_perturbation:
        training_audio_opts["pre_process"] = CodeWrapper("speed_pert")

    train_zip_dataset = OggZipDataset(
        files=train_clean_960_ogg,
        audio_options=training_audio_opts,
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=partition_epoch,
        seq_ordering=seq_ordering,
        additional_options={"epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}} if use_curicculum else None}  # still hardcoded, future work
    )
    train_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": train_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    from i6_experiments.users.rossenbach.datasets.librispeech import get_mixed_cv_segments
    cv_zip_dataset = OggZipDataset(
        files=[dev_clean_ogg, dev_other_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse"
    )
    cv_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": cv_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    devtrain_zip_dataset = OggZipDataset(
        files=train_clean_960_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": devtrain_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        extern_data=extern_data
    )


@lru_cache()
def build_test_dataset(dataset_key, returnn_python_exe, returnn_root, output_path, bpe_size=10000,
    use_raw_features=False):

    ogg_zip_dict = get_ogg_zip_dict("corpora")
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]
    from i6_core.corpus.convert import CorpusToTextDictJob
    test_reference_dict_file = CorpusToTextDictJob(bliss_dict[dataset_key]).out_dictionary

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=True)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream()
    else:
        raise NotImplementedError

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse"
    )
    test_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": test_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, test_reference_dict_file