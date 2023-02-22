from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any

from i6_core.returnn import CodeWrapper
from i6_experiments.common.datasets.switchboard.bpe import get_subword_nmt_bpe
from i6_experiments.common.datasets.switchboard.corpus_train import get_spoken_form_train_bliss_corpus_ldc_ogg
from i6_experiments.common.datasets.switchboard.corpus_eval import get_test_data_dict, get_test_data_ogg_dict, get_hub5e00_cv_ogg
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
    bpe_settings = get_subword_nmt_bpe(bpe_size=bpe_size, unk_label='<unk>')
    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog
    )
    return bpe_targets


@lru_cache()
def get_audio_raw_datastream(preemphasis=None):
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=True, preemphasis=preemphasis)
    )
    return audio_datastream


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    extern_data: Dict[str, Dict[str, Any]]


def build_training_datasets(
        bpe_size=500,
        partition_epoch=6,
        epoch_wise_filter=None,
        seq_ordering="laplace:.1000",
        link_speed_perturbation=False,
        use_raw_features=False,
        preemphasis=None,  # TODO: by default is None, but we want to experiment
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

    # with lowercase + spoken form of numbers instead of digits
    train_ogg = get_spoken_form_train_bliss_corpus_ldc_ogg()

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=False)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream(preemphasis)
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

    additional_opts = {}
    if epoch_wise_filter:
        additional_opts['epoch_wise_filter'] = {}
        for fr, to, max_mean_len in epoch_wise_filter:
            additional_opts['epoch_wise_filter'][(fr, to)] = {"max_mean_len": max_mean_len}

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=partition_epoch,
        seq_ordering=seq_ordering,
        additional_options=additional_opts,
    )
    train_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": train_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    cv_zip_dataset = OggZipDataset(
        files=get_hub5e00_cv_ogg(),
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse"
    )
    cv_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": cv_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
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
def build_test_dataset(dataset_key, bpe_size=500, use_raw_features=False, preemphasis=None):

    ogg_zip_dict = get_test_data_ogg_dict()
    test_ogg = ogg_zip_dict[dataset_key]

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=True)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream(preemphasis)
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

    return test_dataset, get_test_data_dict()[dataset_key]