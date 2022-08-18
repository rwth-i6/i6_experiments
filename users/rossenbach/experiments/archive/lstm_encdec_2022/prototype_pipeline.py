import os.path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_corpus_dict

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import get_default_asr_audio_datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset, MetaDataset, OggZipDataset

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_bpe
from i6_experiments.users.rossenbach.datasets.librispeech import get_mixed_cv_segments


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
    bpe_settings = get_librispeech_bpe(corpus_key="train-clean-100", bpe_size=bpe_size, unk_label='<unk>')
    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog
    )
    return bpe_targets


@lru_cache()
def get_audio_datastream(returnn_python_exe, returnn_root, output_path):
    ogg_zip_dict = get_ogg_zip_dict("corpora")
    train_clean_100_ogg = ogg_zip_dict['train-clean-100']

    audio_datastream = get_default_asr_audio_datastream(
        statistics_ogg_zips=[train_clean_100_ogg],
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        alias_path=output_path,
    )

    return audio_datastream

@dataclass(frozen=True)
class TrainingDatasets:
    train: GenericDataset
    cv: GenericDataset
    devtrain: GenericDataset
    datastreams: Dict[str, Datastream]


@lru_cache()
def build_training_datasets(
        returnn_python_exe, returnn_root, output_path,
        bpe_size=2000,
        partition_epoch=3,
        seq_ordering="laplace:.1000"):
    """

    :param returnn_python_exe:
    :param returnn_root:
    :param output_path:
    :param int bpe_size:
    :param str seq_ordering:
    :return:
    """
    ogg_zip_dict = get_ogg_zip_dict("corpora")
    train_clean_100_ogg = ogg_zip_dict['train-clean-100']
    dev_clean_ogg = ogg_zip_dict['dev-clean']
    dev_other_ogg = ogg_zip_dict['dev-other']

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=False)

    audio_datastream = get_audio_datastream(
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_path=output_path,
    )

    datastreams = {
        'audio_features': audio_datastream,
        'bpe_labels': train_bpe_datastream
    }

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}

    train_zip_dataset = OggZipDataset(
        path=train_clean_100_ogg,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=partition_epoch,
        seq_ordering=seq_ordering,
        other_opts={"epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}}  # still hardcoded, future work
    )
    train_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": train_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    cv_zip_dataset = OggZipDataset(
        path=[dev_clean_ogg, dev_other_ogg],
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse"
    )
    cv_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": cv_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    devtrain_zip_dataset = OggZipDataset(
        path=train_clean_100_ogg,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
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
        datastreams=datastreams,
    )


@lru_cache()
def build_test_dataset(dataset_key, returnn_python_exe, returnn_root, output_path, bpe_size=2000):

    ogg_zip_dict = get_ogg_zip_dict("corpora")
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]
    from i6_core.corpus.convert import CorpusToTextDictJob
    test_reference_dict_file = CorpusToTextDictJob(bliss_dict[dataset_key]).out_dictionary

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=True)

    audio_datastream = get_audio_datastream(returnn_python_exe, returnn_root, output_path)

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}

    test_zip_dataset = OggZipDataset(
        path=[test_ogg],
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse"
    )
    test_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": test_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, test_reference_dict_file


def training(prefix_name, returnn_config, num_epochs, returnn_exe, returnn_root):
    """

    :param prefix_name:
    :param returnn_config:
    :param num_epochs:
    :param Path returnn_exe:
    :param Path returnn_root:
    :return:
    """
    default_rqmt = {
        'mem_rqmt': 15,
        'time_rqmt': 168,
        'log_verbosity': 5,
        'returnn_python_exe': returnn_exe,
        'returnn_root': returnn_root,
    }

    train_job = ReturnnTrainingJob(
        returnn_config=returnn_config,
        num_epochs=num_epochs,
        **default_rqmt
    )
    train_job.add_alias(prefix_name + "/training")
    tk.register_output(prefix_name + "/learning_rates", train_job.out_learning_rates)

    return train_job


def get_best_checkpoint(training_job, output_path):
    """
    :param ReturnnTrainingJob training_job:
    :param str output_path:
    :return:
    """
    from i6_experiments.users.rossenbach.returnn.training import GetBestCheckpointJob
    best_checkpoint_job = GetBestCheckpointJob(
        training_job.out_model_dir,
        training_job.out_learning_rates,
        key="dev_score_output/output_prob",
        index=0)
    best_checkpoint_job.add_alias(os.path.join(output_path, "get_best_checkpoint"))
    return best_checkpoint_job.out_checkpoint


def search_single(
        prefix_name,
        returnn_config,
        checkpoint,
        recognition_dataset,
        recognition_reference,
        returnn_exe,
        returnn_root):
    """
    Run search for a specific test dataset

    :param str prefix_name:
    :param ReturnnConfig returnn_config:
    :param Checkpoint checkpoint:
    :param returnn_standalone.data.datasets.dataset.GenericDataset recognition_dataset:
    :param Path recognition_reference: Path to a py-dict format reference file
    :param Path returnn_exe:
    :param Path returnn_root:
    """
    from i6_core.returnn.search import ReturnnSearchJob, SearchBPEtoWordsJob, ReturnnComputeWERJob
    from i6_experiments.users.rossenbach.returnn.config import get_specific_returnn_config

    search_job = ReturnnSearchJob(
        search_data=recognition_dataset.as_returnn_opts(),
        model_checkpoint=checkpoint,
        returnn_config=get_specific_returnn_config(returnn_config),
        log_verbosity=5,
        mem_rqmt=8,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root
    )
    search_job.add_alias(prefix_name + "/search_job")

    search_words = SearchBPEtoWordsJob(search_job.out_search_file).out_word_search_results
    wer = ReturnnComputeWERJob(search_words, recognition_reference)

    tk.register_output(prefix_name + "/search_out_words.py", search_words)
    tk.register_output(prefix_name + "/wer", wer.out_wer)


def search(prefix_name, returnn_config, checkpoint, test_dataset_tuples, returnn_exe, returnn_root):
    """

    :param str prefix_name:
    :param ReturnnConfig returnn_config:
    :param Checkpoint checkpoint:
    :param test_dataset_tuples:
    :param returnn_exe:
    :param returnn_root:
    :return:
    """
    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        search_single(prefix_name + "/%s" % key, returnn_config, checkpoint, test_dataset, test_dataset_reference, returnn_exe, returnn_root)


