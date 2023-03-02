import os.path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import get_default_asr_audio_datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import ReturnnAudioRawOptions, AudioRawDatastream
from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn.training import ReturnnTrainingJob

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_corpus_dict

from i6_experiments.common.datasets.librispeech import get_subword_nmt_bpe
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
    bpe_settings = get_subword_nmt_bpe(corpus_key="train-clean-100", bpe_size=bpe_size, unk_label='<unk>')
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


def build_training_datasets(
        returnn_python_exe, returnn_root, output_path,
        ls_corpus_key="train-clean-100",
        bpe_size=2000,
        partition_epoch=3,
        use_curicculum=True,
        seq_ordering="laplace:.1000",
        link_speed_perturbation=False,
        use_raw_features=False,
        synthetic_ogg_zip=None,
        original_scale=1,
        synthetic_scale=1,
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
    train_clean_100_ogg = ogg_zip_dict[ls_corpus_key]
    dev_clean_ogg = ogg_zip_dict['dev-clean']
    dev_other_ogg = ogg_zip_dict['dev-other']

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=False)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream()
    else:
        audio_datastream = get_audio_datastream(
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            output_path=output_path,
        )

    extern_data = {
        'audio_features': audio_datastream.as_returnn_extern_data_opts(),
        'bpe_labels': train_bpe_datastream.as_returnn_extern_data_opts()
    }

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}


    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    if link_speed_perturbation:
        training_audio_opts["pre_process"] = CodeWrapper("speed_pert")

    training_ogg_combination = train_clean_100_ogg
    if synthetic_ogg_zip:
        training_ogg_combination = [train_clean_100_ogg] * original_scale + [synthetic_ogg_zip] * synthetic_scale

    train_zip_dataset = OggZipDataset(
        files=training_ogg_combination,
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
        files=train_clean_100_ogg,
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
def build_test_dataset(dataset_key, returnn_python_exe, returnn_root, output_path, bpe_size=2000,
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
        audio_datastream = get_audio_datastream(
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            output_path=output_path,
        )

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


@lru_cache()
def build_profile_dataset(dataset_key, returnn_python_exe, returnn_root, output_path, bpe_size=2000):

    ogg_zip_dict = get_ogg_zip_dict("corpora")
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]
    from i6_core.corpus.convert import CorpusToTextDictJob
    from i6_core.corpus.segments import SegmentCorpusJob
    segments = SegmentCorpusJob(bliss_corpus=bliss_dict, num_segments=1).out_single_segment_files[0]

    test_reference_dict_file = CorpusToTextDictJob(bliss_dict[dataset_key]).out_dictionary

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=True)

    audio_datastream = get_audio_datastream(returnn_python_exe, returnn_root, output_path)

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


def training(prefix_name, returnn_config, returnn_exe, returnn_root, num_epochs=250):
    """

    :param prefix_name:
    :param returnn_config:
    :param returnn_exe:
    :param returnn_root:
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


def get_average_checkpoint(training_job, returnn_exe, returnn_root, num_average:int = 4):
    """
    get an averaged checkpoint using n models

    :param training_job:
    :param num_average:
    :return:
    """
    from i6_experiments.users.rossenbach.returnn.training import GetBestCheckpointJob, AverageCheckpointsJobV2
    epochs = []
    for i in range(num_average):
        best_checkpoint_job = GetBestCheckpointJob(
            training_job.out_model_dir,
            training_job.out_learning_rates,
            key="dev_score_output/output_prob",
            index=i)
        epochs.append(best_checkpoint_job.out_epoch)
    average_checkpoint_job = AverageCheckpointsJobV2(training_job.out_model_dir, epochs=epochs, returnn_python_exe=returnn_exe, returnn_root=returnn_root)
    return average_checkpoint_job.out_checkpoint


def get_average_checkpoint_v2(training_job, returnn_exe, returnn_root, num_average:int = 4):
    """
    get an averaged checkpoint using n models

    :param training_job:
    :param num_average:
    :return:
    """
    from i6_experiments.users.rossenbach.returnn.training import AverageCheckpointsJobV2
    from i6_core.returnn.training import GetBestTFCheckpointJob
    epochs = []
    for i in range(num_average):
        best_checkpoint_job = GetBestTFCheckpointJob(
            training_job.out_model_dir,
            training_job.out_learning_rates,
            key="dev_score_output/output_prob",
            index=i)
        epochs.append(best_checkpoint_job.out_epoch)
    average_checkpoint_job = AverageCheckpointsJobV2(training_job.out_model_dir, epochs=epochs, returnn_python_exe=returnn_exe, returnn_root=returnn_root)
    return average_checkpoint_job.out_checkpoint


def search_single(
        prefix_name,
        returnn_config,
        checkpoint,
        recognition_dataset,
        recognition_reference,
        returnn_exe,
        returnn_root,
        mem_rqmt=8,
):
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
    from i6_core.returnn.search import ReturnnSearchJobV2, SearchBPEtoWordsJob, ReturnnComputeWERJob


    search_job = ReturnnSearchJobV2(
        search_data=recognition_dataset.as_returnn_opts(),
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root
    )
    search_job.add_alias(prefix_name + "/search_job")

    search_words = SearchBPEtoWordsJob(search_job.out_search_file).out_word_search_results
    wer = ReturnnComputeWERJob(search_words, recognition_reference)

    tk.register_output(prefix_name + "/search_out_words.py", search_words)
    tk.register_output(prefix_name + "/wer", wer.out_wer)
    return wer.out_wer


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
    wers = {}
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        wers[key] = search_single(prefix_name + "/%s" % key, returnn_config, checkpoint, test_dataset, test_dataset_reference, returnn_exe, returnn_root)

    from i6_core.report import GenerateReportStringJob, MailJob
    format_string_report = ",".join(["{%s_val}" % (prefix_name + key) for key in test_dataset_tuples.keys()])
    format_string = " - ".join(["{%s}: {%s_val}" % (prefix_name + key, prefix_name + key) for key in test_dataset_tuples.keys()])
    values = {}
    values_report = {}
    for key in test_dataset_tuples.keys():
        values[prefix_name + key] = key
        values["%s_val" % (prefix_name + key)] = wers[key]
        values_report["%s_val" % (prefix_name + key)] = wers[key]

    report = GenerateReportStringJob(report_values=values, report_template=format_string, compress=False).out_report
    mail = MailJob(result=report, subject=prefix_name, send_contents=True).out_status
    tk.register_output(os.path.join(prefix_name, "mail_status"), mail)
    return format_string_report, values_report






