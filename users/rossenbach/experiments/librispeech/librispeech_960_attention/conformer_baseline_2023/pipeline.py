import os.path

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.training import GetBestTFCheckpointJob
from i6_core.returnn.search import ReturnnSearchJobV2, SearchBPEtoWordsJob, ReturnnComputeWERJob
from i6_experiments.users.rossenbach.returnn.training import AverageCheckpointsJobV2

from .default_tools import RETURNN_CPU_EXE, RETURNN_ROOT


@tk.block()
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


def get_best_checkpoint(training_job):
    """
    :param ReturnnTrainingJob training_job:
    :return:
    """
    best_checkpoint_job = GetBestTFCheckpointJob(
        training_job.out_model_dir,
        training_job.out_learning_rates,
        key="dev_score_output/output_prob",
        index=0)
    return best_checkpoint_job.out_checkpoint


@tk.block()
def get_average_checkpoint(training_job, num_average:int = 4):
    """
    get an averaged checkpoint using n models

    :param training_job:
    :param num_average:
    :return:
    """
    from i6_core.returnn.training import AverageTFCheckpointsJob
    epochs = []
    for i in range(num_average):
        best_checkpoint_job = GetBestTFCheckpointJob(
            training_job.out_model_dir,
            training_job.out_learning_rates,
            key="dev_score_output/output_prob",
            index=i)
        epochs.append(best_checkpoint_job.out_epoch)
    average_checkpoint_job = AverageTFCheckpointsJob(training_job.out_model_dir, epochs=epochs, returnn_python_exe=RETURNN_CPU_EXE, returnn_root=RETURNN_ROOT)
    return average_checkpoint_job.out_checkpoint

@tk.block()
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
    search_job = ReturnnSearchJobV2(
        search_data=recognition_dataset.as_returnn_opts(),
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=2,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root
    )
    search_job.add_alias(prefix_name + "/search_job")

    search_words = SearchBPEtoWordsJob(search_job.out_search_file).out_word_search_results
    wer = ReturnnComputeWERJob(search_words, recognition_reference)

    tk.register_output(prefix_name + "/search_out_words.py", search_words)
    tk.register_output(prefix_name + "/wer", wer.out_wer)
    return wer.out_wer


@tk.block()
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


