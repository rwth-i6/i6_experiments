import os.path

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob


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
        #returnn_config=get_specific_returnn_config(returnn_config),
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


