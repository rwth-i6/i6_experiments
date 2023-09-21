import copy
import os.path

from sisyphus import tk

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.training import GetBestTFCheckpointJob
from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.returnn.search import SearchBPEtoWordsJob, ReturnnComputeWERJob
from i6_experiments.users.rossenbach.returnn.training import AverageCheckpointsJobV2

from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT, SCTK_BINARY_PATH


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

@tk.block()
def search_single(
        prefix_name,
        returnn_config,
        checkpoint,
        recognition_dataset: GenericDataset,
        recognition_bliss_corpus,
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
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["forward"] = recognition_dataset.as_returnn_opts()
    search_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=2,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        hdf_outputs=["search_out.py"],
        device="cpu"
    )
    search_job.add_alias(prefix_name + "/search_job")

    search_words = SearchBPEtoWordsJob(search_job.out_hdf_files["search_out.py"]).out_word_search_results

    from i6_core.returnn.search import SearchWordsToCTMJob
    from i6_core.corpus.convert import CorpusToStmJob
    from i6_core.recognition.scoring import ScliteJob

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=search_words,
        bliss_corpus=recognition_bliss_corpus,
    ).out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH)
    tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
    tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    return sclite_job.out_wer


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

