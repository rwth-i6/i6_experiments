import copy
import os.path

from sisyphus import tk

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.training import GetBestTFCheckpointJob
from i6_core.returnn.forward import ReturnnForwardJob, ReturnnForwardJobV2
from i6_core.returnn.search import SearchBPEtoWordsJob, ReturnnComputeWERJob
from i6_experiments.users.rossenbach.returnn.training import AverageCheckpointsJobV2

from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT, SCTK_BINARY_PATH


@tk.block()
def training(prefix_name, returnn_config, returnn_exe, returnn_root, num_epochs=250, large_gpu=False):
    """

    :param prefix_name:
    :param returnn_config:
    :param returnn_exe:
    :param returnn_root:
    :return:
    """
    default_rqmt = {
        'mem_rqmt': 10,
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

    if (large_gpu):
        train_job.rqmt["gpu_mem"] = 24
    
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
        time_rqmt=4,
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
    clean_prefix_name = prefix_name.replace(".", "_")
    format_string_report = ",".join(["{%s_val}" % (clean_prefix_name + key) for key in test_dataset_tuples.keys()])
    format_string = " - ".join(["{%s}: {%s_val}" % (clean_prefix_name + key, clean_prefix_name + key) for key in test_dataset_tuples.keys()])
    values = {}
    values_report = {}
    for key in test_dataset_tuples.keys():
        values[clean_prefix_name + key] = key
        values["%s_val" % (clean_prefix_name + key)] = wers[key]
        values_report["%s_val" % (clean_prefix_name + key)] = wers[key]

    report = GenerateReportStringJob(report_values=values, report_template=format_string, compress=False).out_report
    # mail = MailJob(result=report, subject=prefix_name, send_contents=True).out_status
    tk.register_output(os.path.join(prefix_name, "report"), report)
    return format_string_report, values_report

def compute_phoneme_pred_accuracy(
        prefix_name,
        returnn_config,
        checkpoint,
        recognition_datasets,
        returnn_exe,
        returnn_root,
        mem_rqmt=8,
        ):
    """Replaces the search job for the "encoding_test" experiments, where a simple model is asked 
    to predict the phonemes from the latent variables of a glowTTS setup. These experiments output an hdf with 
    the total accuracy on each batch and there is no need to perform a search on these.

    :param _type_ prefix_name: _description_
    :param _type_ returnn_config: _description_
    :param _type_ checkpoint: _description_
    :param GenericDataset recognition_dataset: _description_
    :param _type_ recognition_bliss_corpus: _description_
    :param _type_ returnn_exe: _description_
    :param _type_ returnn_root: _description_
    :param int mem_rqmt: _description_, defaults to 8
    """   
    for key, (recognition_dataset, test_dataset_reference) in recognition_datasets.items():

        returnn_config = copy.deepcopy(returnn_config)
        returnn_config.config["forward"] = recognition_dataset.as_returnn_opts()
        search_job = ReturnnForwardJob(
            model_checkpoint=checkpoint,
            returnn_config=returnn_config,
            log_verbosity=5,
            mem_rqmt=mem_rqmt,
            time_rqmt=4,
            returnn_python_exe=returnn_exe,
            returnn_root=returnn_root,
            device="cpu",
        )
        search_job.add_alias(prefix_name + "/forward")
        tk.register_output(prefix_name + "/forward", search_job.out_hdf_files["output.hdf"])
        return search_job

@tk.block()
def compute_prior(
        prefix_name,
        returnn_config,
        checkpoint,
        returnn_exe,
        returnn_root,
        mem_rqmt=8,
):
    """
    Run search for a specific test dataset

    :param str prefix_name:
    :param ReturnnConfig returnn_config:
    :param Checkpoint checkpoint:
    :param Path returnn_exe:
    :param Path returnn_root:
    """
    search_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=1,
        device="gpu",
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["prior.txt"],
    )
    search_job.add_alias(prefix_name + "/prior_job")
    return search_job.out_files["prior.txt"]
