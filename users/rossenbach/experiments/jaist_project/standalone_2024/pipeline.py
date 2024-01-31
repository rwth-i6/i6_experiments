"""
Pipeline parts to create the necessary jobs for training / forwarding / search etc...
"""
import copy

from sisyphus import tk

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.recognition.scoring import ScliteJob

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import SearchWordsToCTMJob
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset

from .default_tools import SCTK_BINARY_PATH


@tk.block()
def training(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        num_epochs: int,
        trigger: bool = False,
) -> ReturnnTrainingJob:
    """
    Perform RETURNN training

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for training
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param num_epochs: for how many epochs to train
    :return: training job
    """
    default_rqmt = {
        'mem_rqmt': 32,
        'time_rqmt': 168,
        'cpu_rqmt': 16,
        'log_verbosity': 5,
        'returnn_python_exe': returnn_exe,
        'returnn_root': returnn_root,
    }

    train_job = ReturnnTrainingJob(
        returnn_config=returnn_config,
        num_epochs=num_epochs,
        **default_rqmt
    )
    if trigger:
        print(train_job)
        assert False
    train_job.add_alias(prefix_name + "/training")
    tk.register_output(prefix_name + "/learning_rates", train_job.out_learning_rates)

    return train_job

@tk.block()
def search_single(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        recognition_dataset: GenericDataset,
        recognition_bliss_corpus: tk.Path,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        mem_rqmt: float = 8,
        use_gpu:bool = False,
):
    """
    Run search for a specific test dataset

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param recognition_dataset: Dataset to perform recognition on
    :param recognition_bliss_corpus: path to bliss file used as Sclite evaluation reference
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: some search jobs might need more memory
    :param use_gpu: if to do GPU decoding
    """
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["forward"] = recognition_dataset.as_returnn_opts()
    search_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=24,
        device="gpu" if use_gpu else "cpu",
        cpu_rqmt=2,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py"],
    )
    search_job.add_alias(prefix_name + "/search_job")

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=search_job.out_files["search_out.py"],
        bliss_corpus=recognition_bliss_corpus,
    ).out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH)
    tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
    tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    return sclite_job.out_wer, search_job


@tk.block()
def search(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        test_dataset_tuples,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        use_gpu: bool = False
):
    """
    Run search over multiple datasets and collect statistics

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param test_dataset_tuples: tuple of (Dataset, tk.Path) for the dataset object and the reference bliss
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :return:
    """
    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    wers = {}
    search_jobs = []
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        wers[key], search_job = search_single(prefix_name + "/%s" % key, returnn_config, checkpoint, test_dataset, test_dataset_reference, returnn_exe, returnn_root, use_gpu=use_gpu)
        search_jobs.append(search_job)

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
    #mail = MailJob(result=report, subject=prefix_name, send_contents=True).out_status
    #tk.register_output(os.path.join(prefix_name, "mail_status"), mail)
    return format_string_report, values_report, search_jobs


@tk.block()
def compute_prior(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        mem_rqmt: int = 8,
):
    """
    Run search for a specific test dataset

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: override the default memory requirement
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


@tk.block()
def extract_durations(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        mem_rqmt: int = 8,
):
    """
    Run phoneme duration extraction

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: override the default memory requirement
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
        output_files=["durations.hdf"],
    )
    search_job.add_alias(prefix_name + "/duration_extraction_job")
    return search_job.out_files["durations.hdf"]


@tk.block()
def tts_eval(
        prefix_name,
        returnn_config,
        checkpoint,
        returnn_exe,
        returnn_root,
        mem_rqmt=8,
):
    """
    Run search for a specific test dataset

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: override the default memory requirement
    """
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=1,
        device="cpu",
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["audio_files"],
    )
    forward_job.add_alias(prefix_name + "/tts_eval_job")
    return forward_job

@tk.block()
def tts_generation(
        prefix_name,
        returnn_config,
        checkpoint,
        returnn_exe,
        returnn_root,
        mem_rqmt=16,
):
    """
    Run search for a specific test dataset

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: override the default memory requirement
    """
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=24,
        device="cpu",
        cpu_rqmt=8,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["audio_files", "out_corpus.xml.gz"],
    )
    forward_job.add_alias(prefix_name + "/tts_eval_job")
    return forward_job

