"""
Builder for inference/forward jobs (prior & search).

It uses ASRModels classes which contain the information of how to call trained models (calling their checkpoints).
"""

import copy
from typing import Any, Dict, Tuple, List

from sisyphus import tk, job_path

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.recognition.scoring import ScliteJob
from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.search import SearchOutputRawReplaceJob
from i6_core.returnn.search import SearchWordsToCTMJob
from i6_experiments.common.setups.returnn.datasets import Dataset
from ..model_creation.returnn_config_helpers import get_forward_config
from ..tuning.asr_model import ASRModel
from ...default_tools import SCTK_BINARY_PATH


@tk.block()
def compute_prior(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: PtCheckpoint,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        mem_rqmt: int = 16,
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
        time_rqmt=2,
        device="gpu",
        cpu_rqmt=8,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["prior.txt"],
    )
    if "hubert_tune_v2" in prefix_name:
        search_job.rqmt["time"] += 12
        search_job.rqmt["gpu_mem"] = 24
    search_job.add_alias(f"{prefix_name}/prior_job")
    return search_job.out_files["prior.txt"]


@tk.block()
def search(
        prefix_name: str,
        forward_config: Dict[str, Any],
        asr_model: ASRModel,
        decoder_module: str,
        decoder_args: Dict[str, Any],
        test_dataset_tuples: Dict[str, Tuple[Dataset, tk.Path]],
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        vocab_opts: Dict,
        use_gpu: bool = False,
        debug: bool = False,
) -> Tuple[List[ReturnnForwardJobV2], Dict[str, job_path.Variable]]:
    """
    Run search over multiple datasets and collect statistics

    :param prefix_name: prefix folder path for alias and output files
    :param forward_config: returnn config parameter for the forward job
    :param asr_model: the ASRModel from the training
    :param decoder_module: path to the file containing the decoder definition
    :param decoder_args: arguments for the decoding forward_init_hook
    :param test_dataset_tuples: tuple of (Dataset, tk.Path) for the dataset object and the reference bliss
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param use_gpu: run search with GPU
    """
    if asr_model.prior_file is not None:
        decoder_args["config"]["prior_file"] = asr_model.prior_file

    returnn_search_config = get_forward_config(
        network_module=asr_model.network_module,
        config=forward_config,
        net_args=asr_model.net_args,
        decoder_args=decoder_args,
        decoder=decoder_module,
        debug=debug,
        vocab_opts=vocab_opts,
    )

    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    wers = {}
    search_jobs = []
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        search_name = prefix_name + "/%s" % key
        if "hubert_tune" in search_name:
            mem = 30
        elif "RelPosEnc" in search_name:
            mem = 16
        else:
            mem = 12
        wers[search_name], search_job = search_single(
            search_name,
            returnn_search_config,
            asr_model.checkpoint,
            test_dataset,
            test_dataset_reference,
            returnn_exe,
            returnn_root,
            use_gpu=use_gpu,
            mem_rqmt=mem,
        )
        search_jobs.append(search_job)

    return search_jobs, wers


def search_single(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        recognition_dataset: Dataset,
        recognition_bliss_corpus: tk.Path,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        mem_rqmt: float = 14,
        use_gpu: bool = False,
) -> Tuple[job_path.Variable, ReturnnForwardJobV2]:
    """
    Run search for a specific test dataset.

    Runs:
    - ReturnnForwardJobV2
    - SearchOutputRawReplaceJob
    - SearchWordsToCTMJob
    - CorpusToStmJob
    - ScliteJob

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
    returnn_config.config["forward_data"] = recognition_dataset.as_returnn_opts()
    search_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=24,
        device="gpu" if use_gpu else "cpu",
        cpu_rqmt=8 if mem_rqmt < 30 else 16,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py.gz"],
    )
    search_job.add_alias(prefix_name + "/search_job")

    words = SearchOutputRawReplaceJob(
        search_job.out_files["search_out.py.gz"], [(" ", ""), ("▁", " ")], output_gzip=True
    ).out_search_results

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=words,
        bliss_corpus=recognition_bliss_corpus,
    ).out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
    tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
    tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    return sclite_job.out_wer, search_job
