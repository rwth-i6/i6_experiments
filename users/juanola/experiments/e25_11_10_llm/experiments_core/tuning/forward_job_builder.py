"""
Builder for inference/forward jobs (prior & search).

It uses ASRModels classes which contain the information of how to call trained models (calling their checkpoints).
"""

import copy
from typing import Any, Dict, Tuple, List, Optional

from sisyphus import tk, job_path

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.recognition.scoring import ScliteJob
from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.search import SearchOutputRawReplaceJob
from i6_core.returnn.search import SearchWordsToCTMJob
from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob
from i6_experiments.common.setups.returnn.datasets import Dataset
from i6_experiments.users.juanola.data.lm_dataset import LmDataset
from i6_experiments.users.juanola.sisyphus_jobs.lm.ComputeWordLevelPerplexityJob import ComputeWordLevelPerplexityJob
from i6_experiments.users.juanola.sisyphus_jobs.vocab.GetSubwordRatioJob import GetSubwordRatioJob
from ..model_creation.returnn_config_helpers import get_forward_config
from ..tuning.asr_model import ASRModel
from ...configurations.pipeline.search_config import SearchConfig
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
def compute_ppl(
        prefix_name: str,
        search_config: SearchConfig,
        asr_model: ASRModel,
        forward_module: str,
        forward_method: Optional[str],
        test_dataset_tuples: Dict[str, Tuple[Dataset, tk.Path]],
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        spm_model: tk.Path,
        vocab_opts: Dict,
        forward_args: Optional[Dict[str, Any]] = None,
        debug: bool = False,
) -> List[ReturnnForwardJobV2]:
    """
    Run search over multiple datasets and collect statistics

    :param debug:
    :param vocab_opts:
    :param forward_args:
    :param forward_method:
    :param search_config:
    :param prefix_name: prefix folder path for alias and output files
    :param forward_config: returnn config parameter for the forward job
    :param asr_model: the ASRModel from the training
    :param forward_module: path to the file containing the decoder definition
    :param decoder_args: arguments for the decoding forward_init_hook
    :param test_dataset_tuples: tuple of (Dataset, tk.Path) for the dataset object and the reference bliss
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param use_gpu: run search with GPU
    """
    if forward_args is None:
        forward_args = {}

    forward_config = {
        "batch_size": search_config.batch_size,
        "max_seqs": search_config.max_seqs,
    }

    returnn_search_config = get_forward_config(
        network_import_path=asr_model.network_import_path,
        config=forward_config,
        net_args=asr_model.net_args,
        forward_module=forward_module,
        forward_method=forward_method,
        debug=debug,
        vocab_opts=vocab_opts,
        forward_args=forward_args,
    )

    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    search_jobs = []
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        search_name = f"{prefix_name}/{key}"
        search_job = compute_ppl_single(
            search_name,
            returnn_search_config,
            asr_model.checkpoint,
            test_dataset,
            returnn_exe,
            returnn_root,
            spm_model=spm_model,
            use_gpu=search_config.use_gpu,
            search_gpu_memory=search_config.gpu_memory,
        )
        search_jobs.append(search_job)

    return search_jobs


def compute_ppl_single(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        recognition_dataset: LmDataset,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        spm_model: tk.Path,
        mem_rqmt: float = 12,
        use_gpu: bool = False,
        search_gpu_memory: int = 11,
) -> ReturnnForwardJobV2:
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
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: some search jobs might need more memory
    :param use_gpu: if to do GPU decoding
    :param search_gpu_memory:
    """
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["forward_data"] = recognition_dataset.as_returnn_opts()  # TODO: change this!!!

    ppl_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=1,
        device="gpu",
        cpu_rqmt=8,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["val_loss", "sw_ppl"],
    )
    ppl_job.add_alias(f"{prefix_name}/ppl_job")
    tk.register_output(f"{prefix_name}/val_loss", ppl_job.out_files["val_loss"])
    tk.register_output(f"{prefix_name}/sw_ppl", ppl_job.out_files["sw_ppl"])

    ratio_job = GetSubwordRatioJob(
        text_file=recognition_dataset.corpus_file,
        spm_model=spm_model,
    )
    tk.register_output(f"{prefix_name}/sw_ratio", ratio_job.out_ratio)

    wl_ppl_job = ComputeWordLevelPerplexityJob(sw_ppl_file=ppl_job.out_files["sw_ppl"], exponent=ratio_job.out_ratio)
    tk.register_output(f"{prefix_name}/wl_ppl", wl_ppl_job.out_ppl)

    return ppl_job
