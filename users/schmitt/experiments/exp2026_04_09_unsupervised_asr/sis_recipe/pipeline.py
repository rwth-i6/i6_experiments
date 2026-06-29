"""
Pipeline parts to create the necessary jobs for training / forwarding / search etc...
"""

import copy
import enum
from dataclasses import dataclass, asdict
import os.path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable

from sisyphus import tk, Task, Job

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.recognition.scoring import ScliteJob

from i6_core.returnn.search import SearchOutputRawReplaceJob, SearchTakeBestJob
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import (
    ReturnnTrainingJob,
    AverageTorchCheckpointsJob,
    GetBestPtCheckpointJob,
    PtCheckpoint,
)
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.text.processing import PipelineJob

from i6_experiments.common.setups.returnn.datasets import Dataset
from i6_experiments.users.schmitt.datasets.utils import sclite_generic_score
from i6_experiments.users.zeyer.datasets.utils.serialize import (
    ReturnnDatasetToTextDictJob,
)
from i6_experiments.users.zeyer.datasets.task import ScoreResult, RecogOutput

from .config import get_forward_config, get_training_config
from .default_tools import RETURNN_EXE, RETURNN_ROOT


@dataclass
class ASRModel:
    checkpoint: tk.Path
    net_args: Dict[str, Any]
    network_module: str
    prefix_name: Optional[str]


class ApplyHuggingFaceNormalizerToTextDictJob(Job):
    """
    Apply the same normalizer as used for the HuggingFace tokenizer to the reference, so that it matches better with the decoded output
    """

    def __init__(
        self,
        text_dict: tk.Path,
    ):
        super().__init__()

        self.text_dict = text_dict

        self.out = self.output_path("normalized.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import ast
        from ..data.normalize import normalize
        from tqdm import tqdm
        from i6_core.util import uopen

        normalizer = normalize.EnglishTextNormalizer()

        text_dict_path = self.text_dict.get_path()
        with uopen(text_dict_path, "rt", encoding="utf-8") as f:
            text_dict = ast.literal_eval(f.read())

        with open(self.out.get_path(), "w") as f:
            f.write("{\n")
            for seq_tag, text in tqdm(text_dict.items()):
                normalized = normalizer(text)
                f.write(f"{seq_tag!r}: {normalized!r},\n")
            f.write("}\n")


def generic_sclite_score_recog_out(
    dataset: Dict[str, Any],
    recog_output: tk.Path,
    corpus_name: str,
    target_key: str,
    vocab_opts: Dict,
) -> ScoreResult:
    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset,
            data_key=target_key,
            vocab=vocab_opts,
        ).out_txt
    )

    # make the output of the silence label optional
    ref = RecogOutput(output=PipelineJob(ref.output, pipeline=["sed 's/<SIL>/(<SIL>)/g'"]).out)

    # -D flag is required for optional silence
    return sclite_generic_score.sclite_score_recog_out_to_ref(
        RecogOutput(recog_output), ref=ref, corpus_name=corpus_name, sclite_additional_args=["-D"]
    )


def forward_single(
    prefix_name: str,
    returnn_config: ReturnnConfig,
    checkpoint: tk.Path,
    dataset_dict: Dict,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    rqmt: Dict,
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
    returnn_config.config["forward_data"] = dataset_dict
    gpu_mem = rqmt.get("gpu_mem", None)
    search_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=rqmt.get("mem", 20),
        time_rqmt=rqmt.get("time", 1),
        device="gpu",
        cpu_rqmt=rqmt.get("cpu", 8),
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py.gz"],
    )
    if gpu_mem is not None and gpu_mem != 11:
        search_job.rqmt["gpu_mem"] = gpu_mem
    search_job.add_alias(prefix_name + "/search_job")
    tk.register_output(prefix_name + "/search_out.py.gz", search_job.out_files["search_out.py.gz"])

    return search_job.out_files["search_out.py.gz"]


def search_single(
    prefix_name: str,
    returnn_config: ReturnnConfig,
    checkpoint: tk.Path,
    recognition_dataset: Dataset,
    dataset_name: str,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    rqmt: Optional[Dict[str, int]] = None,
    lowercase_ref: bool = False,
    apply_text_norm: bool = False,
    vocab_opts: Optional[Dict] = None,
    recog_post_proc_funcs: Optional[List[Callable[[tk.Path], tk.Path]]] = None,
    score_function: Optional[Callable] = None,
    score_target_key: Optional[str] = None,
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
    if rqmt is None:
        rqmt = {
            "mem_rqmt": 80,
        }

    returnn_config = copy.deepcopy(returnn_config)
    dataset_dict = recognition_dataset.as_returnn_opts()
    returnn_config.config["forward_data"] = dataset_dict
    search_out = forward_single(
        prefix_name=prefix_name,
        returnn_config=returnn_config,
        checkpoint=checkpoint,
        dataset_dict=dataset_dict,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        rqmt=rqmt,
    )

    search_out = SearchTakeBestJob(search_out, output_gzip=True).out_best_search_results

    if recog_post_proc_funcs:
        for func in recog_post_proc_funcs:
            search_out = func(search_out)

    if score_function is not None:
        score_result = score_function(
            dataset=dataset_dict,
            recog_output=search_out,
            corpus_name=dataset_name,
            use_lowercase=lowercase_ref,
            apply_text_norm=apply_text_norm,
            vocab_opts=vocab_opts,
            alias=prefix_name + "/score_job",
        )
        if isinstance(score_result, ScoreResult):
            search_ctm = None
        else:
            score_result, search_ctm = score_result
    else:
        # data key of the reference to measure edit distance against. Defaults to the config's
        # default_target_key (standard ASR -> text); for same-modality reconstruction pass the
        # output modality's data key (e.g. "data" for audio->audio) so the hypotheses are scored
        # against the matching reference.
        if score_target_key is not None:
            target_key = score_target_key
        else:
            target_key = returnn_config.config.get("default_target_key", "text")
        score_result = generic_sclite_score_recog_out(
            dataset=dataset_dict,
            recog_output=search_out,
            corpus_name=dataset_name,
            vocab_opts=vocab_opts,
            target_key=target_key,
        )
        search_ctm = None

    return score_result, search_out, search_ctm


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
    debug: bool = True,
):
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
        decoder_args["prior_file"] = asr_model.prior_file

    returnn_search_config = get_forward_config(
        network_module=asr_model.network_module,
        config=forward_config,
        net_args=asr_model.net_args,
        decoder_args=decoder_args,
        decoder=decoder_module,
        vocab_opts=vocab_opts,
    )

    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    wers = {}
    search_jobs = []
    ctms = {}
    for key, test_dataset in test_dataset_tuples.items():
        search_name = prefix_name + "/%s" % key
        if "hubert_tune" in search_name:
            mem = 30
        elif "RelPosEnc" in search_name:
            mem = 16
        else:
            mem = 12
        wers[search_name], search_job, ctms[key] = search_single(
            prefix_name=search_name,
            returnn_config=returnn_search_config,
            checkpoint=asr_model.checkpoint,
            recognition_dataset=test_dataset,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            use_gpu=True,
            mem_rqmt=mem,
        )
        search_jobs.append(search_job)

    return search_jobs, wers, ctms


def training(
    training_name,
    datasets,
    train_args,
    num_epochs,
    returnn_exe,
    returnn_root,
    time_rqmt: int = 168,
    mem_rqmt: int = 24,
    additional_configs: Optional[List[ReturnnConfig]] = None,
):
    """
    :param training_name:
    :param datasets:
    :param train_args:
    :param num_epochs:
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    """
    default_rqmt = {
        "mem_rqmt": mem_rqmt,
        "time_rqmt": time_rqmt,
        "cpu_rqmt": 6,
        "log_verbosity": 5,
        "returnn_python_exe": returnn_exe,
        "returnn_root": returnn_root,
    }

    default_rqmt.update(train_args.pop("rqmt", {}))
    gpu_mem = default_rqmt.pop("gpu_mem", None)

    num_gpus = train_args["config"].pop("__num_gpus", 1)
    if num_gpus > 1:
        train_args["config"].update(
            {
                "torch_distributed": {"param_sync_step": 100, "reduce_type": "param"},
                "use_horovod": True,
            }
        )
        default_rqmt.update(
            {
                "distributed_launch_cmd": "torchrun",
                "horovod_num_processes": num_gpus,
            }
        )

    returnn_config = get_training_config(training_datasets=datasets, **train_args)
    if additional_configs:
        for add_cfg in additional_configs:
            returnn_config.update(add_cfg)

    train_job = ReturnnTrainingJob(returnn_config=returnn_config, num_epochs=num_epochs, **default_rqmt)
    if gpu_mem is not None and gpu_mem != 11:
        train_job.rqmt["gpu_mem"] = gpu_mem

    train_job.add_alias(training_name + "/training")
    tk.register_output(training_name + "/learning_rates", train_job.out_learning_rates)
    return train_job


def get_checkpoint(
    training_name,
    train_job,
    get_specific_checkpoint: Optional[int] = None,
    get_best_averaged_checkpoint: Optional[Tuple[int, str]] = None,
    get_last_averaged_checkpoint: Optional[int] = None,
    checkpoint: Optional[PtCheckpoint] = None,
):
    """
    :param training_name:
    :param train_job: output of training
    :param train_args: same args as for training
    :param with_prior: If prior should be used (yes for CTC, no for RNN-T)
    :param datasets: Needed if with_prior == True
    :param get_specific_checkpoint: return a specific epoch (set one get_*)
    :param get_best_averaged_checkpoint: return the average with (n checkpoints, loss-key), n checkpoints can be 1
    :param get_last_averaged_checkpoint: return the average of the last n checkpoints
    :param prior_config: if with_prior is true, can be used to add Returnn config parameters for the prior compute job
    :param checkpoint: instead of using train_job, use this specific checkpoint
    :return:
    """

    params = [
        get_specific_checkpoint,
        get_last_averaged_checkpoint,
        get_best_averaged_checkpoint,
        checkpoint,
    ]
    assert sum([p is not None for p in params]) == 1

    if checkpoint is None and train_job is not None:
        if get_best_averaged_checkpoint is not None:
            num_checkpoints, loss_key = get_best_averaged_checkpoint
            checkpoints = []
            for index in range(num_checkpoints):
                best_job = GetBestPtCheckpointJob(
                    train_job.out_model_dir,
                    train_job.out_learning_rates,
                    key=loss_key,
                    index=index,
                )
                best_job.add_alias(training_name + f"/get_best_job_{index}")
                checkpoints.append(best_job.out_checkpoint)
            if num_checkpoints > 1:
                # perform averaging
                avg = AverageTorchCheckpointsJob(
                    checkpoints=checkpoints,
                    returnn_python_exe=RETURNN_EXE,
                    returnn_root=RETURNN_ROOT,
                )
                avg.rqmt["mem"] = 8
                checkpoint = avg.out_checkpoint
            else:
                # we only have one
                checkpoint = checkpoints[0]
        elif get_last_averaged_checkpoint is not None:
            assert get_last_averaged_checkpoint >= 2, (
                "For the single last checkpoint use get_specific_checkpoint instead"
            )
            num_checkpoints = len(train_job.out_checkpoints)
            avg = AverageTorchCheckpointsJob(
                checkpoints=[
                    train_job.out_checkpoints[num_checkpoints - i] for i in range(get_last_averaged_checkpoint)
                ],
                returnn_python_exe=RETURNN_EXE,
                returnn_root=RETURNN_ROOT,
            )
            checkpoint = avg.out_checkpoint
        else:
            checkpoint = train_job.out_checkpoints[get_specific_checkpoint]
    elif train_job is None:
        checkpoint = None

    return checkpoint
