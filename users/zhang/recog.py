"""
Generic recog, for the model interfaces defined in model_interfaces.py
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Sequence, Collection, Iterator, Callable, List, Tuple
import functools

import sisyphus
#from i6_experiments.users.berger.systems.functors import VocabType
from sisyphus import tk
from sisyphus import tools as sis_tools
from i6_core.util import instanciate_delayed, uopen

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob, PtCheckpoint, AverageTorchCheckpointsJob
from i6_core.returnn.search import (
    ReturnnSearchJobV2,
    SearchRemoveLabelJob,
    SearchCollapseRepeatedLabelsJob,
    SearchTakeBestJob,
)
from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common import nn
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.common.setups import serialization
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zhang.datasets.task import Task
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput, ScoreResultCollection
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, serialize_model_def
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint, ModelWithCheckpoints
from i6_experiments.users.zeyer.returnn.training import get_relevant_epochs_from_training_learning_rate_scores

import numpy as np

from .experiments.ctc import model_recog, model_recog_lm, model_recog_flashlight, recog_nn
from .datasets.librispeech import get_bpe_lexicon, LibrispeechOggZip

if TYPE_CHECKING:
    from returnn.tensor import TensorDict

USE_24GB = False # Making all forward job to use 24gb gpu

def recog_exp(
    prefix_name: str,
    task: Task,
    model: ModelWithCheckpoints,
    recog_def: RecogDef,
    *,
    first_pass_name: str,
    epoch: int, # TODO: should run a GetBestRecogTrainExp beforehand to get the best epoch
    decoding_config: Optional[dict] = None,
    search_config: Dict[str, Any] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    search_mem_rqmt: Union[int, float] = 6,
    exclude_epochs: Collection[int] = (),
    empirical_prior: Optional[tk.Path] = None,
    prior_from_max: bool = False,
    dev_sets: Optional[List[str]] = None,
    search_error_check: bool = False,
    search_rqmt: dict = None,
)-> Tuple[tk.Path, tk.Path, tk.Path]:
    """recog on given epoch"""
    recog_and_score_func = _RecogAndScoreFunc(
        prefix_name,
        decoding_config,
        task,
        model,
        recog_def,
        first_pass_name=first_pass_name,
        search_config=search_config,
        search_post_config=search_post_config,
        recog_post_proc_funcs=recog_post_proc_funcs,
        search_mem_rqmt=search_mem_rqmt,
        empirical_prior=empirical_prior,
        prior_from_max=prior_from_max,
        dev_sets=dev_sets,
        search_rqmt=search_rqmt,
        search_error_check=search_error_check,
        search_error_version=2, # 2: emission mask, 3:lm masking
    )
    # In following jobs, model is implicitly called in recog_and_score_func, here the passed reference only provides epoch information.
    # So, make sure the exp here align with the model used to initialise recog_and_score_func
    summarize_job = GetRecogExp(
        model=model,
        epoch=epoch,
        recog_and_score_func=recog_and_score_func,
    )
    #summarize_job.add_alias(prefix_name + "/train-summarize")
    #tk.register_output(prefix_name + "/recog_results_best", summarize_job.out_summary_json)
    if search_error_check:
        tk.register_output(prefix_name + "/search_error", summarize_job.out_search_error)
        return summarize_job.out_summary_json, summarize_job.out_search_error, summarize_job.out_search_error_rescore
    return summarize_job.out_summary_json, None, summarize_job.out_search_error_rescore


def recog_training_exp(
    prefix_name: str,
    task: Task,
    model: ModelWithCheckpoints,
    recog_def: RecogDef,
    *,
    decoding_config: Optional[dict] = None,
    search_config: Dict[str, Any] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    search_mem_rqmt: Union[int, float] = 6,
    exclude_epochs: Collection[int] = (),
    model_avg: bool = False,
    empirical_prior: Optional[tk.Path] = None,
    prior_from_max: bool = False,
    dev_sets: Optional[List[str]] = None,
    search_rqmt: dict = None,
)-> tk.path:
    """recog on all relevant epochs"""
    recog_and_score_func = _RecogAndScoreFunc(
        prefix_name,
        decoding_config,
        task,
        model,
        recog_def,
        search_config=search_config,
        search_post_config=search_post_config,
        recog_post_proc_funcs=recog_post_proc_funcs,
        search_mem_rqmt=search_mem_rqmt,
        empirical_prior=empirical_prior,
        prior_from_max=prior_from_max,
        dev_sets=dev_sets,
        search_rqmt=search_rqmt,
    )
    # In following jobs, model is implicitly called in recog_and_score_func, here the passed reference only provides epoch information.
    # So, make sure the exp here align with the model used to initialise recog_and_score_func
    summarize_job = GetBestRecogTrainExp(
        exp=model,
        recog_and_score_func=recog_and_score_func,
        main_measure_lower_is_better=task.main_measure_type.lower_is_better,
        exclude_epochs=exclude_epochs,
    )
    summarize_job.add_alias(prefix_name + "/train-summarize")
    tk.register_output(prefix_name + "/recog_results_best", summarize_job.out_summary_json)
    #tk.register_output(prefix_name + "/recog_results_all_epochs", summarize_job.out_results_all_epochs_json)
    if model_avg:
        model_avg_res_job = GetTorchAvgModelResult(
            exp=model, recog_and_score_func=recog_and_score_func, exclude_epochs=exclude_epochs
        )
        tk.register_output(prefix_name + "/recog_results_model_avg", model_avg_res_job.out_results)
    return summarize_job.out_summary_json

class _RecogAndScoreFunc:
    def __init__(
        self,
        prefix_name: str,
        decoding_config: dict,
        task: Task,
        model: ModelWithCheckpoints,
        recog_def: RecogDef,
        *,
        first_pass_name:str = "",
        search_config: Optional[Dict[str, Any]] = None,
        search_post_config: Optional[Dict[str, Any]] = None,
        recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
        search_mem_rqmt: Union[int, float] = 6,
        empirical_prior: Optional[tk.Path] = None,
        prior_from_max: bool = False,
        dev_sets: Optional[List[str]] = None,
        search_rqmt: dict = None,
        search_error_check: bool = False,
        search_error_version: int = 2,
    ):
        # Note: When something is added here, remember to handle it in _sis_hash.
        self.prefix_name = prefix_name
        self.first_pass_name = first_pass_name
        self.decoding_config = decoding_config
        self.task = task
        self.model = model
        self.recog_def = recog_def
        self.search_config = search_config
        self.search_post_config = search_post_config
        self.recog_post_proc_funcs = recog_post_proc_funcs
        self.search_mem_rqmt = search_mem_rqmt
        self.empirical_prior = empirical_prior
        self.prior_from_max = prior_from_max
        self.dev_sets = dev_sets
        self.search_rqmt = search_rqmt
        self.search_error_check = search_error_check
        self.search_error_version = search_error_version

    def __call__(self, epoch_or_ckpt: Union[int, PtCheckpoint]) -> Tuple[ScoreResultCollection, Optional[tk.Path], Optional[tk.Path]]:
        if isinstance(epoch_or_ckpt, int):
            model_with_checkpoint = self.model.get_epoch(epoch_or_ckpt)
        elif isinstance(epoch_or_ckpt, PtCheckpoint):
            model_with_checkpoint = ModelWithCheckpoint(definition=self.model.definition, checkpoint=epoch_or_ckpt)
        else:
            raise TypeError(f"{self} unexpected type {type(epoch_or_ckpt)}")
        # Calculate prior of labels if needed
        if self.task.prior_dataset:
            if self.empirical_prior:
                prior_path = self.empirical_prior
            else:
                prior_path = compute_prior(
                    dataset=self.task.prior_dataset,
                    model=model_with_checkpoint,
                    prior_alias_name=self.prefix_name + f"/prior/{epoch_or_ckpt:03}",
                    #num_shards=self.num_shards_prior,
                    prior_from_max=self.prior_from_max,
                )
                # if isinstance(epoch_or_ckpt, int):
                #     tk.register_output(self.prefix_name + f"/recog_results_per_epoch/{epoch_or_ckpt:03}/prior.txt", prior_path)
        else:
            prior_path = None
        res, search_error, search_error_rescore = recog_model(
            self.task,
            model_with_checkpoint,
            self.recog_def,
            self.decoding_config,
            prior_path,
            config=self.search_config,
            search_post_config=self.search_post_config,
            recog_post_proc_funcs=self.recog_post_proc_funcs,
            search_mem_rqmt=self.search_mem_rqmt,
            dev_sets=self.dev_sets,
            search_rqmt=self.search_rqmt,
            name=self.prefix_name + f"/epoch{epoch_or_ckpt:03}",
            first_pass_name=self.first_pass_name + f"/epoch{epoch_or_ckpt:03}",
            search_error_check=self.search_error_check,
            search_error_version=self.search_error_version,
        )
        #if isinstance(epoch_or_ckpt, int):
            #tk.register_output(self.prefix_name + f"/recog_results_per_epoch/{epoch_or_ckpt:03}/res", res.output)
        return res, search_error, search_error_rescore

    def _sis_hash(self) -> bytes:
        from sisyphus.hash import sis_hash_helper

        d = self.__dict__.copy()
        # Remove irrelevant stuff which should not affect the hash.
        del d["prefix_name"]
        del d["first_pass_name"]
        del d["search_post_config"]
        del d["search_mem_rqmt"]
        del d["search_rqmt"]
        if not self.search_config:
            del d["search_config"]  # compat
        if not self.recog_post_proc_funcs:
            del d["recog_post_proc_funcs"]  # compat
        # Not the whole task object is relevant but only some minimal parts.
        task = d.pop("task")
        assert isinstance(task, Task)
        # TODO: This is actually not really needed here because the trained model should already cover
        #  all training relevant aspects.
        #  But we cannot remove this easily now to not break old hashes...
        for k in ["train_dataset", "train_epoch_split"]:  # for hash relevant parts
            d[f"task.{k}"] = getattr(task, k)
        if getattr(task, "prior_dataset", None):
            d["task.prior_dataset"] = getattr(task, "prior_dataset")
        if not self.empirical_prior:
            del d["empirical_prior"]
        if not self.prior_from_max:
            del d["prior_from_max"]
        d["class"] = "_RecogAndScoreFunc"  # some identifier; not full qualname to allow for moving the class
        return sis_hash_helper(d)


def recog_model(
    task: Task,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    decoding_config: dict,
    prior_path: tk.Path,
    *,
    first_pass_name: str = "",
    config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    recog_pre_post_proc_funcs_ext: Sequence[Callable] = (),
    search_mem_rqmt: Union[int, float] = 6,
    search_rqmt: Optional[Dict[str, Any]] = None, #= {"time": 6},
    dev_sets: Optional[Collection[str]] = None, #["dev-other", "test-other"]
    name: Optional[str] = None,
    search_error_check: bool = False,
    search_error_version: int = 2,
) -> Tuple[ScoreResultCollection, Optional[tk.Path], Optional[tk.Path]]:
    """
    Recog for some given model (a single given checkpoint / epoch).
    (Used by :func:`recog_training_exp` (:class:`_RecogAndScoreFunc`).)

    :param task:
    :param model:
    :param recog_def:
    :param config:
    :param search_post_config:
    :param recog_post_proc_funcs: Those are run before ``task.recog_post_proc_funcs``
        (which usually does BPE to words or so).
        Those are also run before we take the best hyp from a beam of hyps,
        i.e. they run potentially on a set of hyps (if the recog returned a beam).
        Those are run after blank label removal and label repetition collapsing in case ``output_blank_label`` is set.
    :param recog_pre_post_proc_funcs_ext:
        Funcs (recog_out, *, raw_res_search_labels, raw_res_labels, search_labels_to_labels, **other).
        Those are run before recog_post_proc_funcs,
        and they additionally get also
        ``raw_res_search_labels`` (e.g. align labels, e.g. BPE including blank)
        and ``raw_res_labels`` (e.g. BPE labels).
    :param search_mem_rqmt: for the search job. 6GB by default. can also be set via ``search_rqmt``
    :param search_rqmt: e.g. {"gpu": 1, "mem": 6, "cpu": 4, "gpu_mem": 24} or so
    :param dev_sets: which datasets to evaluate on. None means all defined by ``task``
    :param name: defines ``search_alias_name`` for the search job
    :return: scores over all datasets (defined by ``task``) using the main measure (defined by the ``task``),
        specifically what comes out of :func:`search_dataset`,
        then scored via ``task.score_recog_output_func``,
        then collected via ``task.collect_score_results_func``.
    """
    if dev_sets is not None:
        assert all(k in task.eval_datasets for k in dev_sets)
        # task.main_measure_name = dev_sets[0]
    outputs = {}
    search_error = None

    decoding_params = decoding_config.copy()
    rescoring = decoding_params.pop("rescoring", False)

    if rescoring:
        from .experiments.decoding.lm_rescoring import lm_am_framewise_prior_rescore, ngram_rescore_def, ffnn_rescore_def, trafo_lm_rescore_def
        from .experiments.ctc import ctc_model_rescore
        from .experiments.decoding.prior_rescoring import Prior
        from i6_experiments.users.zeyer.datasets.utils.vocab import (
            ExtractVocabLabelsJob,
            ExtractVocabSpecialLabelsJob,
            ExtendVocabLabelsByNewLabelJob,
        )
        from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
        rescoringLM = decoding_params.pop("lm_rescore", None)
        rescoringLM_scale = decoding_params.pop("rescore_lmscale", None)
        rescoreLM_name = decoding_params.pop("rescore_lm_name", "")
        rescore_Priorscale = decoding_params.pop("rescore_priorscale", None)
        vocab_: Bpe
        vocab_ = decoding_params.pop("vocab", None)
        assert vocab_, "Need vocab for current rescoring implementation"
        assert isinstance(vocab_, Bpe), "For now only support Bpe"
        vocab_file = ExtractVocabLabelsJob(vocab_.get_opts()).out_vocab
        #tk.register_output(f"{name}/vocab.txt.gz", vocab_file)
        vocab_opts_file = ExtractVocabSpecialLabelsJob(vocab_.get_opts()).out_vocab_special_labels_dict
        #tk.register_output(f"{name}/vocab_opts.py", vocab_opts_file)
        vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
            vocab=vocab_file, new_label=recog_def.output_blank_label, new_label_idx=-1
        ).out_vocab
        #tk.register_output(f"{name}/vocab_w_blank.txt.gz", vocab_w_blank_file)

        def dummy_rescor():
            pass

        rescor_def = dummy_rescor

        if rescoreLM_name.startswith("ffnn"):
            rescor_def = ffnn_rescore_def
            lm_rescor_rqmt = {"cpu": 2, "mem": 8, "time": 2, "gpu_mem": 10}
        elif rescoreLM_name.startswith("trafo"):
            rescor_def = trafo_lm_rescore_def
            lm_rescor_rqmt = {"cpu": 2, "mem": 8, "time": 2, "gpu_mem": 24}
        elif rescoreLM_name[0].isdigit() and "gram" in rescoreLM_name:
            rescor_def = ngram_rescore_def
            lm_rescor_rqmt = {"cpu": 2, "mem": 8, "time": 2}
        elif "NoLM" in rescoreLM_name:
            lm_rescor_rqmt = {"cpu": 2, "mem": 4, "time": 1}
        else:
            prev_one_ctx = False
            prompt = None
            if isinstance(rescoringLM, dict):
                prev_one_ctx = rescoringLM.get("prev_one_ctx", False)
                prompt = rescoringLM.get("prompt", None)
            time_factor = 2 if prev_one_ctx or prompt else 1
            lm_rescor_rqmt = {"Llama-3.2-1B":{"cpu": 2, "mem": 30, "time": 3*time_factor, "gpu_mem": 24},
                              "Llama-3.1-8B":{"cpu": 2, "mem": 40, "time": 6*time_factor, "gpu_mem": 48},
                              "Qwen3-0.6B-Base":{"cpu": 2, "mem": 25, "time": 2*time_factor, "gpu_mem": 24},
                              "Qwen3-1.7B-Base":{"cpu": 2, "mem": 33, "time": 4*time_factor, "gpu_mem": 24},
                              "Qwen3-4B-Base":{"cpu": 2, "mem": 35, "time": 12*time_factor, "gpu_mem": 24},
                              "Qwen3-8B-Base":{"cpu": 2, "mem": 40, "time": 6*time_factor, "gpu_mem": 48},
                              "Mistral-7B-v0.3":{"cpu": 2, "mem": 40, "time": 4*time_factor, "gpu_mem": 24},}.get(rescoreLM_name)
            assert lm_rescor_rqmt is not None, f"LM type{rescoreLM_name} not found"
            #print(f"Warning: Check LM type{rescoreLM_name}, will use HF_LM rescoring")


        recog_pre_post_proc_funcs_ext = list(recog_pre_post_proc_funcs_ext)
        recog_pre_post_proc_funcs_ext += [
                functools.partial(
                    lm_am_framewise_prior_rescore,
                    # framewise standard prior
                    prior=Prior(file=prior_path, type="log_prob", vocab=vocab_w_blank_file),
                    prior_scale=rescore_Priorscale,
                    am=model,
                    am_rescore_def=ctc_model_rescore,
                    am_rescore_rqmt={"cpu": 2, "mem": 8, "time": 4, "gpu_mem": 10},
                    am_scale=1.0,
                    lm=rescoringLM,
                    lm_scale=rescoringLM_scale,
                    lm_rescore_def=rescor_def,
                    lm_rescore_rqmt=lm_rescor_rqmt,
                    vocab=vocab_file,
                    vocab_opts_file=vocab_opts_file,
                )
        ]

    search_error_rescore_dict = {}

    for dataset_name, dataset in task.eval_datasets.items():
        if dev_sets is not None:
            if dataset_name not in dev_sets:
                continue
        # print(f"dataset:{dataset}, type{type(dataset)}, name:{dataset_name}, type:{type(dataset_name)}")
        recog_out, hyps, search_error_rescore = search_dataset( # Hyps here is raw out from first pass
            decoding_config=decoding_params,
            dataset=dataset,
            model=model,
            recog_def=recog_def,
            prior_path=prior_path,
            config=config,
            search_post_config=search_post_config,
            search_mem_rqmt=search_mem_rqmt,
            search_rqmt=search_rqmt,
            search_alias_name=f"{name}/search/{dataset_name}" if name else None,
            first_pass_name=first_pass_name + f"/{dataset_name}",
            recog_post_proc_funcs=list(recog_post_proc_funcs) + list(task.recog_post_proc_funcs),
            recog_pre_post_proc_funcs_ext=recog_pre_post_proc_funcs_ext,
        )
        search_error_rescore_dict[dataset_name] = search_error_rescore
        score_out = task.score_recog_output_func(dataset, recog_out)
        outputs[dataset_name] = score_out
        if search_error_check: #Just report the search error on test-other
            #config_ = config.copy()
            # if config.get("batch_size"):
            #     config_.pop("batch_size")
            if dataset_name == "test-other":
                search_error = check_search_error(dataset=dataset, model=model, hyps=hyps, config=config,
                                                  decoding_config=decoding_params, search_error_version=search_error_version,
                                                  prior_path=prior_path,
                                                  alias_name=f"{name}/search_error/{dataset_name}" if name else None,
                                                  )
            else:
                check_search_error(dataset=dataset, model=model, hyps=hyps, config=config,
                                   decoding_config=decoding_params, search_error_version=search_error_version,
                                   prior_path=prior_path,
                                   alias_name=f"{name}/search_error/{dataset_name}" if name else None,
                                   )
    # if dev_sets:
    #     assert task.main_measure_name == dev_sets[0]
    dataset_names = set.intersection(set(dev_sets), set(task.eval_datasets.keys()))
    search_error_key = "test-other" if "test-other" in dataset_names else list(dataset_names)[0]
    return task.collect_score_results_func(outputs), search_error, search_error_rescore_dict[search_error_key]


def compute_prior(
        *,
        dataset: DatasetConfig,
        model: ModelWithCheckpoint,
        mem_rqmt: Union[int, float] = 8,
        prior_alias_name: Optional[str] = None,
        num_shards: Optional[int] = None,
        prior_from_max: bool = False,
) -> tk.Path:
    if num_shards is not None:
        prior_frames_res = []
        prior_probs_res = []
        for i in range(num_shards):
            shard_prior_sum_job = ReturnnForwardJobV2(
                model_checkpoint=model.checkpoint,
                returnn_config=prior_config(dataset, model.definition, shard_index=i, num_shards=num_shards),
                output_files=["output_frames.npy", "output_probs.npy"],
                returnn_python_exe=tools_paths.get_returnn_python_exe(),
                returnn_root=tools_paths.get_returnn_root(),
                device="cpu",
                time_rqmt=4,
                mem_rqmt=mem_rqmt,
                cpu_rqmt=4,
            )
            prior_frames_res.append(shard_prior_sum_job.out_files["output_frames.npy"])
            prior_probs_res.append(shard_prior_sum_job.out_files["output_probs.npy"])
        prior_job = PriorCombineShardsJob(prior_frames_res, prior_probs_res)
        if prior_alias_name:
            prior_job.add_alias(prior_alias_name)
        res = prior_job.out_comined_results
    else:
        prior_job = ReturnnForwardJobV2(
            model_checkpoint=model.checkpoint,
            returnn_config=prior_config(dataset, model.definition),
            output_files=["prior.txt"],
            returnn_python_exe=tools_paths.get_returnn_python_exe(),
            returnn_root=tools_paths.get_returnn_root(),
            device="gpu",
            time_rqmt=4,
            mem_rqmt=mem_rqmt,
            cpu_rqmt=4,
        )
        if prior_alias_name:
            prior_job.add_alias(prior_alias_name)
            res = prior_job.out_files["prior.txt"]

    return res

def get_GroundTruth_with_score_on_forced_alignment(
        *,
        dataset: DatasetConfig,
        model: ModelWithCheckpoint,
        prior_path: tk.Path,
        decoding_config: dict,
        mem_rqmt: Union[int, float] = 16,
        alias_name: Optional[str] = None,
) -> tk.Path:
    from .experiments.ctc import scoring_v2
    pre_SearchError_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint,
        returnn_config=search_error_config(dataset, model.definition, scoring_v2,
                                           decoding_config=decoding_config, prior_path=prior_path),
        output_files=[_v2_forward_out_filename],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        device="gpu",
        time_rqmt=4,
        mem_rqmt=8,
        cpu_rqmt=2,
    )
    if alias_name:
        alias_name += "_get_GT_with_score"
        pre_SearchError_job.add_alias(alias_name)
    ground_truth_out = pre_SearchError_job.out_files[_v2_forward_out_filename]
    return ground_truth_out

def check_search_error(
        *,
        dataset: DatasetConfig,
        model: ModelWithCheckpoint,
        hyps: tk.path,
        decoding_config: dict,
        prior_path: tk.Path,
        mem_rqmt: Union[int, float] = 16,
        config: Optional[Dict[str, Any]] = None,
        alias_name: Optional[str] = None,
        search_error_version: int = 2,
) -> tk.Path:
    from .experiments.ctc import scoring, scoring_v2, scoring_v3
    scoring_func = {2:scoring_v2, 3:scoring_v3}[search_error_version]
    scoring_func = scoring_v2 if not decoding_config["use_lm"] else scoring_func
    if decoding_config.get("lm_order"):
        if "gram" in decoding_config.get("lm_order"):
            scoring_func = scoring
    pre_SearchError_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint,
        returnn_config=search_error_config(dataset, model.definition, scoring_func,
                                           decoding_config=decoding_config, config=config, prior_path=prior_path),
        output_files=[_v2_forward_out_filename],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        device="gpu",
        time_rqmt=4,
        mem_rqmt=8,
        cpu_rqmt=2,
    )
    if USE_24GB:
        pre_SearchError_job.rqmt.update({"gpu_mem": 24})
    if config.get("batch_size",None):
        pre_SearchError_job.rqmt.update({"gpu_mem": 24 if config["batch_size"] > 50_000_000 else 10})
    if alias_name:
        alias_name += "_scor_v2" if scoring_func is scoring_v2 else ("_scor_v3" if scoring_func is scoring_v3 else "")
        pre_SearchError_job.add_alias(alias_name)
    ground_truth_out = pre_SearchError_job.out_files[_v2_forward_out_filename]
    from .utils.search_error import ComputeSearchErrorsJob
    res = ComputeSearchErrorsJob(ground_truth_out, hyps).out_search_errors
    #tk.register_output(alias_name + "/search_error_job", res)
    return res

def search_dataset(
    *,
    decoding_config: dict,
    dataset: DatasetConfig,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    prior_path: tk.Path,
    config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    search_mem_rqmt: Union[int, float] = 6,
    search_rqmt: Optional[Dict[str, Any]] = None,
    search_alias_name: Optional[str] = None,
    first_pass_name: Optional[str] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    recog_pre_post_proc_funcs_ext: Sequence[Callable] = (),
) -> Tuple[RecogOutput, tk.path, tk.Path]:
    """
    Recog on the specific dataset using RETURNN.

    The core API which is supposed to perform the search inside RETURNN
    is via ``recog_def``.

    This function already performs a couple of post-processing steps
    such as collapsing repeated labels, removing blank labels,
    others specified in ``recog_post_proc_funcs`` (e.g. BPE to words),
    and finally taking the best hyp from a beam of hyps.

    This function is usually used as part of :func:`recog_model`.

    :param dataset: dataset config
    :param model: model def with checkpoint
    :param recog_def: recog def, which defines how to get the search output from the model and input data
    :param config: any additional search config for RETURNN
    :param search_post_config: any additional search post config (non-hashed settings) for RETURNN
    :param search_mem_rqmt: memory requirement for the search job
    :param search_rqmt: any additional requirements for the search job
    :param search_alias_name: alias name for the search job
    :param recog_post_proc_funcs: post processing functions for the recog output
    :param recog_pre_post_proc_funcs_ext:
    Funcs (recog_out, *, raw_res_search_labels, raw_res_labels, search_labels_to_labels, **other).
    Those are run before recog_post_proc_funcs,
    and they additionally get also
    ``raw_res_search_labels`` (e.g. align labels, e.g. BPE including blank)
    and ``raw_res_labels`` (e.g. BPE labels).
    :return: :class:`RecogOutput`, single best hyp (if there was a beam, we already took the best one)
        over the dataset

    """
    decoding_config = decoding_config.copy()
    cheat = decoding_config.pop("cheat", False)
    check_rescore_search_error = decoding_config.pop("check_search_error_rescore", False)
    env_updates = None
    if (config and config.get("__env_updates")) or (search_post_config and search_post_config.get("__env_updates")):
        env_updates = (config and config.pop("__env_updates", None)) or (
            search_post_config and search_post_config.pop("__env_updates", None)
        )
    if getattr(model.definition, "backend", None) is None:
        search_job = ReturnnSearchJobV2(
            search_data=dataset.get_main_dataset(),
            model_checkpoint=model.checkpoint,
            returnn_config=search_config(
                dataset, model.definition, recog_def, config=config, post_config=search_post_config
            ),
            returnn_python_exe=tools_paths.get_returnn_python_exe(),
            returnn_root=tools_paths.get_returnn_root(),
            output_gzip=True,
            log_verbosity=5,
            mem_rqmt=search_mem_rqmt,
        )
        res = search_job.out_search_file
    else:
        out_files = [_v2_forward_out_filename]
        if config and config.get("__recog_def_ext", False):
            out_files.append(_v2_forward_ext_out_filename)
        search_job = ReturnnForwardJobV2(
            model_checkpoint=model.checkpoint,
            returnn_config=search_config_v2(
                dataset, model.definition, recog_def, decoding_config, prior_path, config=config, post_config=search_post_config
            ),
            output_files=out_files,
            returnn_python_exe=tools_paths.get_returnn_python_exe(),
            returnn_root=tools_paths.get_returnn_root(),
            mem_rqmt=search_mem_rqmt,
        )
        res = search_job.out_files[_v2_forward_out_filename]
    res_with_score = res.copy() # With orig scores, no 2. pass, for search error check
    if search_rqmt:
        search_job.rqmt.update(search_rqmt)
    if USE_24GB:
        search_job.rqmt.update({"gpu_mem":24})
    if env_updates:
        for k, v in env_updates.items():
            search_job.set_env(k, v)
    if first_pass_name:
        search_job.add_alias(first_pass_name)

    raw_res_search_labels = RecogOutput(output=res)
    if recog_def.output_blank_label:
        raw_res_labels = ctc_alignment_to_label_seq(raw_res_search_labels, blank_label=recog_def.output_blank_label)
        # if rescoring:
        #     lm_rescor_name, rescoringLM = rescoringLM
        #     assert isinstance(rescoringLM, tk.Path)
        #     from .experiments.lm.llm import HuggingFaceLmRescoringJob
        #     res = HuggingFaceLmRescoringJob(
        #         model_dir=rescoringLM,
        #         weight=0.2,
        #         recog_out_file=res,
        #         llm_name=lm_rescor_name,
        #         lower_case=False,
        #     )
        #     res.add_alias("rescoring/" + lm_rescor_name +  "/" + search_alias_name)
        #     res = res.out_file
        #     tk.register_output(
        #         "rescoring/" + lm_rescor_name + "/" + search_alias_name,
        #         res)
        res_with_score = res.copy()
    else:
        raw_res_labels = raw_res_search_labels

    res_with_score = ctc_alignment_to_label_seq(RecogOutput(output=res_with_score), blank_label=recog_def.output_blank_label)
    res_with_score = SearchTakeBestWithScoreJob(res_with_score.output, output_gzip=True).out_best_search_results
    # ---Get the GT and scores here---
    gt_res = get_GroundTruth_with_score_on_forced_alignment(dataset=dataset, model=model, prior_path=prior_path, decoding_config=decoding_config, alias_name=first_pass_name)
    # --- ----------------------------
    for f in recog_pre_post_proc_funcs_ext:
        gt_res = f(
            RecogOutput(output=gt_res),
            dataset=dataset,
            raw_res_search_labels=raw_res_search_labels,
            raw_res_labels=raw_res_labels,
            search_labels_to_labels=functools.partial(
                ctc_alignment_to_label_seq, blank_label=recog_def.output_blank_label
            ),
            alias_name=search_alias_name,
        ).output
        res = f(
            RecogOutput(output=res),
            dataset=dataset,
            raw_res_search_labels=raw_res_search_labels,
            raw_res_labels=raw_res_labels,
            search_labels_to_labels=functools.partial(
                ctc_alignment_to_label_seq, blank_label=recog_def.output_blank_label
            ),
            alias_name=search_alias_name,
        ).output
    from .experiments.decoding.rescoring import RescoreSearchErrorJob, RescoreCheatJob
    search_error_rescore = None
    if check_rescore_search_error:# and os.path.basename(search_alias_name) == "test-clean":#"test-other":
        search_error_rescore_job = RescoreSearchErrorJob(combined_search_py_output=res, combined_gt_py_output=gt_res)
        search_error_rescore_job.add_alias(search_alias_name + "/search_error_job")
        search_error_rescore = search_error_rescore_job.out_search_errors
        #tk.register_output(search_alias_name + "/search_error_rescore", search_error_rescore)

    if cheat:
        cheat_job = RescoreCheatJob(combined_search_py_output=res, combined_gt_py_output=gt_res)
        #cheat_job.add_alias(search_alias_name + "/search_error_job")
        res = cheat_job.out_search_results

    if len(recog_pre_post_proc_funcs_ext) < 1:
        res = raw_res_labels.output
    for f in recog_post_proc_funcs:  # for example BPE to words
        res = f(RecogOutput(output=res)).output
    if recog_def.output_with_beam:
        # Don't join scores here (SearchBeamJoinScoresJob).
        #   It's not clear whether this is helpful in general.
        #   As our beam sizes are very small, this might boost some hyps too much.
        res = SearchTakeBestJob(res, output_gzip=True).out_best_search_results
    return RecogOutput(output=res), res_with_score, search_error_rescore#search_job.out_files[_v2_forward_out_filename]

class SearchTakeBestWithScoreJob(sisyphus.Job):
    """
    From RETURNN beam search results, extract the best result for each sequence.
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(self, search_py_output: tk.Path, *, output_gzip: bool = False):
        """
        :param search_py_output: a search output file from RETURNN in python format (n-best)
        :param output_gzip: if True, the output will be gzipped
        """
        self.search_py_output = search_py_output
        self.out_best_search_results = self.output_path("best_search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield sisyphus.Task("run", mini_task=True)

    def run(self):
        """run"""
        import i6_core.util as util
        d = eval(util.uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_best_search_results.get_path())
        with util.uopen(self.out_best_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                assert isinstance(entry, list)
                # n-best list as [(score, text), ...]
                best_score, best_entry = max(entry)
                out.write("%r: [(%r, %r)],\n" % (seq_tag, best_score, best_entry))
            out.write("}\n")

def ctc_alignment_to_label_seq(recog_output: RecogOutput, *, blank_label: str) -> RecogOutput:
    """
    Convert CTC alignment to label sequence.
    (Used by :func:`search_dataset`.)

    :param recog_output: comes out of search, alignment label frames incl blank
    :param blank_label: from the vocab. e.g. via ``recog_def.output_blank_label``
    :return: recog output with repetitions collapsed, and blank removed
    """
    # Also assume we should collapse repeated labels first.
    res = SearchCollapseRepeatedLabelsJob(recog_output.output, output_gzip=True).out_search_results
    res = SearchRemoveLabelJob(res, remove_label=blank_label, output_gzip=True).out_search_results
    return RecogOutput(output=res)

# Those are applied for both training, recog and potential others.
# The values are only used if they are neither set in config nor post_config already.
# They should also not infer with other things from the epilog.
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}


def search_config(
    dataset: DatasetConfig,
    model_def: ModelDef,
    recog_def: RecogDef,
    *,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    """
    config for search
    """
    returnn_recog_config_dict = dict(
        use_tensorflow=True,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        dev=dataset.get_main_dataset(),
        **(config or {}),
    )

    extern_data_raw = dataset.get_extern_data()
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed(extern_data_raw)

    returnn_recog_config = ReturnnConfig(
        config=returnn_recog_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
                    serialization.Import(model_def, import_as="_model_def", ignore_import_as_for_hash=True),
                    serialization.Import(recog_def, import_as="_recog_def", ignore_import_as_for_hash=True),
                    serialization.Import(_returnn_get_network, import_as="get_network", use_for_hash=False),
                    serialization.ExplicitHash(
                        {
                            # Increase the version whenever some incompatible change is made in this recog() function,
                            # which influences the outcome, but would otherwise not influence the hash.
                            "version": 1,
                        }
                    ),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                    serialization.PythonModelineNonhashedCode,
                ]
            )
        ],
        post_config=dict(  # not hashed
            log_batch_size=True,
            tf_log_memory_usage=True,
            tf_session_opts={"gpu_options": {"allow_growth": True}},
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # flat_net_construction=True,
            torch_log_memory_usage=True,
            watch_memory=True,
            use_lovely_tensors=True,
            forward_auto_split_batch_on_oom=True,
        ),
        sort_config=False,
    )

    (returnn_recog_config.config if recog_def.batch_size_dependent else returnn_recog_config.post_config).update(
        dict(
            batching="sorted",
            batch_size=20000,
            max_seqs=200,
        )
    )

    if post_config:
        returnn_recog_config.post_config.update(post_config)

    for k, v in SharedPostConfig.items():
        if k in returnn_recog_config.config or k in returnn_recog_config.post_config:
            continue
        returnn_recog_config.post_config[k] = v

    return returnn_recog_config

def search_error_config(
        dataset: DatasetConfig,
        model_def: Union[ModelDef, ModelDefWithCfg],
        scoring_func: Callable,
        decoding_config: dict,
        prior_path: tk.Path,
        *,
        config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    # changing these does not change the hash
    post_config = dict(  # not hashed
        log_batch_size=True,
        torch_log_memory_usage=True,
        watch_memory=True,
        use_lovely_tensors=True,
    )

    forward_data = dataset.get_main_dataset()

    returnn_config_dict = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        forward_data=forward_data,
    )
    if config:
        returnn_config_dict.update(config)
    if isinstance(model_def, ModelDefWithCfg):
        returnn_config_dict.update(model_def.config)

    extern_data_raw = dataset.get_extern_data()
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed(extern_data_raw)

    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe

    if hasattr(dataset, "vocab"):
        vocab = dataset.vocab
        if isinstance(vocab, Bpe):
            lexicon = get_bpe_lexicon(vocab)
        else:
            print(f"Getting lexicon for vocab {vocab} is not implemented!!!")
            lexicon = None
    else:
        print("No vocab found in dataset!!!")
        lexicon = None
    decoder_params = decoding_config.copy()  # Since decoding_config is shared across runs for different lm

    if decoder_params.get("lm_order",None) is not None:
        lm_name = decoder_params["lm_order"]
        if lm_name[0].isdigit():  # Use count based n-gram, it is already in decoder_params["lm"] TODO: maybe pass the lm config here and get the lm here
            lm = decoder_params.pop("lm", 0)
            assert lm != 0, "no count based lm given in decoding_config!!!"
        elif lm_name.startswith(
                "ffnn") or lm_name.startswith("trafo"):  # Actually lm should be none not only for ffnn, can change to something else later
            decoder_params.pop("lm", 0)
            lm = None
        else:
            raise NotImplementedError(f"Unknown lm_name {lm_name}")
    else:
        lm = None
        decoder_params.pop("beam_size",None)

    # Remove possible irrelevant params
    if not scoring_func.beam_size_dependent:
        decoder_params.pop("beam_size",None)

    decoder_params["nbest"] = 1
    decoder_params.pop("beam_threshold", None) # Only relevant for flashlight decoder, wont use it any way

    args = {"lm": lm, "lexicon": lexicon, "hyperparameters": decoder_params}
    if prior_path:
        args["prior_file"] = prior_path

    returnn_config = ReturnnConfig(
        config=returnn_config_dict,
        post_config=post_config,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
                    *serialize_model_def(model_def),
                    serialization.PartialImport(
                        code_object_path=scoring_func,
                        unhashed_package_root=None,
                        hashed_arguments=args,
                        unhashed_arguments={},
                        import_as="_scoring_def",
                        ignore_import_as_for_hash=True),
                    serialization.Import(_returnn_v2_get_model, import_as="get_model"),
                    serialization.Import(_returnn_search_error_step, import_as="forward_step"),
                    serialization.Import(_returnn_search_error_forward_callback, import_as="forward_callback"),
                    serialization.ExplicitHash({"version": 7}),  #1: eos added 2: eos not added 5. 1+aux info added 6. added label_prob mask 7. pre-stable
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                    serialization.PythonModelineNonhashedCode,
                ]
            )
        ],
        sort_config=False
    )
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config

    # There might be some further functions in the config, e.g. some dataset postprocessing.
    returnn_config = get_serializable_config(
        returnn_config,
        # The only dim tags we directly have in the config are via extern_data, maybe also model_outputs.
        # All other dim tags are inside functions such as get_model or train_step,
        # so we do not need to care about them here, only about the serialization of those functions.
        # Those dim tags and those functions are already handled above.
        serialize_dim_tags=False,
    )

    batch_size_dependent = scoring_func.batch_size_dependent
    if "__batch_size_dependent" in returnn_config.config:
        batch_size_dependent = returnn_config.config.pop("__batch_size_dependent")
    if "__batch_size_dependent" in returnn_config.post_config:
        batch_size_dependent = returnn_config.post_config.pop("__batch_size_dependent")
    for k, v in dict(
        batching="sorted",
        batch_size=20000 * model_def.batch_size_factor,
        max_seqs=200,
    ).items():
        if k in returnn_config.config:
            v = returnn_config.config.pop(k)
        if k in returnn_config.post_config:
            v = returnn_config.post_config.pop(k)
        (returnn_config.config if batch_size_dependent else returnn_config.post_config)[k] = v

    if post_config:
        returnn_config.post_config.update(post_config)

    for k, v in SharedPostConfig.items():
        if k in returnn_config.config or k in returnn_config.post_config:
            continue
        returnn_config.post_config[k] = v
    # print(f"\n\nscoring config:{returnn_config.config}")
    # print(f"\nscoring post config:{returnn_config.post_config}")
    return returnn_config


def _returnn_search_error_step(*, model, extern_data: TensorDict, **kwargs):
    from returnn.config import get_global_config
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx+1}/{batch_size} seq_tag: {seq_tag!r}")

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    default_target_key = config.typed_value("target")
    targets = extern_data[default_target_key]
    targets_spatial_dim = targets.get_time_dim_tag()
    scoring_def = config.typed_value("_scoring_def")
    scoring_out, score_dim, n_oovs, oov_dim = scoring_def(model=model, data=data, targets=targets,
                                                          data_spatial_dim=data_spatial_dim,seq_tags=extern_data["seq_tag"])
    rf.get_run_ctx().mark_as_output(scoring_out, "scores", dims=[batch_dim, score_dim])
    rf.get_run_ctx().mark_as_output(n_oovs, "n_oovs", dims=[batch_dim, oov_dim])
    rf.get_run_ctx().mark_as_output(targets, "targets", dims=[batch_dim, targets_spatial_dim])

def _returnn_search_error_forward_callback():
    from typing import TextIO
    from returnn.tensor import Tensor, Dim, TensorDict
    from returnn.forward_iface import ForwardCallbackIface

    class _ReturnnSearchErrorForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.out_file: Optional[TextIO] = None

        def init(self, *, model):
            import gzip

            self.out_file = gzip.open(_v2_forward_out_filename, "wt")
            self.out_file.write("{\n")


        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            score: Tensor = outputs["scores"] # [1]
            target: Tensor = outputs["targets"] # [1, out_spatial]
            n_oov: Tensor = outputs["n_oovs"] # [1]
            # self.out_file.write(f"{seq_tag!r}: [\n")
            # self.out_file.write(f"  ({score!r}, {target!r})\n")
            # self.out_file.write("],\n")
            assert target.sparse_dim and target.sparse_dim.vocab  # should come from the model
            self.out_file.write(f"{seq_tag!r}: [\n")
            score = float(score.raw_tensor[0])
            n_oov = int(n_oov.raw_tensor[0])
            target_ids = target.raw_tensor
            # #################
            # import pdb
            # pdb.set_trace()
            # ###############
            target_serialized = target.sparse_dim.vocab.get_seq_labels(target_ids)
            target_serialized = target_serialized.replace("@@ ", "")
            self.out_file.write(f"  ({score!r}, {target_serialized!r},{n_oov!r}),\n")
            self.out_file.write("],\n")

        def finish(self):
            self.out_file.write("}\n")
            self.out_file.close()

    return _ReturnnSearchErrorForwardCallbackIface()

def prior_config(
        dataset: DatasetConfig,
        model_def: Union[ModelDef, ModelDefWithCfg],
        *,
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
) -> ReturnnConfig:
    # changing these does not change the hash
    post_config = dict(  # not hashed
        log_batch_size=True,
        torch_log_memory_usage=True,
        watch_memory=True,
        use_lovely_tensors=True,
    )

    if num_shards is not None and shard_index is not None:
        forward_data = dataset.get_sharded_main_dataset(shard_index, num_shards)
    else:
        forward_data = dataset.get_main_dataset()

    returnn_config_dict = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        forward_data=forward_data,
        #####
        batch_size=500 * 16000,
        max_seqs=240,
    )
    if isinstance(model_def, ModelDefWithCfg):
        returnn_config_dict.update(model_def.config)

    extern_data_raw = dataset.get_extern_data()
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed(extern_data_raw)

    callback_fn = _returnn_get_prior_forward_callback if num_shards is None or shard_index is None else _returnn_get_sharded_prior_forward_callback

    returnn_config = ReturnnConfig(
        config=returnn_config_dict,
        post_config=post_config,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
                    *serialize_model_def(model_def),
                    serialization.Import(_returnn_v2_get_model, import_as="get_model"),
                    serialization.Import(_returnn_prior_step, import_as="forward_step"),
                    serialization.Import(callback_fn, import_as="forward_callback"),
                    serialization.ExplicitHash({"version": 1 + (2000 if num_shards is not None else 0)}),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                    serialization.PythonModelineNonhashedCode,
                ]
            )
        ],
        sort_config=False
    )
    return returnn_config


def _returnn_prior_step(*, model, extern_data: TensorDict, **kwargs):
    from returnn.config import get_global_config
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()

    logits, enc, audio_features_len = model(data, in_spatial_dim=data_spatial_dim)
    logprobs = model.log_probs_wb_from_logits(logits)

    rf.get_run_ctx().mark_as_output(logprobs, "logprobs", dims=[batch_dim, audio_features_len, logprobs.dims[-1]])
    rf.get_run_ctx().mark_as_output(audio_features_len.dyn_size_ext.raw_tensor.cpu(), "lengths")


def _returnn_get_sharded_prior_forward_callback():
    from returnn.tensor import Tensor, Dim, TensorDict
    from returnn.forward_iface import ForwardCallbackIface
    from scipy.special import logsumexp

    class _ReturnnPriorForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.sum_logprobs = None
            self.sum_frames = 0

        def init(self, *, model):
            pass

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            logprobs: Tensor = outputs["logprobs"]
            lengths: Tensor = outputs["lengths"]

            assert lengths.raw_tensor == logprobs.raw_tensor.shape[0], "Prior calculation lengths are not the same!"

            self.sum_frames += np.sum(lengths.raw_tensor)
            if self.sum_logprobs is None:
                self.sum_logprobs = logsumexp(logprobs.raw_tensor, axis=0)
            else:
                self.sum_logprobs = np.logaddexp(self.sum_logprobs, logsumexp(logprobs.raw_tensor, axis=0))

        def finish(self):
            all_frames = self.sum_frames
            all_frames = np.array([all_frames])
            all_logprobs = self.sum_logprobs
            with open("output_frames.npy", "wb") as f:
                np.save(f, all_frames)
            with open("output_probs.npy", "wb") as f:
                np.save(f, all_logprobs)

    return _ReturnnPriorForwardCallbackIface()


def _returnn_get_prior_forward_callback():
    from returnn.tensor import Tensor, Dim, TensorDict
    from returnn.forward_iface import ForwardCallbackIface
    from scipy.special import logsumexp

    class _ReturnnPriorForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.sum_logprobs = None
            self.sum_frames = 0

        def init(self, *, model):
            pass

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            logprobs: Tensor = outputs["logprobs"]
            lengths: Tensor = outputs["lengths"]

            assert lengths.raw_tensor == logprobs.raw_tensor.shape[0], "Prior calculation lengths are not the same!"

            self.sum_frames += np.sum(lengths.raw_tensor)
            if self.sum_logprobs is None:
                self.sum_logprobs = logsumexp(logprobs.raw_tensor, axis=0)
            else:
                self.sum_logprobs = np.logaddexp(self.sum_logprobs, logsumexp(logprobs.raw_tensor, axis=0))

        def finish(self):
            all_frames = self.sum_frames
            all_logprobs = self.sum_logprobs
            log_average_probs = all_logprobs - np.log(all_frames)
            average_probs = np.exp(log_average_probs)
            print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
            with open("prior.txt", "w") as f:
                np.savetxt(f, log_average_probs, delimiter=" ")
            print("Saved prior in prior.txt in +log space.")

    return _ReturnnPriorForwardCallbackIface()

def search_config_v2(
    dataset: DatasetConfig,
    model_def: Union[ModelDef, ModelDefWithCfg],
    recog_def: RecogDef,
    decoding_config: dict,
    prior_path: tk.Path,
    *,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    """
    Create config for search.

    v2: Use any backend (usually PyTorch) and the new API (get_model, forward_step).

    TODO should use sth like unhashed_package_root (https://github.com/rwth-i6/i6_experiments/pull/157)
    """
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config

    returnn_recog_config_dict = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        forward_data=dataset.get_main_dataset(),
    )
    if config:
        returnn_recog_config_dict.update(config)
    if isinstance(model_def, ModelDefWithCfg):
        returnn_recog_config_dict.update(model_def.config)

    extern_data_raw = dataset.get_extern_data()
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed(extern_data_raw)

    decoder_params = decoding_config.copy()  # Since decoding_config is shared across runs for different lm
    if recog_def is model_recog_lm:
        from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe

        if hasattr(dataset, "vocab"):
            vocab = dataset.vocab
            if isinstance(vocab, Bpe):
                lexicon = get_bpe_lexicon(vocab)
            else:
                print(f"Getting lexicon for vocab {vocab} is not implemented!!!")
                lexicon = None
        else:
            print("No vocab found in dataset!!!")
            lexicon = None

        if "lm_order" in decoder_params:
            lm_name = decoder_params["lm_order"]
            if lm_name[0].isdigit(): #Use count based n-gram as arpa(bin) file_name it is already in decoder_params["lm"] TODO: maybe pass the lm config here and get the lm here
                lm = decoder_params.pop("lm", 0)
                assert lm != 0, "no count based lm given in decoding_config!!!"
            elif lm_name.startswith("ffnn") or lm_name.startswith("trafo"): # Actually lm should be none not only for ffnn, can change to something else later
                decoder_params.pop("lm", 0)
                lm = None
            else:
                raise NotImplementedError(f"Unknown lm_name {lm_name}")
        else:
            lm = None
            decoder_params.pop("lm_weight")
            if decoder_params.get("nbest", 1) == 1:
                decoder_params.pop("beam_size") #No LM just equivalent to greedy

        args = {"lm": lm, "lexicon": lexicon, "hyperparameters": decoder_params}
        if prior_path:
            args["prior_file"] = prior_path

    elif recog_def in (model_recog_flashlight, recog_nn, model_recog):
        if recog_def == recog_nn and decoder_params.get("nbest",1) > 1:
            decoder_params["use_recombination"] = True
            decoder_params["recomb_after_topk"] = True
            decoder_params["recomb_blank"] = True
        args = {"hyperparameters": decoder_params}
        if prior_path:
            args["prior_file"] = prior_path
        if "lm_order" not in decoder_params: # No LM for first pass
            decoder_params.pop("prior_weight", 0)
            decoder_params.pop("lm_weight", 0)
            args.pop("prior_file",0)


    returnn_recog_config = ReturnnConfig(
        config=returnn_recog_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
                    *serialize_model_def(model_def),
                    serialization.Import(_returnn_v2_get_model, import_as="get_model"),
                    serialization.PartialImport(
                        code_object_path=recog_def,
                        unhashed_package_root=None,
                        hashed_arguments=args,
                        unhashed_arguments={}, #TODO: For NoLM, the lm scale should not be hashed
                        import_as="_recog_def",
                        ignore_import_as_for_hash=True),
                    serialization.Import(_returnn_v2_forward_step, import_as="forward_step"),
                    serialization.Import(_returnn_v2_get_forward_callback, import_as="forward_callback"),
                    serialization.ExplicitHash(
                        {
                            # Increase the version whenever some incompatible change is made in this recog() function,
                            # which influences the outcome, but would otherwise not influence the hash.
                            "version": 15,
                        }
                    ),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                    serialization.PythonModelineNonhashedCode,
                ]
            )
        ],
        post_config=dict(  # not hashed
            log_batch_size=True,
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # flat_net_construction=True,
            torch_log_memory_usage=True,
            watch_memory=True,
            use_lovely_tensors=True,
        ),
        sort_config=False,
    )

    # There might be some further functions in the config, e.g. some dataset postprocessing.
    returnn_recog_config = get_serializable_config(
        returnn_recog_config,
        # The only dim tags we directly have in the config are via extern_data, maybe also model_outputs.
        # All other dim tags are inside functions such as get_model or train_step,
        # so we do not need to care about them here, only about the serialization of those functions.
        # Those dim tags and those functions are already handled above.
        serialize_dim_tags=False,
    )

    batch_size_dependent = recog_def.batch_size_dependent
    if "__batch_size_dependent" in returnn_recog_config.config:
        batch_size_dependent = returnn_recog_config.config.pop("__batch_size_dependent")
    if "__batch_size_dependent" in returnn_recog_config.post_config:
        batch_size_dependent = returnn_recog_config.post_config.pop("__batch_size_dependent")
    for k, v in dict(
        batching="sorted",
        batch_size=20000 * model_def.batch_size_factor,
        max_seqs=200,
    ).items():
        if k in returnn_recog_config.config:
            v = returnn_recog_config.config.pop(k)
        if k in returnn_recog_config.post_config:
            v = returnn_recog_config.post_config.pop(k)
        (returnn_recog_config.config if batch_size_dependent else returnn_recog_config.post_config)[k] = v

    if post_config:
        returnn_recog_config.post_config.update(post_config)

    for k, v in SharedPostConfig.items():
        if k in returnn_recog_config.config or k in returnn_recog_config.post_config:
            continue
        returnn_recog_config.post_config[k] = v
    # print(f"\n\nsearch config:{returnn_recog_config.config}")
    # print(f"\nsearch post config:{returnn_recog_config.post_config}")
    return returnn_recog_config


def _returnn_get_network(*, epoch: int, **_kwargs_unused) -> Dict[str, Any]:
    """called from the RETURNN config"""
    from returnn_common import nn
    from returnn.config import get_global_config
    from returnn.tf.util.data import Data

    nn.reset_default_root_name_ctx()
    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("extern_data")
    data = Data(name=default_input_key, **extern_data_dict[default_input_key])
    targets = Data(name=default_target_key, **extern_data_dict[default_target_key])
    data_spatial_dim = data.get_time_dim_tag()
    data = nn.get_extern_data(data)
    targets = nn.get_extern_data(targets)
    model_def = config.typed_value("_model_def")
    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
    recog_def = config.typed_value("_recog_def")
    recog_out = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim, targets_dim=targets.sparse_dim)
    assert isinstance(recog_out, nn.Tensor)
    recog_out.mark_as_default_output()
    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    return net_dict


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    assert targets.sparse_dim and targets.sparse_dim.vocab, f"no vocab for {targets}"

    model_def = config.typed_value("_model_def")
    model = model_def(epoch=epoch, in_dim=data.feature_dim_or_sparse_dim, target_dim=targets.sparse_dim)
    return model


def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim
    from returnn.config import get_global_config

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx+1}/{batch_size} seq_tag: {seq_tag!r}")

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    recog_def = config.typed_value("_recog_def")
    extra = {}
    if config.bool("cheating", False):
        default_target_key = config.typed_value("target")
        targets = extern_data[default_target_key]
        extra.update(dict(targets=targets, targets_spatial_dim=targets.get_time_dim_tag()))
    recog_out = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim, **extra)
    if len(recog_out) == 5:
        # recog results including beam {batch, beam, out_spatial},
        # log probs {batch, beam},
        # extra outputs {batch, beam, ...},
        # out_spatial_dim,
        # final beam_dim
        assert len(recog_out) == 5, f"mismatch, got {len(recog_out)} outputs with recog_def_ext=True"
        hyps, scores, extra, out_spatial_dim, beam_dim = recog_out
    elif len(recog_out) == 4:
        # same without extra outputs
        assert len(recog_out) == 4, f"mismatch, got {len(recog_out)} outputs recog_def_ext=False"
        hyps, scores, out_spatial_dim, beam_dim = recog_out
        extra = {}
    else:
        raise ValueError(f"unexpected num outputs {len(recog_out)} from recog_def")
    assert isinstance(hyps, Tensor) and isinstance(scores, Tensor)
    assert isinstance(out_spatial_dim, Dim) and isinstance(beam_dim, Dim)
    rf.get_run_ctx().mark_as_output(hyps, "hyps", dims=[batch_dim, beam_dim, out_spatial_dim])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, beam_dim])
    assert isinstance(extra, dict)
    for k, v in extra.items():
        assert isinstance(k, str) and isinstance(v, Tensor)
        assert v.dims[:2] == (batch_dim, beam_dim)
        rf.get_run_ctx().mark_as_output(v, k, dims=v.dims)


_v2_forward_out_filename = "output.py.gz"
_v2_forward_ext_out_filename = "output_ext.py.gz"


def _returnn_v2_get_forward_callback():
    from typing import TextIO
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.config import get_global_config

    config = get_global_config()
    recog_def_ext = config.bool("__recog_def_ext", False)

    class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.out_file: Optional[TextIO] = None
            self.out_ext_file: Optional[TextIO] = None

        def init(self, *, model):
            import gzip

            self.out_file = gzip.open(_v2_forward_out_filename, "wt")
            self.out_file.write("{\n")

            if recog_def_ext:
                self.out_ext_file = gzip.open(_v2_forward_ext_out_filename, "wt")
                self.out_ext_file.write("{\n")

        # Copied From Martens, TODO make it clear what makes the hyps output different.
        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            hyps: Tensor = outputs["hyps"]  # [beam, out_spatial]
            scores: Tensor = outputs["scores"]  # [beam]
            if hyps.sparse_dim and hyps.sparse_dim.vocab: # a bit hacky but works
                assert hyps.sparse_dim and hyps.sparse_dim.vocab  # should come from the model
                assert hyps.dims[1].dyn_size_ext, f"hyps {hyps} do not define seq lengths"
                # AED/Transducer etc will have hyps len depending on beam -- however, CTC will not.
                hyps_len = hyps.dims[1].dyn_size_ext  # [beam] or []
                assert hyps.raw_tensor.shape[:1] == scores.raw_tensor.shape  # (beam,)
                if hyps_len.raw_tensor.shape:
                    assert scores.raw_tensor.shape == hyps_len.raw_tensor.shape  # (beam,)
                num_beam = hyps.raw_tensor.shape[0]
                # Consistent to old search task, list[(float,str)].
                self.out_file.write(f"{seq_tag!r}: [\n")
                for i in range(num_beam):
                    score = float(scores.raw_tensor[i])
                    hyp_ids = hyps.raw_tensor[
                        i, : hyps_len.raw_tensor[i] if hyps_len.raw_tensor.shape else hyps_len.raw_tensor
                    ]
                    hyp_serialized = hyps.sparse_dim.vocab.get_seq_labels(hyp_ids)
                    self.out_file.write(f"  ({score!r}, {hyp_serialized!r}),\n")
                self.out_file.write("],\n")

                if self.out_ext_file:
                    self.out_ext_file.write(f"{seq_tag!r}: [\n")
                    for v in outputs.data.values():
                        assert v.dims[0].dimension == num_beam
                    for i in range(num_beam):
                        d = {k: v.raw_tensor[i].tolist() for k, v in outputs.data.items() if k not in {"hyps", "scores"}}
                        self.out_ext_file.write(f"  {d!r},\n")
                    self.out_ext_file.write("],\n")
            else: # We are already having the words at hand
                assert hyps.raw_tensor.shape[:1] == scores.raw_tensor.shape  # (beam,)
                num_beam = hyps.raw_tensor.shape[0]
                # Consistent to old search task, list[(float,str)].
                self.out_file.write(f"{seq_tag!r}: [\n")
                for i in range(num_beam):
                    score = float(scores.raw_tensor[i])
                    words = hyps.raw_tensor[i][0]
                    self.out_file.write(f"  ({score!r}, {words!r}),\n")
                self.out_file.write("],\n")

                assert not self.out_ext_file, "not implemented"

        def finish(self):
            self.out_file.write("}\n")
            self.out_file.close()
            if self.out_ext_file:
                self.out_ext_file.write("}\n")
                self.out_ext_file.close()

    return _ReturnnRecogV2ForwardCallbackIface()

class GetRecogExp(sisyphus.Job):
    """
    Collect all info from recogs.
    The output is a JSON dict with the format::
    kept best_... for temporary compatibility
        {
            'best_scores': {...}  (ScoreResultCollection)
            'best_epoch': int,  (sub-epoch by RETURNN)
            ...  (other meta info)
        }

    !! Hash of this job mostly depends on the hash of recog_and_score_func
    TODO: Sometimes it fails silently, try to find out why and slowly get rid of this Job
    """

    def __init__(
        self,
        model: ModelWithCheckpoints,
        epoch: int,
        *,
        recog_and_score_func: Callable[[int], Tuple[ScoreResultCollection,Optional[tk.Path],Optional[tk.Path]]],
        version: int = 2,
    ):
        """
        :param model: modelwithcheckpoints, all fixed checkpoints + scoring file for potential other relevant checkpoints (see update())
        :param recog_and_score_func: epoch -> scores. called in graph proc
        """
        super(GetRecogExp, self).__init__()
        self.model = model
        self.epoch = epoch
        self.recog_and_score_func = recog_and_score_func
        self.out_summary_json = self.output_path("summary.json")
        self.out_search_error = self.output_path("search_error")
        self.out_search_error_rescore = self.output_path("search_error_rescore")
        self._scores_outputs = {}  # type: Dict[int, Tuple[ScoreResultCollection,Optional[tk.Path],Optional[tk.Path]]]  # epoch -> scores out
        self._add_recog(self.epoch)

    # @classmethod
    # def hash(cls, parsed_args: Dict[str, Any]) -> str:
    #     """
    #     :param parsed_args:
    #     :return: hash for job given the arguments
    #     """
    #     # Extend the default hash() function.
    #     d = parsed_args.copy()
    #     if not d["exclude_epochs"]:
    #         d.pop("exclude_epochs")
    #     exp: ModelWithCheckpoints = d["exp"]
    #     assert isinstance(exp, ModelWithCheckpoints)
    #     assert exp.fixed_epochs  # need some fixed epochs to define the hash
    #     last_fixed_epoch = max(exp.fixed_epochs)
    #     recog_and_score_func = d["recog_and_score_func"]
    #     res = recog_and_score_func(last_fixed_epoch)
    #     assert isinstance(res, ScoreResultCollection)
    #     # Add this to the hash, to make sure the pipeline of the recog and scoring influences the hash.
    #     d["_last_fixed_epoch_results"] = res
    #     return sis_tools.sis_hash(d)

    def _add_recog(self, epoch: int):
        if epoch in self._scores_outputs:
            return
        res, search_error, search_error_rescore = self.recog_and_score_func(epoch)
        #assert isinstance(res, Tuple[ScoreResultCollection,Optional[tk.path]])
        self._scores_outputs[epoch] = (res,search_error, search_error_rescore)

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task("run", rqmt={"cpu":1, "time":1})#mini_task=True)

    def run(self):
        """run"""
        import json

        best_epoch = self.epoch
        best_scores = json.load(open(self._scores_outputs[best_epoch][0].output.get_path()))

        res = {"best_scores": best_scores, "best_epoch": best_epoch}
        with open(self.out_summary_json.get_path(), "w") as f:
            f.write(json.dumps(res))
            f.write("\n")
        import shutil
        def set_output(dst, src):
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            if (src and src.get_path()):
                os.symlink(src.get_path(), dst)
                # shutil.copy2(self._scores_outputs[best_epoch][1].get_path(),self.out_search_error.get_path())

        search_error_dst = self.out_search_error.get_path()
        search_error_src = self._scores_outputs[best_epoch][1]
        set_output(search_error_dst, search_error_src)

        search_error_re_dst = self.out_search_error_rescore.get_path()
        search_error_re_src = self._scores_outputs[best_epoch][2]
        set_output(search_error_re_dst, search_error_re_src)

class GetBestRecogTrainExp(sisyphus.Job):
    """
    Collect all info from recogs.
    The output is a JSON dict with the format::

        {
            'best_scores': {...}  (ScoreResultCollection)
            'best_epoch': int,  (sub-epoch by RETURNN)
            ...  (other meta info)
        }
    """

    def __init__(
        self,
        exp: ModelWithCheckpoints,
        *,
        recog_and_score_func: Callable[[int], ScoreResultCollection],
        main_measure_lower_is_better: bool = True,
        check_train_scores_n_best: int = 2,
        exclude_epochs: Collection[int] = (),
    ):
        """
        :param exp: model, all fixed checkpoints + scoring file for potential other relevant checkpoints (see update())
        :param recog_and_score_func: epoch -> scores. called in graph proc
        :param check_train_scores_n_best: check train scores for N best checkpoints (per each measure)
        """
        super(GetBestRecogTrainExp, self).__init__()
        self.exp = exp
        self.recog_and_score_func = recog_and_score_func
        self.main_measure_lower_is_better = main_measure_lower_is_better
        self.check_train_scores_n_best = check_train_scores_n_best
        self.exclude_epochs = exclude_epochs
        self._update_checked_relevant_epochs = False
        self.out_summary_json = self.output_path("summary.json")
        self.out_results_all_epochs_json = self.output_path("results_all_epoch.json")
        self._scores_outputs = {}  # type: Dict[int, ScoreResultCollection]  # epoch -> scores out
        for epoch in exp.fixed_epochs:
            self._add_recog(epoch)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        # Extend the default hash() function.
        d = parsed_args.copy()
        if not d["exclude_epochs"]:
            d.pop("exclude_epochs")
        exp: ModelWithCheckpoints = d["exp"]
        assert isinstance(exp, ModelWithCheckpoints)
        assert exp.fixed_epochs  # need some fixed epochs to define the hash
        last_fixed_epoch = max(exp.fixed_epochs)
        recog_and_score_func = d["recog_and_score_func"]
        res = recog_and_score_func(last_fixed_epoch)
        assert isinstance(res[0], ScoreResultCollection)
        # Add this to the hash, to make sure the pipeline of the recog and scoring influences the hash.
        d["_last_fixed_epoch_results"] = res
        return sis_tools.sis_hash(d)

    def update(self):
        """
        This is run when all inputs have become available,
        and we can potentially add further inputs.
        The exp (ModelWithCheckpoints) includes a ref to scores_and_learning_rates
        which is only available when the training job finished,
        thus this is only run at the very end.

        Note that this is thus called multiple times,
        once scores_and_learning_rates becomes available,
        and then once the further recogs become available.
        However, only want to check for relevant checkpoints once.
        """
        if not self._update_checked_relevant_epochs and self.exp.scores_and_learning_rates.available():
            from datetime import datetime

            log_filename = tk.Path("update.log", self).get_path()
            try:
                os.makedirs(os.path.dirname(log_filename), exist_ok=True)
                log_stream = open(log_filename, "a")
            except PermissionError:  # maybe some other user runs this, via job import
                log_stream = open("/dev/stdout", "w")
            with log_stream:
                log_stream.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                log_stream.write(": get_relevant_epochs_from_training_learning_rate_scores\n")
                for epoch in get_relevant_epochs_from_training_learning_rate_scores(
                    model_dir=self.exp.model_dir,
                    model_name=self.exp.model_name,
                    scores_and_learning_rates=self.exp.scores_and_learning_rates,
                    n_best=self.check_train_scores_n_best,
                    log_stream=log_stream,
                ):
                    self._add_recog(epoch)
            self._update_checked_relevant_epochs = True

    def _add_recog(self, epoch: int):
        if epoch in self._scores_outputs:
            return
        if epoch in self.exclude_epochs:
            return
        res = self.recog_and_score_func(epoch)[0]
        assert isinstance(res, ScoreResultCollection)
        self.add_input(res.main_measure_value)
        self.add_input(res.output)
        self._scores_outputs[epoch] = res

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task("run", mini_task=True)

    def run(self):
        """run"""
        import ast
        import json

        scores = []  # (value,epoch) tuples
        for epoch, score in sorted(self._scores_outputs.items()):
            assert isinstance(score, ScoreResultCollection)
            value = ast.literal_eval(open(score.main_measure_value.get_path(), "r").read())
            if not self.main_measure_lower_is_better:
                value = -value
            scores.append((value, epoch))
        _, best_epoch = min(scores)
        best_scores = json.load(open(self._scores_outputs[best_epoch].output.get_path()))
        res = {"best_scores": best_scores, "best_epoch": best_epoch}
        with open(self.out_summary_json.get_path(), "w") as f:
            f.write(json.dumps(res))
            f.write("\n")

        with open(self.out_results_all_epochs_json.get_path(), "w") as f:
            f.write("{\n")
            count = 0
            for epoch, score in sorted(self._scores_outputs.items()):
                assert isinstance(score, ScoreResultCollection)
                if count > 0:
                    f.write(",\n")
                res = json.load(open(score.output.get_path()))
                f.write(f'  "{epoch}": {json.dumps(res)}')
                count += 1
            f.write("\n}\n")


class GetBestTuneValue(sisyphus.Job):
    def __init__(
            self,
            scores: list[tk.Path],
            tune_values: list[float],
            default_scale: float = None,
    ):
        self.scores = scores
        self.tune_values = tune_values
        self.default_scale = default_scale
        self.out_best_tune = self.output_path("best_tune.json")
        self.out_best_tune_var = self.output_var("best_tune")

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task("run", mini_task=True)#rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        """run"""
        import json

        best_score_idx = -1
        best_score_val = 1000000.0
        for i in range(len(self.scores)):
            d = eval(uopen(self.scores[i], "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            assert isinstance(d, dict), "Has to be a dict containing the best score."

            if d["best_scores"]["dev-other"]  < best_score_val: #  + d["best_scores"]["test-other"] !  Implicitly first best score when multiple best. Dependent on the ordering!
                best_score_idx = i
                best_score_val = d["best_scores"]["dev-other"] #  + d["best_scores"]["test-other"]

        best_tune = self.tune_values[best_score_idx]
        if self.default_scale:
            self.out_best_tune_var.set(best_tune + self.default_scale)
        res = {"best_tune": best_tune}
        with open(self.out_best_tune.get_path(), "w") as f:
            f.write(json.dumps(res))
            f.write("\n")



class GetTorchAvgModelResult(sisyphus.Job):
    """
    Collect all info from recogs.
    The output is a JSON dict with the format::

        {
            'best_scores': {...}  (ScoreResultCollection)
            'best_epoch': int,  (sub-epoch by RETURNN)
            ...  (other meta info)
        }
    """

    def __init__(
        self,
        exp: ModelWithCheckpoints,
        *,
        recog_and_score_func: Callable[[PtCheckpoint], ScoreResultCollection],
        end_fraction: float = 0.1,
        train_scores_n_best: Optional[int] = 2,
        include_fixed_epochs: bool = False,
        include_last_n: Optional[int] = 2,
        exclude_epochs: Collection[int] = (),
    ):
        """
        :param exp: model, all fixed checkpoints + scoring file for potential other relevant checkpoints (see update())
        :param recog_and_score_func: epoch -> scores. called in graph proc
        :param end_fraction: takes the last models, e.g. 0.05 means, if last epoch is 500, take only from epochs 450-500
        :param train_scores_n_best: take best N via dev scores from train job (per each measure) (if in end_fraction)
            If None, determine automatically from returnn config. Use 0 to disable.
        :param include_fixed_epochs: consider all `keep` epochs (but only if inside end_fraction)
        :param include_last_n: make sure less or equal to keep_last_n (only those inside end_fraction).
            If None, determine automatically from returnn config. Use 0 to disable.
        :param exclude_epochs:
        """
        super(GetTorchAvgModelResult, self).__init__()
        self.exp = exp
        self.recog_and_score_func = recog_and_score_func
        self.end_fraction = end_fraction
        self.exclude_epochs = exclude_epochs
        self._update_checked_relevant_epochs = False
        self.out_avg_checkpoint = PtCheckpoint(self.output_path("model/average.pt"))
        self.out_results = self.output_path("results.json")
        self.out_main_measure_value = self.output_path("main_measure_value.json")
        self.res = ScoreResultCollection(main_measure_value=self.out_main_measure_value, output=self.out_results)
        self.out_merged_epochs_list = self.output_path("merged_epochs_list.json")
        self._in_checkpoints: Dict[int, PtCheckpoint] = {}
        self._in_avg_checkpoint: Optional[PtCheckpoint] = None
        self._scores_output: Optional[ScoreResultCollection] = None
        self.last_epoch = exp.last_fixed_epoch_idx
        self.first_epoch = int(exp.last_fixed_epoch_idx * (1 - end_fraction))
        if train_scores_n_best is None:
            train_scores_n_best = self._get_keep_best_n_from_learning_rate_file(exp.scores_and_learning_rates)
        self.train_scores_n_best = train_scores_n_best
        if include_fixed_epochs:
            for epoch in exp.fixed_epochs:
                self._add_recog(epoch)
        if include_last_n is None:
            include_last_n = self._get_keep_last_n_from_learning_rate_file(exp.scores_and_learning_rates)
        for epoch in range(self.last_epoch - include_last_n + 1, self.last_epoch + 1):
            self._add_recog(epoch)
        self.update()

    @staticmethod
    def _get_keep_last_n_from_learning_rate_file(lr_file: tk.Path) -> int:
        # exp.fixed_epochs does not cover keep_last_n (intentionally, it does not want to cover too much),
        # but for the model average, we want to consider those as well.
        training_job = lr_file.creator
        assert isinstance(training_job, ReturnnTrainingJob)
        cleanup_old_models = training_job.returnn_config.post_config.get("cleanup_old_models", None)
        keep_last_n = cleanup_old_models.get("keep_last_n", None) if isinstance(cleanup_old_models, dict) else None
        if keep_last_n is None:
            keep_last_n = 2  # default
        return keep_last_n

    @staticmethod
    def _get_keep_best_n_from_learning_rate_file(lr_file: tk.Path) -> int:
        # exp.fixed_epochs does not cover keep_last_n (intentionally, it does not want to cover too much),
        # but for the model average, we want to consider those as well.
        training_job = lr_file.creator
        assert isinstance(training_job, ReturnnTrainingJob)
        cleanup_old_models = training_job.returnn_config.post_config.get("cleanup_old_models", None)
        keep_best_n = cleanup_old_models.get("keep_best_n", None) if isinstance(cleanup_old_models, dict) else None
        if keep_best_n is None:
            keep_best_n = 4  # default
        return keep_best_n

    def update(self):
        """
        This is run when all inputs have become available,
        and we can potentially add further inputs.
        The exp (ModelWithCheckpoints) includes a ref to scores_and_learning_rates
        which is only available when the training job finished,
        thus this is only run at the very end.

        Note that this is thus called multiple times,
        once scores_and_learning_rates becomes available,
        and then once the further recogs become available.
        However, only want to check for relevant checkpoints once.
        """
        if not self._update_checked_relevant_epochs and self.exp.scores_and_learning_rates.available():
            from datetime import datetime

            log_filename = tk.Path("update.log", self).get_path()
            os.makedirs(os.path.dirname(log_filename), exist_ok=True)
            with open(log_filename, "a") as log_stream:
                log_stream.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                log_stream.write(": get_relevant_epochs_from_training_learning_rate_scores\n")
                for epoch in get_relevant_epochs_from_training_learning_rate_scores(
                    model_dir=self.exp.model_dir,
                    model_name=self.exp.model_name,
                    scores_and_learning_rates=self.exp.scores_and_learning_rates,
                    n_best=self.train_scores_n_best,
                    log_stream=log_stream,
                ):
                    self._add_recog(epoch)
            self._update_checked_relevant_epochs = True
            self._make_output()

    def _add_recog(self, epoch: int):
        if epoch < self.first_epoch:
            return
        if epoch in self.exclude_epochs:
            return
        if epoch in self._in_checkpoints:
            return
        self._in_checkpoints[epoch] = self.exp.get_epoch(epoch).checkpoint

    def _make_output(self):
        in_checkpoints = [ckpt for epoch, ckpt in sorted(self._in_checkpoints.items())]
        in_avg_checkpoint = AverageTorchCheckpointsJob(
            checkpoints=in_checkpoints,
            returnn_python_exe=tools_paths.get_returnn_python_exe(),
            returnn_root=tools_paths.get_returnn_root(),
        ).out_checkpoint
        self._in_avg_checkpoint = in_avg_checkpoint
        self.add_input(in_avg_checkpoint.path)
        res = self.recog_and_score_func(in_avg_checkpoint)
        assert isinstance(res, ScoreResultCollection)
        self.add_input(res.main_measure_value)
        self.add_input(res.output)
        self._scores_output = res

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task("run", mini_task=True)

    def run(self):
        """run"""
        import shutil

        shutil.copy(self._scores_output.main_measure_value.get_path(), self.out_main_measure_value.get_path())
        shutil.copy(self._scores_output.output.get_path(), self.out_results.get_path())
        # We don't want to copy the checkpoint if we can avoid it.
        # Currently assume a hardlink is always possible.
        os.makedirs(os.path.dirname(self.out_avg_checkpoint.path.get_path()), exist_ok=True)
        os.link(self._in_avg_checkpoint.path.get_path(), self.out_avg_checkpoint.path.get_path())

        with open(self.out_merged_epochs_list.get_path(), "w") as f:
            f.write("[%s]\n" % ", ".join(str(ep) for ep in sorted(self._in_checkpoints.keys())))


class PriorCombineShardsJob(sisyphus.Job):
    def __init__(self, shard_prior_frames_outputs: list[tk.Path], shard_prior_probs_outputs: list[tk.Path]):
        self.shard_prior_frames_outputs = shard_prior_frames_outputs
        self.shard_prior_probs_outputs = shard_prior_probs_outputs
        self.out_comined_results = self.output_path("prior.txt")

    def tasks(self):
        """task"""
        yield sisyphus.Task("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        """run"""
        all_frames = 0
        all_logprobs = None
        for path_frame, path_prob in zip(self.shard_prior_frames_outputs, self.shard_prior_probs_outputs):
            all_frames += np.load(path_frame)
            logprobs = np.load(path_prob)
            if all_logprobs is None:
                all_logprobs = logprobs
            else:
                all_logprobs = np.logaddexp(all_logprobs, logprobs)

        print(f"All frames: {all_frames}, All probs: {all_logprobs}")

        log_average_probs = all_logprobs - np.log(all_frames)
        average_probs = np.exp(log_average_probs)
        print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
        with uopen(self.out_comined_results, "w") as f:
            np.savetxt(f, log_average_probs, delimiter=" ")
        print("Saved prior in prior.txt in +log space.")
