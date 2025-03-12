"""
Generic recog, for the model interfaces defined in model_interfaces.py
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Sequence, Collection, Iterator, Callable

import sisyphus
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
from .datasets.task import Task
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput, ScoreResultCollection
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, serialize_model_def
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint, ModelWithCheckpoints
from i6_experiments.users.zeyer.returnn.training import get_relevant_epochs_from_training_learning_rate_scores
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

from i6_experiments.users.mann.nn.util import DelayedCodeWrapper

import numpy as np

from .utils import PartialImportCustom, ReturnnConfigCustom
from .experiments.ctc_baseline.ctc import model_recog_lm, model_recog_flashlight, model_recog_lm_albert
from .experiments.language_models.n_gram import get_kenlm_n_gram, get_binary_lm
from .datasets.librispeech import get_bpe_lexicon, LibrispeechOggZip
from .scoring import ComputeWERJob, _score_recog

if TYPE_CHECKING:
    from returnn.tensor import TensorDict


def recog_training_exp(
    prefix_name: str,
    task: Task,
    model: ModelWithCheckpoints,
    recog_def: RecogDef,
    *,
    decoder_hyperparameters: Optional[dict] = None,
    save_pseudo_labels: Optional[tuple[dict, Optional[LibrispeechOggZip]]] = None,
    pseudo_nbest: int = 1,
    calculate_pseudo_label_scores: bool = True,
    search_config: Dict[str, Any] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    search_mem_rqmt: Union[int, float] = 6,
    exclude_epochs: Collection[int] = (),
    model_avg: bool = False,
    num_shards_recog: Optional[int] = None,
    num_shards_pseudo: Optional[int] = None,
    num_shards_prior: Optional[int] = None,
    is_last: bool = False,
    empirical_prior: Optional[tk.Path] = None,
    prior_from_max: bool = False,
    return_summary: bool = False,
    cache_manager: bool = True,
    check_train_scores_nbest: int = 2
) -> Optional[tk.Path]:
    """recog on all relevant epochs"""
    recog_and_score_func = _RecogAndScoreFunc(
        prefix_name,
        decoder_hyperparameters,
        task,
        model,
        recog_def,
        search_config=search_config,
        search_post_config=search_post_config,
        recog_post_proc_funcs=recog_post_proc_funcs,
        search_mem_rqmt=search_mem_rqmt,
        num_shards_recog=num_shards_recog,
        num_shards_prior=num_shards_prior,
        empirical_prior=empirical_prior,
        prior_from_max=prior_from_max,
        cache_manager=cache_manager
    )
    summarize_job = GetBestRecogTrainExp(
        exp=model,
        recog_and_score_func=recog_and_score_func,
        main_measure_lower_is_better=task.main_measure_type.lower_is_better,
        check_train_scores_n_best=check_train_scores_nbest,
        exclude_epochs=exclude_epochs,
    )
    summarize_job.add_alias(prefix_name + "/train-summarize")
    tk.register_output(prefix_name + "/recog_results_best", summarize_job.out_summary_json)
    tk.register_output(prefix_name + "/recog_results_all_epochs", summarize_job.out_results_all_epochs_json)
    if model_avg:
        model_avg_res_job = GetTorchAvgModelResult(
            exp=model, recog_and_score_func=recog_and_score_func, exclude_epochs=exclude_epochs
        )
        tk.register_output(prefix_name + "/recog_results_model_avg", model_avg_res_job.out_results)
        
    if return_summary:
        return summarize_job.out_summary_json
    
    # Create pseudo labels
    if save_pseudo_labels is not None:
        pseudo_labels_ds = save_pseudo_labels[0]
        dev_dataset = next(iter(pseudo_labels_ds.values()))
        task_pseudo_labels = Task(
            name="librispeech_pseudo_labels",
            train_dataset=task.train_dataset,
            train_epoch_split=task.train_epoch_split,
            dev_dataset=dev_dataset,
            eval_datasets=pseudo_labels_ds,
            main_measure_type=task.main_measure_type,
            main_measure_name=dev_dataset.get_main_name(),
            score_recog_output_func=task.score_recog_output_func,
            prior_dataset=task.prior_dataset,
            recog_post_proc_funcs=task.recog_post_proc_funcs,
        )
        
        pseudo_hyperparameters = decoder_hyperparameters.copy()
        if pseudo_nbest > 1 and not is_last:
            pseudo_hyperparameters["ps_nbest"] = pseudo_nbest
            
        pseudo_label_recog_func = _RecogAndScoreFunc(
            prefix_name + "/pseudo_labels",
            pseudo_hyperparameters,
            task_pseudo_labels,
            model,
            recog_def,
            save_pseudo_labels=pseudo_labels_ds,
            calculate_scores=calculate_pseudo_label_scores,
            search_config=search_config,
            search_post_config=search_post_config,
            recog_post_proc_funcs=recog_post_proc_funcs,
            search_mem_rqmt=search_mem_rqmt,
            num_shards_recog=num_shards_pseudo,
            num_shards_prior=num_shards_prior,
            empirical_prior=empirical_prior,
            prior_from_max=prior_from_max,
            cache_manager=cache_manager
        )
        
        extract_pseudo_labels_job = ExtractPseudoLabels(
            pseudo_recog_func=pseudo_label_recog_func,
            recog_input=summarize_job.out_summary_json,
            calculate_score=False,
        )
        extract_pseudo_labels_job.add_alias(prefix_name + "/pseudo_labels/extract")
        if is_last:
            tk.register_output(prefix_name + "/pseudo_labels/extract", extract_pseudo_labels_job.out_best_labels_path)
        
        # Calculate score for 100h
        if calculate_pseudo_label_scores:
            train_100_ds = save_pseudo_labels[1]
            task_score = Task(
                name="librispeech_score_train100",
                train_dataset=task.train_dataset,
                train_epoch_split=task.train_epoch_split,
                dev_dataset=train_100_ds,
                eval_datasets={"train-clean-100": train_100_ds},
                main_measure_type=task.main_measure_type,
                main_measure_name=train_100_ds.get_main_name(),
                score_recog_output_func=task.score_recog_output_func,
                prior_dataset=task.prior_dataset,
                recog_post_proc_funcs=task.recog_post_proc_funcs,
            )
            
            score_func = _RecogAndScoreFunc(
                prefix_name + "/pseudo_labels",
                decoder_hyperparameters,
                task_score,
                model,
                recog_def,
                save_pseudo_labels=None,
                calculate_scores=True,
                search_config=search_config,
                search_post_config=search_post_config,
                recog_post_proc_funcs=recog_post_proc_funcs,
                search_mem_rqmt=search_mem_rqmt,
                num_shards_recog=num_shards_pseudo,
                num_shards_prior=num_shards_prior,
                register_output=False,
                empirical_prior=empirical_prior,
                prior_from_max=prior_from_max,
                cache_manager=cache_manager
            )
            
            score_job = GetScoreJob(score_func, summarize_job.out_summary_json)
            tk.register_output(prefix_name + "/pseudo_labels/score100", score_job.out_score)
    
        return extract_pseudo_labels_job.out_best_labels_path
    return None


class _RecogAndScoreFunc:
    def __init__(
        self,
        prefix_name: str,
        decoder_hyperparameters: dict,
        task: Task,
        model: ModelWithCheckpoints,
        recog_def: RecogDef,
        *,
        save_pseudo_labels: Optional[dict] = None,
        calculate_scores: bool = True,
        search_config: Optional[Dict[str, Any]] = None,
        search_post_config: Optional[Dict[str, Any]] = None,
        recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
        search_mem_rqmt: Union[int, float] = 6,
        num_shards_recog: Optional[int] = None,
        num_shards_prior: Optional[int] = None,
        register_output: bool = True,
        empirical_prior: Optional[tk.Path] = None,
        prior_from_max: bool = False,
        cache_manager: bool = True
    ):
        # Note: When something is added here, remember to handle it in _sis_hash.
        self.prefix_name = prefix_name
        self.decoder_hyperparameters = decoder_hyperparameters
        self.task = task
        self.model = model
        self.recog_def = recog_def
        self.search_config = search_config
        self.search_post_config = search_post_config
        self.recog_post_proc_funcs = recog_post_proc_funcs
        self.search_mem_rqmt = search_mem_rqmt
        self.save_pseudo_labels = save_pseudo_labels
        self.calculate_scores = calculate_scores
        self.num_shards_recog = num_shards_recog
        self.num_shards_prior = num_shards_prior
        self.register_output = register_output
        self.empirical_prior = empirical_prior
        self.prior_from_max = prior_from_max
        self.cache_manager = cache_manager

    def __call__(self, epoch_or_ckpt: Union[int, PtCheckpoint]) -> tuple[ScoreResultCollection, tk.Path]:
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
                    num_shards=self.num_shards_prior,
                    prior_from_max=self.prior_from_max,
                    cache_manager=self.cache_manager
                )
                if isinstance(epoch_or_ckpt, int):
                    tk.register_output(self.prefix_name + f"/recog_results_per_epoch/{epoch_or_ckpt:03}/prior.txt", prior_path)
        else:
            prior_path = None

        res, label_paths = recog_model(
            self.task,
            model_with_checkpoint,
            self.recog_def,
            self.decoder_hyperparameters,
            prior_path,
            save_pseudo_labels=True if self.save_pseudo_labels else False,
            calculate_scores=self.calculate_scores,
            config=self.search_config,
            search_post_config=self.search_post_config,
            recog_post_proc_funcs=self.recog_post_proc_funcs,
            search_mem_rqmt=self.search_mem_rqmt,
            name=self.prefix_name + f"/search/{epoch_or_ckpt:03}",
            num_shards=self.num_shards_recog,
            cache_manager=self.cache_manager
        )
        if self.calculate_scores and isinstance(epoch_or_ckpt, int) and self.register_output:
            tk.register_output(self.prefix_name + f"/recog_results_per_epoch/{epoch_or_ckpt:03}/score", res.output)
        return res, label_paths

    def _sis_hash(self) -> bytes:
        from sisyphus.hash import sis_hash_helper

        d = self.__dict__.copy()
        # Remove irrelevant stuff which should not affect the hash.
        del d["prefix_name"]
        del d["search_post_config"]
        del d["search_mem_rqmt"]
        del d["num_shards_prior"]
        del d["num_shards_recog"]
        del d["register_output"]
        del d["cache_manager"]
        if not self.search_config:
            del d["search_config"]  # compat
        else:
            conf = copy.deepcopy(d["search_config"])
            if "preload_from_files" in conf:
                for v in conf["preload_from_files"].values():
                    if "filename" in v and isinstance(v["filename"], DelayedCodeWrapper):
                        v["filename"] = v["filename"].args[0]
            d["search_config"] = conf
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
    decoder_hyperparameters: dict,
    prior_path: tk.Path,
    *,
    save_pseudo_labels: bool = False,
    calculate_scores: bool = True,
    config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    search_mem_rqmt: Union[int, float] = 6,
    search_rqmt: Optional[Dict[str, Any]] = None,
    dev_sets: Optional[Collection[str]] = None,
    name: Optional[str] = None,
    num_shards: Optional[int] = None,
    cache_manager: bool = True
) -> tuple[ScoreResultCollection, tk.Path]:
    """recog"""
    if dev_sets is not None:
        assert all(k in task.eval_datasets for k in dev_sets)
    outputs = {}
    recog_paths = {}
    for dataset_name, dataset in task.eval_datasets.items():
        if dev_sets is not None:
            if dataset_name not in dev_sets:
                continue
        recog_out, beam_recog_out = search_dataset(
            decoder_hyperparameters=decoder_hyperparameters,
            dataset=dataset,
            model=model,
            recog_def=recog_def,
            prior_path=prior_path,
            save_pseudo_labels=save_pseudo_labels,
            config=config,
            search_post_config=search_post_config,
            search_mem_rqmt=search_mem_rqmt,
            search_rqmt=search_rqmt,
            search_alias_name=f"{name}/{dataset_name}" if name else None,
            recog_post_proc_funcs=list(recog_post_proc_funcs) + list(task.recog_post_proc_funcs),
            num_shards=num_shards,
            cache_manager=cache_manager,
            # count_repeated_path=f"{name.replace('/search/', '/repeated_count/')}" if dataset_name == "dev-other" else None
        )
        if calculate_scores:
            if dataset_name.startswith("train"):
                score_out = _score_recog(dataset, recog_out, alias_name=name + f"/sclite/{dataset_name}")
                # score_out = ComputeWERJob(recog_out.output, corpus_text_dict).out_wer
            else:
                score_out = task.score_recog_output_func(dataset, recog_out)
            outputs[dataset_name] = score_out
        if save_pseudo_labels:
            if "ps_nbest" in decoder_hyperparameters and decoder_hyperparameters["ps_nbest"] > 1:
                recog_paths[dataset_name] = beam_recog_out
            else:
                recog_paths[dataset_name] = recog_out.output
    return task.collect_score_results_func(outputs) if calculate_scores else None, recog_paths if save_pseudo_labels else None


def compute_prior(
    *,
    dataset: DatasetConfig,
    model: ModelWithCheckpoint,
    mem_rqmt: Union[int, float] = 8,
    prior_alias_name: Optional[str] = None,
    num_shards: Optional[int] = None,
    prior_from_max: bool = False,
    cache_manager: bool = True
) -> tk.Path:
    if num_shards is not None:
        prior_frames_res = []
        prior_probs_res = []
        for i in range(num_shards):
            shard_prior_sum_job = ReturnnForwardJobV2(
                model_checkpoint=DelayedCodeWrapper("cf('{}')", model.checkpoint) if cache_manager else model.checkpoint,
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
            model_checkpoint=DelayedCodeWrapper("cf('{}')", model.checkpoint) if cache_manager else model.checkpoint,
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

def search_dataset(
    *,
    decoder_hyperparameters: dict,
    dataset: DatasetConfig,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    prior_path: tk.Path,
    save_pseudo_labels: bool = False,
    config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    search_mem_rqmt: Union[int, float] = 8,
    search_rqmt: Optional[Dict[str, Any]] = None,
    search_alias_name: Optional[str] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    num_shards: Optional[int] = None,
    cache_manager: bool = True,
    count_repeated_path: Optional[str] = None
) -> tuple[RecogOutput, tk.Path]:
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
    :return: :class:`RecogOutput`, single best hyp (if there was a beam, we already took the best one)
        over the dataset
    """
    env_updates = None
    if (config and config.get("__env_updates")) or (search_post_config and search_post_config.get("__env_updates")):
        env_updates = (config and config.pop("__env_updates", None)) or (
            search_post_config and search_post_config.pop("__env_updates", None)
        )
    if save_pseudo_labels:
        time_rqmt = 16.0
        cpu_rqmt = 4
        if search_mem_rqmt < 8:
            search_mem_rqmt = 8
    else:
        time_rqmt = 16.0
        cpu_rqmt = 4
        if search_mem_rqmt < 8:
            search_mem_rqmt = 24
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
            device="cpu",
            time_rqmt=time_rqmt,
            mem_rqmt=search_mem_rqmt,
            cpu_rqmt=cpu_rqmt,
        )
        res = search_job.out_search_file
    else:
        out_files = [_v2_forward_out_filename]
        if config and config.get("__recog_def_ext", False):
            out_files.append(_v2_forward_ext_out_filename)
        if num_shards is not None:
            shard_search_res = []
            for i in range(num_shards):
                shard_search_job = ReturnnForwardJobV2(
                    model_checkpoint=DelayedCodeWrapper("cf('{}')", model.checkpoint) if cache_manager else model.checkpoint,
                    returnn_config=search_config_v2(
                        dataset, model.definition, recog_def, decoder_hyperparameters, prior_path, config=config, post_config=search_post_config, shard_index=i, num_shards=num_shards
                    ),
                    output_files=out_files,
                    returnn_python_exe=tools_paths.get_returnn_python_exe(),
                    returnn_root=tools_paths.get_returnn_root(),
                    device="cpu",
                    time_rqmt=time_rqmt,
                    mem_rqmt=search_mem_rqmt,
                    cpu_rqmt=cpu_rqmt,
                )
                res = shard_search_job.out_files[_v2_forward_out_filename]
                if search_rqmt:
                    shard_search_job.rqmt.update(search_rqmt)
                if env_updates:
                    for k, v in env_updates.items():
                        shard_search_job.set_env(k, v)
                shard_search_res.append(res)
            search_job = SearchCombineShardsJob(shard_search_res)
            res = search_job.out_comined_results
        else:
            search_job = ReturnnForwardJobV2(
                model_checkpoint=DelayedCodeWrapper("cf('{}')", model.checkpoint) if cache_manager else model.checkpoint,
                returnn_config=search_config_v2(
                    dataset, model.definition, recog_def, decoder_hyperparameters, prior_path, config=config, post_config=search_post_config
                ),
                output_files=out_files,
                returnn_python_exe=tools_paths.get_returnn_python_exe(),
                returnn_root=tools_paths.get_returnn_root(),
                device="cpu",
                time_rqmt=time_rqmt,
                mem_rqmt=search_mem_rqmt,
                cpu_rqmt=cpu_rqmt,
            )
            res = search_job.out_files[_v2_forward_out_filename]
    if num_shards is None:
        if search_rqmt:
            search_job.rqmt.update(search_rqmt)
        if env_updates:
            for k, v in env_updates.items():
                search_job.set_env(k, v)
    if search_alias_name:
        search_job.add_alias(search_alias_name)
        
    use_lexicon = decoder_hyperparameters.get("use_lexicon", False) # if we have lexicon we already have the full words
    if not use_lexicon:
        if recog_def.output_blank_label:
            if recog_def is not model_recog_lm or "greedy" in decoder_hyperparameters:
                # Also assume we should collapse repeated labels first.
                res = SearchCollapseRepeatedLabelsJob(res, output_gzip=True).out_search_results
            res = SearchRemoveLabelJob(res, remove_label=recog_def.output_blank_label, output_gzip=True).out_search_results
        for f in recog_post_proc_funcs:  # for example BPE to words
            res = f(RecogOutput(output=res)).output
        if recog_def is model_recog_flashlight or recog_def is model_recog_lm_albert:
            from i6_core.returnn.search import SearchOutputRawReplaceJob
            res = SearchOutputRawReplaceJob(res, [("@@", "")], output_gzip=True).out_search_results
        if count_repeated_path:
            cnt = CountRepeatedLabelsJob(res).out_count_results
            tk.register_output(count_repeated_path, cnt)
    beam_res = res
    if recog_def.output_with_beam:
        # Don't join scores here (SearchBeamJoinScoresJob).
        #   It's not clear whether this is helpful in general.
        #   As our beam sizes are very small, this might boost some hyps too much.
        res = SearchTakeBestJob(res, output_gzip=True).out_best_search_results
    return RecogOutput(output=res), beam_res


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
        batch_size= 500 * 16000,
        max_seqs= 240,
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
    decoder_hyperparameters: dict,
    prior_path: tk.Path,
    *,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> ReturnnConfig:
    """
    Create config for search.

    v2: Use any backend (usually PyTorch) and the new API (get_model, forward_step).

    TODO should use sth like unhashed_package_root (https://github.com/rwth-i6/i6_experiments/pull/157)
    """
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config

    if num_shards is not None and shard_index is not None:
        forward_data = dataset.get_sharded_main_dataset(shard_index, num_shards)
    else:
        forward_data = dataset.get_main_dataset()
    
    returnn_recog_config_dict = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        forward_data=forward_data,
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
    
    args_cached = {}
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
        
        if "lm_order" in decoder_hyperparameters:
            lm_name = decoder_hyperparameters["lm_order"]
            if isinstance(lm_name, int):
                lm = get_binary_lm(get_kenlm_n_gram(vocab = dataset.vocab, N_order = int(lm_name)))
            elif lm_name.startswith("ffnn"):
                lm = None
            else:
                raise NotImplementedError(f"Unknown lm_name {lm_name}")
        else:
            lm = get_binary_lm(get_arpa_lm_dict()["4gram"])
            
        args = {"arpa_4gram_lm": lm, "lexicon": lexicon, "hyperparameters": decoder_hyperparameters}
        if lexicon:
            args_cached["lexicon"] = DelayedCodeWrapper("cf('{}')", lexicon.get_path())
        if lm:
            args_cached["arpa_4gram_lm"] = DelayedCodeWrapper("cf('{}')", lm.get_path())
    
        if prior_path:
            args["prior_file"] = prior_path
            args_cached["prior_file"] = DelayedCodeWrapper("cf('{}')", prior_path.get_path())
    elif recog_def is model_recog_flashlight or recog_def is model_recog_lm_albert:
        args = {"hyperparameters": decoder_hyperparameters}
        if prior_path:
            args["prior_file"] = prior_path
            args_cached["prior_file"] = DelayedCodeWrapper("cf('{}')", prior_path.get_path())
        if recog_def is model_recog_lm_albert or recog_def is model_recog_flashlight:
            args["version"] = 9
    else:
        args = {}

    returnn_recog_config = ReturnnConfigCustom(
        config=returnn_recog_config_dict,
        python_prolog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                ]
            )
        ],
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
                    *serialize_model_def(model_def),
                    serialization.Import(_returnn_v2_get_model, import_as="get_model"),
                    PartialImportCustom(
                        code_object_path=recog_def,
                        unhashed_package_root=None,
                        hashed_arguments=args,
                        unhashed_arguments=args_cached,
                        import_as="_recog_def",
                        ignore_import_as_for_hash=True),
                    serialization.Import(_returnn_v2_forward_step, import_as="forward_step"),
                    serialization.Import(_returnn_v2_get_forward_callback, import_as="forward_callback"),
                    serialization.ExplicitHash(
                        {
                            # Increase the version whenever some incompatible change is made in this recog() function,
                            # which influences the outcome, but would otherwise not influence the hash.
                            "version": 13,
                        }
                    ),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
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
    if recog_def.func is model_recog_lm_albert or recog_def.func is model_recog_flashlight:
        seq_tags = extern_data["seq_tag"]
        recog_out = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim, seq_tags=seq_tags, **extra)
    else:
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
        res, _ = recog_and_score_func(last_fixed_epoch)
        assert isinstance(res, ScoreResultCollection)
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
        res, _ = self.recog_and_score_func(epoch)
        assert isinstance(res, ScoreResultCollection)
        self.add_input(res.main_measure_value)
        self.add_input(res.output)
        self._scores_outputs[epoch] = res

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

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
    ):
        self.scores = scores
        self.tune_values = tune_values
        self.out_best_tune = self.output_path("best_tune.json")

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        """run"""
        import json

        best_score_idx = -1
        best_score_val = 1000000.0
        for i in range(len(self.scores)):
            d = eval(uopen(self.scores[i], "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            assert isinstance(d, dict), "Has to be a dict containing the best score."
            
            if d["best_scores"]["dev-other"] < best_score_val:
                best_score_idx = i
                best_score_val = d["best_scores"]["dev-other"]
                
        best_tune = self.tune_values[best_score_idx]
        
        res = {"best_tune": best_tune}
        with open(self.out_best_tune.get_path(), "w") as f:
            f.write(json.dumps(res))
            f.write("\n")
                
class ExtractPseudoLabels(sisyphus.Job):
    def __init__(
        self,
        pseudo_recog_func: Optional[Callable[[int], ScoreResultCollection]],
        recog_input: tk.Path,
        calculate_score: bool,
    ):
        super(ExtractPseudoLabels, self).__init__()
        self.pseudo_recog_func = pseudo_recog_func
        self.recog_input = recog_input
        self.out_best_labels_path = self.output_path("best_labels_path.json")
        self._recog_score = None
        self._recog_label_paths = None
        self.calculate_score = calculate_score

    def update(self):
        if self._recog_label_paths:
            return
        
        d = eval(uopen(self.recog_input, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "Has to be a dict containing the best epoch during scoring."
        
        self.best_epoch = d["best_epoch"]
        
        res, label_paths = self.pseudo_recog_func(self.best_epoch)
        assert isinstance(label_paths, dict)
        for k in label_paths.keys():
            self.add_input(label_paths[k])
        self._recog_label_paths = label_paths
        if self.calculate_score:
            assert isinstance(res, ScoreResultCollection)
            self.add_input(res.output)
            self._recog_score = res

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        """run"""
        import json
        
        if self.calculate_score:
            score = json.load(open(self._recog_score.output.get_path()))
        else:
            score = -1
        for k in self._recog_label_paths.keys():
            self._recog_label_paths[k] = self._recog_label_paths[k].get_path()
        
        label_path_dict = {"path": self._recog_label_paths, "score": score, "epoch": self.best_epoch}
        with open(self.out_best_labels_path.get_path(), "w") as f:
            f.write(json.dumps(label_path_dict))
            f.write("\n")
            
    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        # Extend the default hash() function.
        d = parsed_args.copy()
        
        # Add this to the hash to change the hash.
        d["_nr"] = 1
        return sis_tools.sis_hash(d)
    
class GetScoreJob(sisyphus.Job):
    def __init__(
        self,
        score_func: Optional[Callable[[int], ScoreResultCollection]],
        recog_input: tk.Path,
    ):
        super(GetScoreJob, self).__init__()
        self.score_func = score_func
        self.recog_input = recog_input
        self._recog_score = None
        self.out_score = self.output_path("score")
        
    def update(self):
        if self._recog_score:
            return
        
        d = eval(uopen(self.recog_input, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "Has to be a dict containing the best epoch during scoring."
        
        self.best_epoch = d["best_epoch"]
        
        res, _ = self.score_func(self.best_epoch)
        assert isinstance(res, ScoreResultCollection)
        self.add_input(res.output)
        self._recog_score = res

    def tasks(self) -> Iterator[sisyphus.Task]:
        yield sisyphus.Task("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        import json
        
        score = json.load(open(self._recog_score.output.get_path()))
        with open(self.out_score.get_path(), "w") as f:
            f.write(json.dumps({"score": score, "epoch": self.best_epoch}))
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
        res, _ = self.recog_and_score_func(in_avg_checkpoint)
        assert isinstance(res, ScoreResultCollection)
        self.add_input(res.main_measure_value)
        self.add_input(res.output)
        self._scores_output = res

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

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
            
class SearchCombineShardsJob(sisyphus.Job):

    def __init__(self, shard_search_outputs: list[tk.Path]):
        self.shard_search_outputs = shard_search_outputs
        self.out_comined_results = self.output_path(_v2_forward_out_filename)

    def tasks(self):
        """task"""
        yield sisyphus.Task("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        """run"""
        res_dict = {}
        for path in self.shard_search_outputs:
            d = eval(uopen(path, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            assert isinstance(d, dict)  # seq_tag -> bpe string
            res_dict.update(d)
        with uopen(self.out_comined_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in res_dict.items():
                assert isinstance(entry, list)
                out.write("%r: %r,\n" % (seq_tag, entry))
            out.write("}\n")
            
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
        
class CountRepeatedLabelsJob(sisyphus.Job):

    def __init__(self, search_py_output: tk.Path):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param output_gzip: gzip the output
        """
        self.search_py_output = search_py_output
        self.out_count_results = self.output_path("count_results.py")

    def tasks(self):
        """task"""
        yield sisyphus.Task("run", mini_task=True)

    def run(self):
        """run"""
        d = eval(uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_count_results.get_path())
        with uopen(self.out_count_results, "wt") as out:
            out.write("{\n")
            num_repeated = 0
            num_total = 0
            nbest = 0
            for seq_tag, entry in d.items():
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    nbest=len(entry)
                    for score, text in entry:
                        prev = len(text.split(" "))
                        num_total += prev
                        new = len(self._filter(text).split(" "))
                        num_repeated += (prev - new)
                else:
                    nbest=1
                    prev = len(entry.split(" "))
                    num_total += prev
                    new = len(self._filter(entry).split(" "))
                    num_repeated += (prev - new)
            out.write("%r: %r,\n" % ("Repeated", num_repeated))
            out.write("%r: %r,\n" % ("Total", num_total))
            out.write("%r: %r,\n" % ("Percentage", num_repeated / num_total))
            out.write("%r: %r,\n" % ("Nbest", nbest))
            out.write("}\n")

    def _filter(self, txt: str) -> str:
        tokens = txt.split(" ")
        tokens = [t1 for (t1, t2) in zip(tokens, [None] + tokens) if t1 != t2]
        return " ".join(tokens)