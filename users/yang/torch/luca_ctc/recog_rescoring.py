"""
Generic recog, for the model interfaces defined in model_interfaces.py
"""

from __future__ import annotations

import os
import dataclasses
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Sequence, Collection, Iterator, Callable, List
import copy

import sisyphus
from sisyphus import tk
from sisyphus import tools as sis_tools
from i6_core.util import instanciate_delayed

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
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput, ScoreResultCollection
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, serialize_model_def
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint, ModelWithCheckpoints
from i6_experiments.users.zeyer.returnn.training import get_relevant_epochs_from_training_learning_rate_scores

if TYPE_CHECKING:
    from returnn.tensor import TensorDict



_v2_out_hyp_hdf = "out_hyps.hdf"
_v2_out_lens_hdf = "out_lens.hdf"
_v2_out_scores_hdf = "out_scores.hdf"
_v2_out_packed_batch_sizes = "out_packed_batch_sizes.hdf"
def recog_training_exp(
    prefix_name: str,
    task: Task,
    model: ModelWithCheckpoints,
    recog_def: RecogDef,
    *,
    search_config: Dict[str, Any] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    search_mem_rqmt: Union[int, float] = 6,
    exclude_epochs: Collection[int] = (),
    model_avg: bool = False,
    train_exp_name: Optional[str] = None,
    dev_sets: Optional[List[str]] = None,
    recompute_prior: bool = False,
    prior_config: Optional[dict] = None,
    prior_task: Optional[Task] = None, # where to compute the task
    search_rqmt = {"time": 6},
    merge_contraction: bool = False,
):
    """recog on all relevant epochs"""
    recog_and_score_func = _RecogAndScoreFunc(
        prefix_name,
        task,
        model,
        recog_def,
        search_config=search_config,
        search_post_config=search_post_config,
        recog_post_proc_funcs=recog_post_proc_funcs,
        search_mem_rqmt=search_mem_rqmt,
        train_exp_name=train_exp_name,
        dev_sets=dev_sets,
        recompute_prior=recompute_prior,
        prior_config=prior_config,
        prior_task=prior_task,
        search_rqmt=search_rqmt,
        merge_contraction=merge_contraction,
    )
    summarize_job = GetBestRecogTrainExp(
        exp=model,
        recog_and_score_func=recog_and_score_func,
        main_measure_lower_is_better=task.main_measure_type.lower_is_better,
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


class _RecogAndScoreFunc:
    def __init__(
        self,
        prefix_name: str,
        task: Task,
        model: ModelWithCheckpoints,
        recog_def: RecogDef,
        *,
        search_config: Optional[Dict[str, Any]] = None,
        search_post_config: Optional[Dict[str, Any]] = None,
        recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
        search_mem_rqmt: Union[int, float] = 6,
        train_exp_name: Optional[str] = None,
        dev_sets: Optional[List[str]] = None,
        recompute_prior: bool = False,
        prior_config: Optional[dict] = None,
        prior_task = None,
        search_rqmt = {"time": 5},
        merge_contraction: bool = False,
    ):
        # Note: When something is added here, remember to handle it in _sis_hash.
        self.prefix_name = prefix_name
        self.task = task
        self.model = model
        self.recog_def = recog_def
        self.search_config = search_config
        self.search_post_config = search_post_config
        self.recog_post_proc_funcs = recog_post_proc_funcs
        self.search_mem_rqmt = search_mem_rqmt
        self.train_exp_name = train_exp_name
        self.dev_sets = dev_sets
        self.recompute_prior = recompute_prior
        self.prior_config = prior_config
        self.prior_task = prior_task
        self.search_rqmt = search_rqmt
        self.merge_contraction = merge_contraction

    def __call__(self, epoch_or_ckpt: Union[int, PtCheckpoint]) -> ScoreResultCollection:
        if isinstance(epoch_or_ckpt, int):
            model_with_checkpoint = self.model.get_epoch(epoch_or_ckpt)
        elif isinstance(epoch_or_ckpt, PtCheckpoint):
            model_with_checkpoint = ModelWithCheckpoint(definition=self.model.definition, checkpoint=epoch_or_ckpt)
        else:
            raise TypeError(f"{self} unexpected type {type(epoch_or_ckpt)}")
        res = recog_model(
            self.task,
            model_with_checkpoint,
            self.recog_def,
            config=self.search_config,
            search_post_config=self.search_post_config,
            recog_post_proc_funcs=self.recog_post_proc_funcs,
            search_mem_rqmt=self.search_mem_rqmt,
            # name=self.prefix_name,
            train_exp_name=self.train_exp_name,
            dev_sets=self.dev_sets,
            recompute_prior=self.recompute_prior,
            prior_config=self.prior_config,
            prior_task = self.prior_task,
            name=self.prefix_name,
            epoch=epoch_or_ckpt,
            search_rqmt=self.search_rqmt,
            merge_contraction=self.merge_contraction,
        )
        if isinstance(epoch_or_ckpt, int):
            tk.register_output(self.prefix_name + f"/recog_results_per_epoch/{epoch_or_ckpt:03}", res.output)
        return res

    def _sis_hash(self) -> bytes:
        from sisyphus.hash import sis_hash_helper

        d = self.__dict__.copy()
        # Remove irrelevant stuff which should not affect the hash.
        del d["prefix_name"]
        del d["search_post_config"]
        del d["search_mem_rqmt"]
        if not self.search_config:
            del d["search_config"]  # compat
        if not self.recog_post_proc_funcs:
            del d["recog_post_proc_funcs"]  # compat
        # Not the whole task object is relevant but only some minimal parts.
        task = d.pop("task")
        assert isinstance(task, Task)
        for k in ["train_dataset", "train_epoch_split"]:  # for hash relevant parts
            d[f"task.{k}"] = getattr(task, k)
        d["class"] = "_RecogAndScoreFunc"  # some identifier; not full qualname to allow for moving the class
        return sis_hash_helper(d)


def recog_model(
    task: Task,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    *,
    config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    search_mem_rqmt: Union[int, float] = 6,
    merge_contraction: bool = False,
    search_rqmt: Optional[Dict[str, Any]] = {"time": 5},
    dev_sets: Optional[Collection[str]] = None,
    name: Optional[str] = None,
    train_exp_name: Optional[str] = None,
    recompute_prior: bool = False,
    prior_config: Optional[dict] = None,
    prior_task = None,
    epoch: Optional[int] = None,
) -> ScoreResultCollection:
    """recog"""
    if dev_sets is not None:
        assert all(k in task.eval_datasets for k in dev_sets)
    outputs = {}
    for dataset_name, dataset in task.eval_datasets.items():
        if dev_sets is not None:
            if dataset_name not in dev_sets:
                continue
        if recompute_prior:
            from i6_experiments.users.phan.prior.prior_config import compute_prior_job, _prior_out_filename
            from i6_experiments.users.phan.prior.model_forward_prior import model_forward_prior
            assert prior_config is not None
            prior_job = compute_prior_job(
                task=prior_task,
                model=model,
                recog_def=model_forward_prior,
                config=prior_config,
                search_rqmt={"time": 12},
            )
            prior_job.set_vis_name(f"Compute prior job, {train_exp_name}, {os.path.split(model.checkpoint.__repr__())[-1][:-1]}")
            config["search_args"]["prior_file"] = prior_job.out_files[_prior_out_filename]
        search_alias_name = None
        if name:
            search_alias_name = f"{name}/search/{dataset_name}"
            if epoch:
                search_alias_name += f"/{epoch}"
        if config.get("max_seqs", None) == 1:
            search_rqmt["time"] = 24
        recog_out = search_dataset(
            dataset=dataset,
            model=model,
            recog_def=recog_def,
            config=config,
            search_post_config=search_post_config,
            search_mem_rqmt=search_mem_rqmt,
            search_rqmt=search_rqmt,
            search_alias_name=search_alias_name,
            recog_post_proc_funcs=list(recog_post_proc_funcs) + list(task.recog_post_proc_funcs),
            dataset_name=dataset_name,
            train_exp_name=train_exp_name,
            merge_contraction=merge_contraction,
        )
        score_out = task.score_recog_output_func(dataset, recog_out, merge_contraction=merge_contraction)
        outputs[dataset_name] = score_out
    return task.collect_score_results_func(outputs)


def recog_first_pass_hdf(
    task: Task,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    *,
    config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    search_mem_rqmt: Union[int, float] = 6,
    merge_contraction: bool = False,
    search_rqmt: Optional[Dict[str, Any]] = {"time": 5},
    dev_sets: Optional[Collection[str]] = None,
    name: Optional[str] = None,
    train_exp_name: Optional[str] = None,
    recompute_prior: bool = False,
    prior_config: Optional[dict] = None,
    prior_task = None,
    epoch: Optional[int] = None,
    hdf_output='v1',
):
    """recog"""
    if dev_sets is not None:
        assert all(k in task.eval_datasets for k in dev_sets)
    outputs = {}
    for dataset_name, dataset in task.eval_datasets.items():
        if dev_sets is not None:
            if dataset_name not in dev_sets:
                continue
        if recompute_prior:
            from i6_experiments.users.phan.prior.prior_config import compute_prior_job, _prior_out_filename
            from i6_experiments.users.phan.prior.model_forward_prior import model_forward_prior
            assert prior_config is not None
            prior_job = compute_prior_job(
                task=prior_task,
                model=model,
                recog_def=model_forward_prior,
                config=prior_config,
                search_rqmt={"time": 12},
            )
            prior_job.set_vis_name(f"Compute prior job, {train_exp_name}, {os.path.split(model.checkpoint.__repr__())[-1][:-1]}")
            config["search_args"]["prior_file"] = prior_job.out_files[_prior_out_filename]
        search_alias_name = None
        if name:
            search_alias_name = f"{name}/search/{dataset_name}"
            if epoch:
                search_alias_name += f"/{epoch}"
        if config.get("max_seqs", None) == 1:
            search_rqmt["time"] = 24
        out_hdf = search_dataset(
            dataset=dataset,
            model=model,
            recog_def=recog_def,
            config=config,
            search_post_config=search_post_config,
            search_mem_rqmt=search_mem_rqmt,
            search_rqmt=search_rqmt,
            search_alias_name=search_alias_name,
            recog_post_proc_funcs=list(recog_post_proc_funcs) + list(task.recog_post_proc_funcs),
            dataset_name=dataset_name,
            train_exp_name=train_exp_name,
            merge_contraction=merge_contraction,
            output_for_rescoring=hdf_output
        )
        outputs[dataset_name] = out_hdf
    return outputs







def search_dataset(
    *,
    dataset: DatasetConfig,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    search_mem_rqmt: Union[int, float] = 6,
    merge_contraction: bool = False,
    search_rqmt: Optional[Dict[str, Any]] = None,
    search_alias_name: Optional[str] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    dataset_name: Optional[str] = None,
    train_exp_name: Optional[str] = None,
    output_for_rescoring=None,
):
    """
    recog on the specific dataset, no post-processing
    """
    env_updates = None
    if (config and config.get("__env_updates")) or (search_post_config and search_post_config.get("__env_updates")):
        env_updates = (config and config.pop("__env_updates", None)) or (
            search_post_config and search_post_config.pop("__env_updates", None)
        )
    assert getattr(model.definition, "backend", None) is not None, "only support torch backend"
    if not output_for_rescoring:
        out_files = [_v2_forward_out_filename]

        if config and config.get("__recog_def_ext", False):
            out_files.append(_v2_forward_ext_out_filename)
        search_job = ReturnnForwardJobV2(
            model_checkpoint=model.checkpoint,
            returnn_config=search_config_v2(
                dataset, model.definition, recog_def, config=config, post_config=search_post_config,
                hdf_output=output_for_rescoring,
            ),
            output_files=out_files,
            returnn_python_exe=tools_paths.get_returnn_python_exe(),
            returnn_root=tools_paths.get_returnn_root(),
            mem_rqmt=search_mem_rqmt,
            time_rqmt=7,
        )
        search_job.set_vis_name(f"{train_exp_name}, {os.path.split(model.checkpoint.__repr__())[-1][:-1]}, {dataset_name}, {config.get('search_args', '') if config else ''}")
        res = search_job.out_files[_v2_forward_out_filename]
    elif output_for_rescoring == 'v1':
        out_files = [_v2_out_hyp_hdf, _v2_out_lens_hdf, _v2_out_scores_hdf, _v2_out_packed_batch_sizes]
        search_job = ReturnnForwardJobV2(
            model_checkpoint=model.checkpoint,
            returnn_config=search_config_v2(
                dataset, model.definition, recog_def, config=config, post_config=search_post_config,
                hdf_output=output_for_rescoring,
            ),
            output_files=out_files,
            returnn_python_exe=tools_paths.get_returnn_python_exe(),
            returnn_root=tools_paths.get_returnn_root(),
            mem_rqmt=search_mem_rqmt,
            time_rqmt=7,
        )
        search_job.set_vis_name(
            f"{train_exp_name}, {os.path.split(model.checkpoint.__repr__())[-1][:-1]}, {dataset_name}, {config.get('search_args', '') if config else ''}")
        out_hyps_hdf = search_job.out_files[_v2_out_hyp_hdf]
        out_lens_hdf = search_job.out_files[_v2_out_lens_hdf]
        out_scores_hdf = search_job.out_files[_v2_out_scores_hdf]
        out_packed_batch_sizes = search_job.out_files[_v2_out_packed_batch_sizes]
    elif output_for_rescoring == 'v2':

        out_files = [_v2_out_hyp_hdf]
        search_job = ReturnnForwardJobV2(
            model_checkpoint=model.checkpoint,
            returnn_config=search_config_v2(
                dataset, model.definition, recog_def, config=config, post_config=search_post_config,
                hdf_output=output_for_rescoring,
            ),
            output_files=out_files,
            returnn_python_exe=tools_paths.get_returnn_python_exe(),
            returnn_root=tools_paths.get_returnn_root(),
            mem_rqmt=search_mem_rqmt,
            time_rqmt=7,
        )
        search_job.set_vis_name(
            f"{train_exp_name}, {os.path.split(model.checkpoint.__repr__())[-1][:-1]}, {dataset_name}, {config.get('search_args', '') if config else ''}")
        out_hyps_hdf = search_job.out_files[_v2_out_hyp_hdf]
    else:
        NotImplementedError


    if search_rqmt:
        search_job.rqmt.update(search_rqmt)
    if env_updates:
        for k, v in env_updates.items():
            search_job.set_env(k, v)
    if search_alias_name:
        search_job.add_alias(search_alias_name)
    # if recog_def.output_blank_label:
    #     # collapes of the repeated label is handled in the rec function
    #     #res = SearchCollapseRepeatedLabelsJob(res, output_gzip=True).out_search_results
    #     res = SearchRemoveLabelJob(res, remove_label=recog_def.output_blank_label, output_gzip=True).out_search_results
    # for f in recog_post_proc_funcs:  # for example BPE to words
    #     res = f(RecogOutput(output=res)).output
    # if recog_def.output_with_beam:
    #     # Don't join scores here (SearchBeamJoinScoresJob).
    #     #   It's not clear whether this is helpful in general.
    #     #   As our beam sizes are very small, this might boost some hyps too much.
    #     res = SearchTakeBestJob(res, output_gzip=True).out_best_search_results
    # if merge_contraction:
    #     from i6_experiments.users.yang.torch.ctc_ilm_kd.datasets.librispeech_tedlium2 import MergeContractionsJob
    #     res = MergeContractionsJob(res).out_dict
    if not output_for_rescoring:
        return RecogOutput(output=res)
    elif output_for_rescoring == 'v1':
        return HdfRescoringOutput(out_hyps=out_hyps_hdf,
                                  out_lens=out_lens_hdf,
                                  out_scores=out_scores_hdf,
                                  out_packed_batch_sizes=out_packed_batch_sizes)
    elif output_for_rescoring =='v2':
        return HdfRescoringOutput(out_hyps=out_hyps_hdf)
    else:
        NotImplementedError



# Those are applied for both training, recog and potential others.
# The values are only used if they are neither set in config nor post_config already.
# They should also not infer with other things from the epilog.
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}




def search_config_v2(
    dataset: DatasetConfig,
    model_def: Union[ModelDef, ModelDefWithCfg],
    recog_def: RecogDef,
    *,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
    hdf_output=None,
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
    if hdf_output == 'v1':
        returnn_callback = _returnn_v2_get_forward_hdf_callback
    elif hdf_output == 'v2':
        returnn_callback = _returnn_v2_get_forward_hdf_callback_v2
    else:
        returnn_callback = _returnn_v2_get_forward_callback

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
                    serialization.Import(recog_def, import_as="_recog_def", ignore_import_as_for_hash=True),
                    serialization.Import(_returnn_v2_forward_step, import_as="forward_step"),
                    serialization.Import(returnn_callback, import_as="forward_callback"),
                    serialization.ExplicitHash(
                        {
                            # Increase the version whenever some incompatible change is made in this recog() function,
                            # which influences the outcome, but would otherwise not influence the hash.
                            "version": 2,
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
    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
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

def dummy_get_model(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.tensor import Dim, batch_dim
    from returnn.config import get_global_config

    config = get_global_config()
    feature_dim = Dim(description='feature', dimension=80)
    vocab_dim = Dim(description='vocab', dimension=10026)


    model_def = config.typed_value("_model_def")
    model = model_def(epoch=epoch, in_dim=feature_dim, target_dim=vocab_dim)
    return model

def dummy_forward(*, model, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from torch.nn.utils.rnn import pad_sequence
    from returnn.tensor import Tensor, Dim, batch_dim
    from returnn.config import get_global_config
    import torch

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx+1}/{batch_size} seq_tag: {seq_tag!r}")

    config = get_global_config()
    hyps = extern_data['hyps']
    lens = extern_data['hyp_lens']
    am_score = extern_data['am_scores']
    lm_score = extern_data['lm_scores']
    hyps_raw = hyps.raw_tensor # (batch, beam * T)
    lens_raw = lens.raw_tensor # (batch, beam)
    batch_size = hyps_raw.shape[0]
    beam_size = lens_raw.shape[1]
    max_length = torch.max(lens_raw)
    hyps_unpack = []
    for b in range(batch_size):
        length_b =lens_raw[b]
        length_b_total = length_b.sum()
        flat_hyp = hyps_raw[b,:length_b_total]
        split_hyp = torch.split(flat_hyp, length_b.tolist())
        pad_hyp = pad_sequence(split_hyp,batch_first=True)
        hyps_unpack.append(pad_hyp)
    max_T = max([x.size(1) for x in hyps_unpack])
    for i in range(len(hyps_unpack)):
        pad_len = max_T - hyps_unpack[i].size(1)
        if pad_len > 0:
            pad_tensor = torch.zeros((beam_size, pad_len), dtype=hyps_raw.dtype, device=hyps_raw.device)
            hyps_unpack[i] = torch.cat([hyps_unpack[i], pad_tensor], dim=1)
    hyps_unpack = torch.stack(hyps_unpack) # (batch, beam, T)
    #print(hyps_unpack[:,:,:5].cpu().numpy())
    am_score_raw = am_score.raw_tensor
    lm_score_raw = lm_score.raw_tensor
    #print('am_score:', am_score_raw.cpu().numpy())
    #print('lm_score:', lm_score_raw.cpu().numpy())
    # remove bos
    hyps_unpack = hyps_unpack[:,:,1:]
    print("lens", lens.raw_tensor)
    lens = lens-1
    #hyps_tensor = rf.Tensor(name="hyps", raw_tensor=hyps_unpack)
    rf.get_run_ctx().mark_as_output(hyps_unpack, "hyps")
    rf.get_run_ctx().mark_as_output(am_score, "am_scores")
    rf.get_run_ctx().mark_as_output(lm_score, "lm_scores")
    rf.get_run_ctx().mark_as_output(lens_raw, "hyp_lens")


def debug_hdf_reading(path: tk.Path,
                      model: ModelWithCheckpoint,
                      beam_size=5):
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config
    from returnn.tensor import Dim, batch_dim, single_step_dim
    from returnn.tensor.marked_dim import ImplicitDynSizeDim, ImplicitSparseDim
    from i6_experiments.users.yang.torch.ctc_ilm_kd.model_conformer_with_ilm_v2 import dummy_model_def
    model_def = dummy_model_def
    data_dict={
        "class": "MetaDataset",
        "data_map": {
            "hyps": ("hyps", "data"),
            "am_scores": ("hyps", "am_scores"),
            "lm_scores": ("hyps", "lm_scores"),
            "hyp_lens": ("hyps", "hyp_lens"),
        },
        "datasets": {
            "hyps":{
                "class": "HDFDataset",
                "files": [path],
                "use_cache_manager": True,

            }
        },
        "seq_order_control_dataset": "hyps"
    }
    out_spatial_dim = Dim(description="out_spatial", dimension=None, kind=Dim.Types.Spatial)
    score_dim = Dim(description='score', dimension=beam_size, kind = None)
    vocab_dim = Dim(description="vocab", dimension=10026, kind=Dim.Types.Feature)
    beam_dim = Dim(description='beam_size', dimension=beam_size, kind=None)


    extern_data_raw = {
        "hyps": {"dim_tags": [batch_dim, out_spatial_dim], "sparse_dim": vocab_dim},
        "hyp_lens": {"dim_tags":[batch_dim, beam_dim]},
        "am_scores": {"dim_tags": [batch_dim, beam_dim]},
        "lm_scores": {"dim_tags": [batch_dim, beam_dim]},
    }
    extern_data_raw = instanciate_delayed(extern_data_raw)
    returnn_recog_config_dict = dict(
        backend="torch",
        behavior_version=21,
        # dataset
        default_input="data",
        target="classes",
        forward_data=data_dict,
    )
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
                    serialization.Import(dummy_get_model, import_as="get_model"),
                    serialization.Import(dummy_forward, import_as="forward_step"),
                    serialization.Import(dummy_callback, import_as="forward_callback"),
                    serialization.ExplicitHash(
                        {
                            # Increase the version whenever some incompatible change is made in this recog() function,
                            # which influences the outcome, but would otherwise not influence the hash.
                            "version": 2,
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

    for k, v in dict(
            batching="sorted",
            batch_size=20000,
            max_seqs=20,
    ).items():
        if k in returnn_recog_config.config:
            v = returnn_recog_config.config.pop(k)
        if k in returnn_recog_config.post_config:
            v = returnn_recog_config.post_config.pop(k)
        returnn_recog_config.config[k] = v


    for k, v in SharedPostConfig.items():
        if k in returnn_recog_config.config or k in returnn_recog_config.post_config:
            continue
        returnn_recog_config.post_config[k] = v

    search_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint,
        returnn_config=returnn_recog_config,
        output_files=["output.py.gz"],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        mem_rqmt=6,
        time_rqmt=7,
    )
    return search_job.out_files["output.py.gz"]

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

        def finish(self):
            self.out_file.write("}\n")
            self.out_file.close()
            if self.out_ext_file:
                self.out_ext_file.write("}\n")
                self.out_ext_file.close()

    return _ReturnnRecogV2ForwardCallbackIface()

def dummy_callback():
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
            scores: Tensor = outputs["lm_scores"]  # [beam]
            # AED/Transducer etc will have hyps len depending on beam -- however, CTC will not.
            num_beam = hyps.raw_tensor.shape[0]
            assert hyps.raw_tensor.shape[0] == scores.raw_tensor.shape[0]
            # Consistent to old search task, list[(float,str)].
            self.out_file.write(f"{seq_tag!r}: [\n")
            for i in range(num_beam):
                score = float(scores.raw_tensor[i])
                hyp_ids = hyps.raw_tensor[
                    i]
                self.out_file.write(f"  ({score!r}, {hyp_ids!r}),\n")
            self.out_file.write("],\n")


        def finish(self):
            self.out_file.write("}\n")
            self.out_file.close()
            if self.out_ext_file:
                self.out_ext_file.write("}\n")
                self.out_ext_file.close()

    return _ReturnnRecogV2ForwardCallbackIface()




def _returnn_v2_get_forward_hdf_callback():
    from typing import TextIO
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.config import get_global_config
    from returnn.datasets.hdf import SimpleHDFWriter
    import numpy as np

    config = get_global_config()
    recog_def_ext = config.bool("__recog_def_ext", False)

    class _ReturnnRecogV2HdfForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):


            self.out_hyps: Optional[SimpleHDFWriter] = None
            self.out_lens: Optional[SimpleHDFWriter] = None
            self.out_scores: Optional[SimpleHDFWriter] = None
            self.out_packed_batch_sizes: Optional[SimpleHDFWriter] = None

        def init(self, *, model):
            # multiple hdf files
            target_dim = model.target_dim.dyn_size_ext.raw_tensor.item() if model.target_dim.dyn_size_ext is not None else model.target_dim.size

            self.out_hyps = SimpleHDFWriter(_v2_out_hyp_hdf,dim=target_dim,ndim=1)
            self.out_lens = SimpleHDFWriter(_v2_out_lens_hdf, dim=None, ndim=1)
            self.out_scores = SimpleHDFWriter(_v2_out_scores_hdf, dim=None, ndim=1)
            self.out_packed_batch_sizes = SimpleHDFWriter(_v2_out_packed_batch_sizes, dim=None, ndim=1)

            if recog_def_ext:
                self.out_ext_file = gzip.open(_v2_forward_ext_out_filename, "wt")
                self.out_ext_file.write("{\n")

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            import torch
            from torch.nn.utils.rnn import pack_padded_sequence

            from i6_experiments.users.schmitt import hdf
            hyps: Tensor = outputs["hyps"]  # [beam, out_spatial]
            scores: Tensor = outputs["scores"]  # [beam]
            assert hyps.sparse_dim and hyps.sparse_dim.vocab  # should come from the model
            assert hyps.dims[1].dyn_size_ext, f"hyps {hyps} do not define seq lengths"
            hyps_len = hyps.dims[1].dyn_size_ext  # [beam] or []
            assert hyps.raw_tensor.shape[:1] == scores.raw_tensor.shape  # (beam,)
            if hyps_len.raw_tensor.shape:
                assert scores.raw_tensor.shape == hyps_len.raw_tensor.shape  # (beam,)
            num_beam = hyps.raw_tensor.shape[0]
            hyps_torch = torch.from_numpy(hyps.raw_tensor)
            # manually add a label at the begining to avoid empty sequence
            zero_row = torch.zeros((num_beam,1), dtype=hyps_torch.dtype)
            hyps_torch = torch.cat([zero_row, hyps_torch], axis=1)

            hyp_lengths = hyps_len.raw_tensor
            hyp_lengths = hyp_lengths + 1
            seq_lens = sum(hyp_lengths)
            packed_hyps = pack_padded_sequence(hyps_torch, hyp_lengths, batch_first=True, enforce_sorted=False)
            # tensors write to hdf files
            # we write the hyps and length tensors separately,
            # packed hyp tensor:
            packed_hyps_data = packed_hyps.data[None].cpu().numpy()
            # seq length for each hyp:
            packed_batch_sizes = packed_hyps.batch_sizes.cpu().numpy()
            packed_batch_sizes_len = len(packed_batch_sizes)
            packed_batch_sizes = packed_batch_sizes[None]
            hyp_sorted_indices = packed_hyps.sorted_indices
            hyp_unsorted_indices = packed_hyps.unsorted_indices

            # we store hyp_seq_len, hyp_sorted_indices and hyp_unsorted_indices in one hdf file, as the length of
            # these tensors are fixed: (beam,) for each tensor

            len_info = torch.stack([hyp_sorted_indices, hyp_unsorted_indices])
            len_info = len_info.view(1,-1).cpu().numpy() # and the length should be 2 * beam
            # we assume scores are stored in a tensor with shape (beam, num_scores), where num_scores is the number of scores
            # scores are stored in a separate hdf file
            score_tensor = scores.raw_tensor
            if len(score_tensor.shape)==2:
                num_scores = score_tensor.shape[1]
            elif len(score_tensor.shape)==1:
                num_scores = 1 # only one score for each hyp in the beam
            else:
                NotImplementedError
            hyp_scores = score_tensor[None]# length of hyp_scores: beam * num_scores

            self.out_hyps.insert_batch(
                packed_hyps_data,
                seq_len=np.array([seq_lens]),
                seq_tag=[seq_tag],
                #extra={"seq_sizes": batch_seq_sizes}
                extra=None
            )
            self.out_lens.insert_batch(
                len_info,
                seq_len=np.array([num_beam *2]),
                seq_tag=[seq_tag],
                extra=None
            )
            self.out_scores.insert_batch(
                hyp_scores,
                seq_len=np.array([num_beam * num_scores]),
                seq_tag=[seq_tag],
                extra=None
            )
            self.out_packed_batch_sizes.insert_batch(
                packed_batch_sizes,
                seq_len=np.array([packed_batch_sizes_len]),
                seq_tag=[seq_tag],
                extra=None
            )

        def finish(self):
            self.out_lens.close()
            self.out_hyps.close()
            self.out_scores.close()
            self.out_packed_batch_sizes.close()

    return _ReturnnRecogV2HdfForwardCallbackIface()


def _returnn_v2_get_forward_hdf_callback_v2():
    from typing import TextIO
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.config import get_global_config
    from returnn.datasets.hdf import SimpleHDFWriter
    import numpy as np

    config = get_global_config()
    recog_def_ext = config.bool("__recog_def_ext", False)

    class _ReturnnRecogV2HdfForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):


            self.out_hyps: Optional[SimpleHDFWriter] = None

        def init(self, *, model):
            # multiple hdf files
            target_dim = model.target_dim.dyn_size_ext.raw_tensor.item() if model.target_dim.dyn_size_ext is not None else model.target_dim.size

            self.out_hyps = SimpleHDFWriter(_v2_out_hyp_hdf,dim=target_dim,ndim=1)

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):

            from i6_experiments.users.schmitt import hdf
            hyps: Tensor = outputs["hyps"]  # [beam, out_spatial]
            scores: Tensor = outputs["scores"]  # [beam]
            assert hyps.sparse_dim and hyps.sparse_dim.vocab  # should come from the model
            assert hyps.dims[1].dyn_size_ext, f"hyps {hyps} do not define seq lengths"
            hyps_len = hyps.dims[1].dyn_size_ext  # [beam] or []
            assert hyps.raw_tensor.shape[:1] == scores.raw_tensor.shape  # (beam,)
            if hyps_len.raw_tensor.shape:
                assert scores.raw_tensor.shape == hyps_len.raw_tensor.shape  # (beam,)
            num_beam = hyps.raw_tensor.shape[0]
            hyps_raw = hyps.raw_tensor



            # simply flatten the hyps
            # manually add a token to the beginning, so that each hyp is not empty

            beam_size = hyps_raw.shape[0]
            max_len = hyps_raw.shape[1]

            zero_pad = np.zeros((beam_size,1), dtype=hyps_raw.dtype)
            hyps_extend = np.concatenate([zero_pad,hyps_raw], axis=1)
            hyp_lengths = hyps_len.raw_tensor
            hyp_lengths = hyp_lengths + 1 # lengths of extended seqs
            flat_hyps_extend = np.array([])
            for b in range(beam_size):
                flat_hyps_extend = np.concatenate([flat_hyps_extend, hyps_extend[b,:hyp_lengths[b]]], axis=0)
            flat_hyps_extend = flat_hyps_extend[None] # (1, beam * T)
            score_raw = scores.raw_tensor
            # dummy am and lm score:
            am_score = score_raw
            lm_score = score_raw
            assert len(am_score.shape) == 1
            assert len(lm_score.shape) == 1
            assert am_score.shape[0] == beam_size
            assert lm_score.shape[0] == beam_size

            # store all other information in "extra
            extra = {
                "hyp_lens": hyp_lengths[None],
                "am_scores": am_score[None],
                "lm_scores": lm_score[None],
            }
            flat_seq_length = hyp_lengths.sum()

            self.out_hyps.insert_batch(
                flat_hyps_extend.astype(np.int32),
                seq_len=np.array([flat_seq_length]),
                seq_tag=[seq_tag],
                extra=extra
            )


        def finish(self):
            self.out_hyps.close()

    return _ReturnnRecogV2HdfForwardCallbackIface()



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

        # # ------------ avoid running recog on the last epoch
        # last_fixed_epoch = max(exp.fixed_epochs)
        # recog_and_score_func = d["recog_and_score_func"]
        # res = recog_and_score_func(last_fixed_epoch)
        # assert isinstance(res, ScoreResultCollection)
        # # Add this to the hash, to make sure the pipeline of the recog and scoring influences the hash.
        # d["_last_fixed_epoch_results"] = res
        return sis_tools.sis_hash(d)

    # def update(self):
    #     """
    #     This is run when all inputs have become available,
    #     and we can potentially add further inputs.
    #     The exp (ModelWithCheckpoints) includes a ref to scores_and_learning_rates
    #     which is only available when the training job finished,
    #     thus this is only run at the very end.

    #     Note that this is thus called multiple times,
    #     once scores_and_learning_rates becomes available,
    #     and then once the further recogs become available.
    #     However, only want to check for relevant checkpoints once.
    #     """
    #     if not self._update_checked_relevant_epochs and self.exp.scores_and_learning_rates.available():
    #         from datetime import datetime

    #         log_filename = tk.Path("update.log", self).get_path()
    #         try:
    #             os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    #             log_stream = open(log_filename, "a")
    #         except PermissionError:  # maybe some other user runs this, via job import
    #             log_stream = open("/dev/stdout", "w")
    #         with log_stream:
    #             log_stream.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #             log_stream.write(": get_relevant_epochs_from_training_learning_rate_scores\n")
    #             for epoch in get_relevant_epochs_from_training_learning_rate_scores(
    #                 model_dir=self.exp.model_dir,
    #                 model_name=self.exp.model_name,
    #                 scores_and_learning_rates=self.exp.scores_and_learning_rates,
    #                 n_best=self.check_train_scores_n_best,
    #                 log_stream=log_stream,
    #             ):
    #                 self._add_recog(epoch)
    #         self._update_checked_relevant_epochs = True

    def _add_recog(self, epoch: int):
        if epoch in self._scores_outputs:
            return
        if epoch in self.exclude_epochs:
            return
        res = self.recog_and_score_func(epoch)
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




@dataclasses.dataclass
class HdfRescoringOutput:
    """
    Corresponds to the target values of datasets defined by :class:`Task`
    """
    out_hyps: tk.Path
    out_lens: tk.Path = None
    out_scores: tk.Path = None
    out_packed_batch_sizes: tk.Path = None

class HdfRescoringOutputV2:
    """
    Corresponds to the target values of datasets defined by :class:`Task`
    """
    out_hyps: tk.Path
    out_lens: tk.Path
    out_am_scores: tk.Path = None
    out_lm_scores: tk.Path = None
    out_extra_scores: dict = None