"""
Generic recog, for the model interfaces defined in model_interfaces.py
"""

from __future__ import annotations

import os
from typing import Optional, Union, Any, Dict, Sequence, Collection, Iterator, Callable

import sisyphus
from sisyphus import tk

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.search import ReturnnSearchJobV2, SearchRemoveLabelJob, SearchBeamJoinScoresJob, SearchTakeBestJob
from returnn_common import nn
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.common.setups.returnn_common import serialization

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput, ScoreResultCollection
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, ModelWithCheckpoint, ModelWithCheckpoints
from i6_experiments.users.zeyer.returnn.training import get_relevant_epochs_from_training_learning_rate_scores


def recog_training_exp(
        prefix_name: str,
        task: Task,
        model: ModelWithCheckpoints,
        recog_def: RecogDef,
        *,
        search_post_config: Optional[Dict[str, Any]] = None,
        search_mem_rqmt: Union[int, float] = 6,
        exclude_epochs: Collection[int] = (),
):
    """recog on all relevant epochs"""
    summarize_job = GetBestRecogTrainExp(
        exp=model,
        recog_and_score_func=_RecogAndScoreFunc(
            prefix_name, task, model, recog_def,
            search_post_config=search_post_config, search_mem_rqmt=search_mem_rqmt),
        main_measure_lower_is_better=task.main_measure_type.lower_is_better,
        exclude_epochs=exclude_epochs,
    )
    summarize_job.add_alias(prefix_name + "/train-summarize")
    tk.register_output(prefix_name + "/recog_results_best", summarize_job.out_summary_json)
    tk.register_output(prefix_name + "/recog_results_all_epochs", summarize_job.out_results_all_epochs_json)


class _RecogAndScoreFunc:
    def __init__(self,
                 prefix_name: str, task: Task, model: ModelWithCheckpoints, recog_def: RecogDef, *,
                 search_post_config: Optional[Dict[str, Any]] = None,
                 search_mem_rqmt: Union[int, float] = 6,
                 ):
        # Note: When something is added here, remember to handle it in _sis_hash.
        self.prefix_name = prefix_name
        self.task = task
        self.model = model
        self.recog_def = recog_def
        self.search_post_config = search_post_config
        self.search_mem_rqmt = search_mem_rqmt

    def __call__(self, epoch: int) -> ScoreResultCollection:
        model_with_checkpoint = self.model.get_epoch(epoch)
        res = recog_model(
            self.task, model_with_checkpoint, self.recog_def,
            search_post_config=self.search_post_config, search_mem_rqmt=self.search_mem_rqmt)
        tk.register_output(self.prefix_name + f"/recog_results_per_epoch/{epoch:03}", res.output)
        return res

    def _sis_hash(self) -> bytes:
        from sisyphus.hash import sis_hash_helper
        d = self.__dict__.copy()
        # Remove irrelevant stuff which should not affect the hash.
        del d["prefix_name"]
        del d["search_post_config"]
        del d["search_mem_rqmt"]
        # Not the whole task object is relevant but only some minimal parts.
        task = d.pop("task")
        assert isinstance(task, Task)
        for k in ["train_dataset", "train_epoch_split"]:  # for hash relevant parts
            d[f"task.{k}"] = getattr(task, k)
        d["class"] = "_RecogAndScoreFunc"  # some identifier; not full qualname to allow for moving the class
        return sis_hash_helper(d)


def recog_model(
        task: Task, model: ModelWithCheckpoint, recog_def: RecogDef, *,
        search_post_config: Optional[Dict[str, Any]] = None,
        search_mem_rqmt: Union[int, float] = 6,
) -> ScoreResultCollection:
    """recog"""
    outputs = {}
    for name, dataset in task.eval_datasets.items():
        recog_out = search_dataset(
            dataset=dataset, model=model, recog_def=recog_def,
            search_post_config=search_post_config,
            search_mem_rqmt=search_mem_rqmt,
            recog_post_proc_funcs=task.recog_post_proc_funcs)
        score_out = task.score_recog_output_func(dataset, recog_out)
        outputs[name] = score_out
    return task.collect_score_results_func(outputs)


def search_dataset(
    *,
    dataset: DatasetConfig, model: ModelWithCheckpoint, recog_def: RecogDef,
    search_post_config: Optional[Dict[str, Any]] = None,
    search_mem_rqmt: Union[int, float] = 6,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = ()
) -> RecogOutput:
    """
    recog on the specific dataset
    """
    search_job = ReturnnSearchJobV2(
        search_data=dataset.get_main_dataset(),
        model_checkpoint=model.checkpoint,
        returnn_config=search_config(dataset, model.definition, recog_def, post_config=search_post_config),
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        output_gzip=True,
        log_verbosity=5,
        mem_rqmt=search_mem_rqmt,
    )
    res = search_job.out_search_file
    if recog_def.output_blank_label:
        res = SearchRemoveLabelJob(res, remove_label=recog_def.output_blank_label).out_search_results
    for f in recog_post_proc_funcs:  # for example BPE to words
        res = f(RecogOutput(output=res)).output
    if recog_def.output_with_beam:
        res = SearchBeamJoinScoresJob(res).out_search_results
        res = SearchTakeBestJob(res).out_best_search_results
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
        dataset: DatasetConfig, model_def: ModelDef, recog_def: RecogDef,
        *,
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
    )

    returnn_recog_config = ReturnnConfig(
        config=returnn_recog_config_dict,
        python_epilog=[serialization.Collection(
            [
                serialization.NonhashedCode(
                    nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(
                        dataset.get_extern_data())),
                serialization.Import(model_def, "_model_def", ignore_import_as_for_hash=True),
                serialization.Import(recog_def, "_recog_def", ignore_import_as_for_hash=True),
                serialization.Import(_returnn_get_network, "get_network", use_for_hash=False),
                serialization.ExplicitHash({
                    # Increase the version whenever some incompatible change is made in this recog() function,
                    # which influences the outcome, but would otherwise not influence the hash.
                    "version": 1,
                }),
                serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                serialization.PythonCacheManagerFunctionNonhashedCode,
                serialization.PythonModelineNonhashedCode,
            ]
        )],
        post_config=dict(  # not hashed
            log_batch_size=True,
            tf_log_memory_usage=True,
            tf_session_opts={"gpu_options": {"allow_growth": True}},
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # flat_net_construction=True,
        ),
        sort_config=False,
    )

    (returnn_recog_config.config if recog_def.batch_size_dependent else returnn_recog_config.post_config).update(dict(
        batching="sorted",
        batch_size=20000,
        max_seqs=200,
    ))

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
    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.feature_dim)
    recog_def = config.typed_value("_recog_def")
    recog_out = recog_def(
        model=model,
        data=data, data_spatial_dim=data_spatial_dim, targets_dim=targets.feature_dim)
    assert isinstance(recog_out, nn.Tensor)
    recog_out.mark_as_default_output()
    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    return net_dict


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

    __sis_hash_exclude__ = {"exclude_epochs": ()}

    def __init__(self, exp: ModelWithCheckpoints, *,
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
                log_stream.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                log_stream.write(": get_relevant_epochs_from_training_learning_rate_scores\n")
                for epoch in get_relevant_epochs_from_training_learning_rate_scores(
                        model_dir=self.exp.model_dir, model_name=self.exp.model_name,
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
        res = self.recog_and_score_func(epoch)
        assert isinstance(res, ScoreResultCollection)
        self.add_input(res.main_measure_value)
        self.add_input(res.output)
        self._scores_outputs[epoch] = res

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task('run', mini_task=True)

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
                    f.write(',\n')
                res = json.load(open(score.output.get_path()))
                f.write(f'  "{epoch}": {json.dumps(res)}')
                count += 1
            f.write("\n}\n")
