"""
Generic recog, for the model interfaces defined in model_interfaces.py
"""

from __future__ import annotations

from typing import Dict, Any, Iterator, Callable

import sisyphus
from sisyphus import tk

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.search import ReturnnSearchJobV2, SearchRemoveLabelJob, SearchBeamJoinScoresJob, SearchTakeBestJob
from returnn_common import nn
from returnn_common.datasets.interface import DatasetConfig
from i6_experiments.common.setups.returnn_common import serialization

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.datasets.task import RecogOutput, Task, ScoreResultCollection
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, ModelWithCheckpoint, ModelWithCheckpoints
from i6_experiments.users.zeyer.returnn.training import get_relevant_epochs_from_training_learning_rate_scores


def recog_training_exp(prefix_name: str, task: Task, model: ModelWithCheckpoints, recog_def: RecogDef):
    """recog on all relevant epochs"""

    def _recog_and_score_func(epoch: int) -> ScoreResultCollection:
        model_with_checkpoint = model.get_epoch(epoch)
        return recog_model(prefix_name + f"/ep{epoch:03}", task, model_with_checkpoint, recog_def)

    summarize_job = SummarizeRecogTrainExp(exp=model, recog_and_score_func=_recog_and_score_func)
    tk.register_output(prefix_name + "/best_recog_results", summarize_job.out_summary_json)


def recog_model(prefix_name: str, task: Task, model: ModelWithCheckpoint, recog_def: RecogDef) -> ScoreResultCollection:
    """recog"""
    outputs = {}
    for name, dataset in task.eval_datasets.items():
        recog_out = search_dataset(dataset=dataset, model=model, recog_def=recog_def)
        for f in task.recog_post_proc_funcs:
            recog_out = f(recog_out)
        score_out = task.score_recog_output_func(dataset, recog_out)
        outputs[name] = score_out
    res = task.collect_score_results_func(outputs)
    tk.register_output(prefix_name + "/recog_results", res.output)
    tk.register_output(prefix_name + "/recog_results_main", res.main_measure_value)
    return res


def search_dataset(dataset: DatasetConfig, model: ModelWithCheckpoint, recog_def: RecogDef) -> RecogOutput:
    """
    recog on the specific dataset
    """
    search_job = ReturnnSearchJobV2(
        search_data=dataset.get_main_dataset(),
        model_checkpoint=model.checkpoint,
        returnn_config=search_config(dataset, model.definition, recog_def),
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    res = search_job.out_search_file
    if recog_def.output_blank_label:
        res = SearchRemoveLabelJob(res, remove_label=recog_def.output_blank_label).out_search_results
        res = SearchBeamJoinScoresJob(res).out_search_results
    if recog_def.output_with_beam:
        res = SearchTakeBestJob(res).out_best_search_results
    return RecogOutput(output=res)


def search_config(dataset: DatasetConfig, model_def: ModelDef, recog_def: RecogDef) -> ReturnnConfig:
    """
    config for search
    """

    returnn_recog_config_dict = dict(
        use_tensorflow=True,
        behavior_version=12,

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


class SummarizeRecogTrainExp(sisyphus.Job):
    """collect all info from recogs"""

    def __init__(self, exp: ModelWithCheckpoints, *,
                 recog_and_score_func: Callable[[int], ScoreResultCollection],
                 check_train_scores_n_best: int = 2):
        """
        :param exp: model, all fixed checkpoints + scoring file for potential other relevant checkpoints (see update())
        :param recog_and_score_func: epoch -> scores. called in graph proc
        :param check_train_scores_n_best: check train scores for N best checkpoints (per each measure)
        """
        super(SummarizeRecogTrainExp, self).__init__()
        self.exp = exp
        self.recog_and_score_func = recog_and_score_func
        self.check_train_scores_n_best = check_train_scores_n_best
        self._update_checked_relevant_epochs = False
        self.out_summary_json = self.output_path("summary.json")
        self._scores_outputs = {}  # type: Dict[int, ScoreResultCollection]  # epoch -> scores out
        for epoch in exp.fixed_kept_epochs:
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
            self._update_checked_relevant_epochs = True
            for epoch in get_relevant_epochs_from_training_learning_rate_scores(
                    model_dir=self.exp.model_dir, model_name=self.exp.model_name,
                    scores_and_learning_rates=self.exp.scores_and_learning_rates,
                    n_best=self.check_train_scores_n_best):
                self._add_recog(epoch)

    def _add_recog(self, epoch: int):
        if epoch in self._scores_outputs:
            return
        self._scores_outputs[epoch] = self.recog_and_score_func(epoch)

    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task('run', mini_task=True)

    def run(self):
        """run"""
        # TODO ... summarize ...
