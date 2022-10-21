"""
Based on the original config which we want to reproduce here, namely:
rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config.py.
"""


from __future__ import annotations
from typing import Any, Callable, Sequence
import os
import textwrap
from sisyphus import tk
from returnn_common import nn
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import ReturnnSearchJobV2, SearchRemoveLabelJob, SearchBeamJoinScoresJob, SearchTakeBestJob
from i6_experiments.common.setups.returnn_common import serialization

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.datasets.task import Task, DatasetConfig
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput, ScoreResultCollection
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, ModelWithCheckpoint, ModelWithCheckpoints
from i6_experiments.users.zeyer.recog import GetBestRecogTrainExp
from i6_experiments.users.zeyer.datasets.switchboard_2020.task import get_switchboard_task_bpe1k
from ..model import config_code

__my_dir__ = os.path.dirname(os.path.abspath(__file__))
name = "orig_native"
raw_config_filename = f"{__my_dir__}/_{name}_config.py"
config_code_dir = os.path.dirname(os.path.abspath(config_code.__file__))


def broken_sis_run_with_prefix(prefix_name: str):
    """run the exp"""
    task = get_switchboard_task_bpe1k()
    model = train(prefix_name, task=task, extra_hash=name)
    recog_training_exp(prefix_name, task, model)


def train(prefix_name: str,
          *,
          task: Task,
          num_epochs: int = 150,
          extra_hash: Any,
          ) -> ModelWithCheckpoints:
    """train"""
    returnn_train_config_dict = dict(
        use_tensorflow=True,

        # dataset
        default_input=task.train_dataset.get_default_input(),
        target=task.train_dataset.get_default_target(),
        train=task.train_dataset.get_train_dataset(),
        eval_datasets=task.train_dataset.get_eval_datasets(),
    )

    returnn_train_config = ReturnnConfig(
        returnn_train_config_dict,
        python_epilog=[serialization.Collection(
            [
                serialization.NonhashedCode(
                    nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(
                        task.train_dataset.get_extern_data())),
                serialization.ExplicitHash({
                    "version": 1,
                    "extra": extra_hash
                }),
                serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                serialization.PythonCacheManagerFunctionNonhashedCode,
                PythonLoadOrigConfigNonhashedCode,
                serialization.PythonModelineNonhashedCode,
            ]
        )],
        post_config=dict(  # not hashed
            log_batch_size=True,
            tf_log_memory_usage=True,
            tf_session_opts={"gpu_options": {"allow_growth": True}},
            cleanup_old_models=True,
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # stop_on_nonfinite_train_score = False,
            # flat_net_construction=True,
        ),
        sort_config=False,
    )

    returnn_train_job = ReturnnTrainingJob(
        returnn_train_config,
        log_verbosity=5, num_epochs=num_epochs,
        time_rqmt=80, mem_rqmt=15, cpu_rqmt=4)
    returnn_train_job.add_alias(prefix_name + "/train")

    return ModelWithCheckpoints.from_training_job(
        definition=None,
        training_job=returnn_train_job,
        num_pretrain_epochs=num_epochs)


class _RecogDef:
    pass


def recog_training_exp(prefix_name: str, task: Task, model: ModelWithCheckpoints):
    """recog on all relevant epochs"""
    recog_def = _RecogDef()
    recog_def: RecogDef
    recog_def.output_with_beam = True
    recog_def.output_blank_label = "<blank>"
    recog_def.batch_size_dependent = False
    summarize_job = GetBestRecogTrainExp(
        exp=model,
        recog_and_score_func=_RecogAndScoreFunc(prefix_name, task, model, recog_def),
        main_measure_lower_is_better=task.main_measure_type.lower_is_better)
    tk.register_output(prefix_name + "/recog_results_best", summarize_job.out_summary_json)
    tk.register_output(prefix_name + "/recog_results_all_epochs", summarize_job.out_results_all_epochs_json)


class _RecogAndScoreFunc:
    def __init__(self, prefix_name: str, task: Task, model: ModelWithCheckpoints, recog_def: RecogDef):
        self.prefix_name = prefix_name
        self.task = task
        self.model = model
        self.recog_def = recog_def

    def __call__(self, epoch: int) -> ScoreResultCollection:
        model_with_checkpoint = self.model.get_epoch(epoch)
        res = recog_model(self.task, model_with_checkpoint, self.recog_def)
        tk.register_output(self.prefix_name + f"/recog_results_per_epoch/{epoch:03}", res.output)
        return res


def recog_model(task: Task, model: ModelWithCheckpoint, recog_def: RecogDef) -> ScoreResultCollection:
    """recog"""
    outputs = {}
    for name, dataset in task.eval_datasets.items():
        recog_out = search_dataset(
            dataset=dataset, model=model, recog_def=recog_def,
            recog_post_proc_funcs=task.recog_post_proc_funcs)
        score_out = task.score_recog_output_func(dataset, recog_out)
        outputs[name] = score_out
    return task.collect_score_results_func(outputs)


def search_dataset(
    dataset: DatasetConfig, model: ModelWithCheckpoint, recog_def: RecogDef,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = ()
) -> RecogOutput:
    """
    recog on the specific dataset
    """
    search_job = ReturnnSearchJobV2(
        search_data=dataset.get_main_dataset(),
        model_checkpoint=model.checkpoint,
        returnn_config=search_config(dataset, model.definition, recog_def),
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        output_gzip=True,
        log_verbosity=5,
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


def search_config(dataset: DatasetConfig, model_def: ModelDef, recog_def: RecogDef) -> ReturnnConfig:
    """
    config for search
    """

    returnn_recog_config_dict = dict(
        use_tensorflow=True,

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
                serialization.ExplicitHash({
                    "version": 1,
                }),
                serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                serialization.PythonCacheManagerFunctionNonhashedCode,
                PythonLoadOrigConfigNonhashedCode,
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


PythonLoadOrigConfigNonhashedCode = serialization.NonhashedCode(
    textwrap.dedent(
        f"""\
        from returnn.config import get_global_config
        config = get_global_config()
        config.load_file({raw_config_filename!r})
        """
    )
)
