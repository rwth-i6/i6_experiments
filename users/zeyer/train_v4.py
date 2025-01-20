"""
Helper for training.

Note, changes from the earlier v2/v3:
We just use :mod:`serialization_v2` now.
This reduces a lot of complexity.
There is no special handling for dim tags or anything else now.

Note, there is *no* logic for unhashed_package_root here.
If we want that, I think it might make sense to implement that in a more generic way.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Dict, Any
import copy
import functools
from i6_experiments.users.zeyer.model_interfaces import ModelT, ModelDef, ModelDefWithCfg, TrainDef
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from .serialization_v2 import ReturnnConfigWithNewSerialization

if TYPE_CHECKING:
    from returnn.tensor import TensorDict
    from i6_experiments.users.zeyer.datasets.task import Task, DatasetConfig
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, Checkpoint


def train(
    prefix_name: str,
    *,
    task: Optional[Task] = None,
    train_dataset: Optional[DatasetConfig] = None,
    train_epoch_split: Optional[int] = None,
    config: Dict[str, Any],
    post_config: Optional[Dict[str, Any]] = None,
    env_updates: Optional[Dict[str, str]] = None,
    model_def: Union[ModelDefWithCfg, ModelDef[ModelT]],
    train_def: TrainDef[ModelT],
    init_params: Optional[Checkpoint] = None,
    gpu_mem: Optional[int] = None,
    num_processes: Optional[int] = None,
    **kwargs,
) -> ModelWithCheckpoints:
    """
    train

    Note on hash:
    - model_def/train_def: just the module name + function name goes into the hash, not the content!
    - others just as one would expect, i.e. all the config

    Note, there is *no* logic for unhashed_package_root here.
    """
    from sisyphus import tk
    from i6_core.returnn.training import ReturnnTrainingJob
    from i6_experiments.users.zeyer.datasets.utils import multi_proc as mp_ds_utils
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.zeyer.recog import SharedPostConfig

    if train_dataset is None:
        assert task
        train_dataset = task.train_dataset
    train_dataset_dict = train_dataset.get_train_dataset()
    if train_epoch_split is None:
        if task:
            train_epoch_split = task.train_epoch_split
        elif "partition_epoch" in train_dataset_dict:
            train_epoch_split = train_dataset_dict["partition_epoch"]
    # Usually always apply MultiProcDataset. But some exceptions for now:
    apply_multi_proc = train_dataset_dict["class"] != "LmDataset"
    del train_dataset_dict
    del task

    config = config.copy()
    kwargs = kwargs.copy()
    if "__num_epochs" in config:
        kwargs["num_epochs"] = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")
    if "__mem_rqmt" in config:
        kwargs["mem_rqmt"] = config.pop("__mem_rqmt")
    if "__cpu_rqmt" in config:
        kwargs["cpu_rqmt"] = config.pop("__cpu_rqmt")
    if not kwargs.get("distributed_launch_cmd"):
        kwargs["distributed_launch_cmd"] = "torchrun" if num_processes else "mpirun"
    if "__train_audio_preprocess" in config:
        train_dataset = copy.copy(train_dataset)
        assert hasattr(train_dataset, "train_audio_preprocess")
        train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")
    mp_ds_opts = {}
    if "__multi_proc_dataset" in config:
        mp_ds = config.pop("__multi_proc_dataset")
        if isinstance(mp_ds, bool):
            apply_multi_proc = mp_ds
        elif isinstance(mp_ds, int):
            if mp_ds > 0:
                apply_multi_proc = True
                mp_ds_opts["num_workers"] = mp_ds
            else:
                apply_multi_proc = False
        elif isinstance(mp_ds, dict):
            apply_multi_proc = True
            mp_ds_opts.update(mp_ds)
        else:
            raise ValueError(f"invalid __multi_proc_dataset: {mp_ds}")

    returnn_train_config_dict: Dict[str, Any] = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=train_dataset.get_default_input(),
        target=train_dataset.get_default_target(),
        extern_data=train_dataset.get_extern_data(),
        train=(
            mp_ds_utils.multi_proc_dataset_opts(train_dataset.get_train_dataset(), **mp_ds_opts)
            if apply_multi_proc
            else train_dataset.get_train_dataset()
        ),
        eval_datasets=(
            mp_ds_utils.multi_proc_eval_datasets_opts(train_dataset.get_eval_datasets())
            if apply_multi_proc
            else train_dataset.get_eval_datasets()
        ),
        learning_rate_control_error_measure=train_def.learning_rate_control_error_measure,
        newbob_multi_num_epochs=train_epoch_split or 1,
        get_model=functools.partial(
            _returnn_get_model, model_def=model_def.model_def if isinstance(model_def, ModelDefWithCfg) else model_def
        ),
        train_step=functools.partial(_returnn_train_step, train_def=train_def),
    )
    returnn_train_config_dict = dict_update_deep(returnn_train_config_dict, config)
    if isinstance(model_def, ModelDefWithCfg):
        returnn_train_config_dict = dict_update_deep(returnn_train_config_dict, model_def.config)

    max_seq_length_default_target = returnn_train_config_dict.pop("max_seq_length_default_target", None)
    if max_seq_length_default_target is not None:
        max_seq_length = returnn_train_config_dict.setdefault("max_seq_length", {})
        assert isinstance(max_seq_length, dict)
        max_seq_length[train_dataset.get_default_target()] = max_seq_length_default_target
    max_seq_length_default_input = returnn_train_config_dict.pop("max_seq_length_default_input", None)
    if max_seq_length_default_input is not None:
        max_seq_length = returnn_train_config_dict.setdefault("max_seq_length", {})
        assert isinstance(max_seq_length, dict)
        max_seq_length[train_dataset.get_default_input()] = max_seq_length_default_input

    if init_params:
        returnn_train_config_dict["import_model_train_epoch1"] = init_params

    returnn_train_config = ReturnnConfigWithNewSerialization(
        returnn_train_config_dict,
        post_config=dict(  # not hashed
            log_batch_size=True,
            cleanup_old_models=True,
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # stop_on_nonfinite_train_score = False,
            torch_log_memory_usage=True,
            watch_memory=True,
            use_lovely_tensors=True,
            use_train_proc_manager=True,
        ),
    )
    if post_config:
        returnn_train_config.post_config = dict_update_deep(returnn_train_config.post_config, post_config)

    for k, v in SharedPostConfig.items():
        if k in returnn_train_config.config or k in returnn_train_config.post_config:
            continue
        returnn_train_config.post_config[k] = v

    for k, v in dict(
        log_verbosity=5,
        num_epochs=150,
        time_rqmt=80,
        mem_rqmt=30 if gpu_mem and gpu_mem > 11 else 15,
        cpu_rqmt=4 if (not num_processes or num_processes <= 4) else 3,
        horovod_num_processes=num_processes,  # legacy name but also applies for Torch
    ).items():
        if k not in kwargs or kwargs[k] is None:
            kwargs[k] = v
    returnn_train_job = ReturnnTrainingJob(returnn_train_config, **kwargs)
    returnn_train_job.add_alias(prefix_name + "/train")
    if gpu_mem:
        returnn_train_job.rqmt["gpu_mem"] = gpu_mem
    if env_updates:
        for k, v in env_updates.items():
            returnn_train_job.set_env(k, v)
    tk.register_output(prefix_name + "/train_scores", returnn_train_job.out_learning_rates)

    return ModelWithCheckpoints.from_training_job(definition=model_def, training_job=returnn_train_job)


def _returnn_get_model(*, epoch: int, model_def: ModelT, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    assert targets.sparse_dim and targets.sparse_dim.vocab, f"no vocab for {targets}"

    model = model_def(epoch=epoch, in_dim=data.feature_dim_or_sparse_dim, target_dim=targets.sparse_dim)
    return model


def _returnn_train_step(*, model, extern_data: TensorDict, train_def: TrainDef, **_kwargs_unused):
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[default_target_key]
    targets_spatial_dim = targets.get_time_dim_tag()
    train_def(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
    )
