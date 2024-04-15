"""
helpers for training
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Dict, Any, Sequence
from i6_experiments.users.gaudino.model_interfaces import ModelT, ModelDef, ModelDefWithCfg, TrainDef, serialize_model_def

if TYPE_CHECKING:
    from returnn.tensor import TensorDict
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.gaudino.datasets.task import Task
    from i6_experiments.users.gaudino.model_with_checkpoints import ModelWithCheckpoints, Checkpoint


def train(
    prefix_name: str,
    *,
    task: Task,
    config: Dict[str, Any],
    post_config: Optional[Dict[str, Any]] = None,
    env_updates: Optional[Dict[str, str]] = None,
    epilog: Sequence[serialization.SerializerObject] = (),
    model_def: Union[ModelDefWithCfg, ModelDef[ModelT]],
    train_def: TrainDef[ModelT],
    init_params: Optional[Checkpoint] = None,
    extra_hash: Any = None,
    gpu_mem: Optional[int] = None,
    num_processes: Optional[int] = None,
    **kwargs,
) -> ModelWithCheckpoints:
    """
    train

    Note on hash:
    - model_def/train_def: just the module name + function name goes into the hash, not the content!
    - extra_hash: explicitly goes into the hash
    - others just as one would expect

    TODO should use sth like unhashed_package_root (https://github.com/rwth-i6/i6_experiments/pull/157)
    """
    from i6_core.util import instanciate_delayed
    from i6_core.returnn.training import ReturnnTrainingJob
    from i6_core.returnn.config import ReturnnConfig
    from i6_experiments.common.setups import serialization
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config
    from i6_experiments.users.zeyer.utils.serialization import get_import_py_code
    from i6_experiments.users.gaudino.datasets.utils import multi_proc as mp_ds_utils
    from i6_experiments.users.gaudino.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.gaudino.recog_2 import SharedPostConfig
    from returnn_common import nn

    returnn_train_config_dict: Dict[str, Any] = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=task.train_dataset.get_default_input(),
        target=task.train_dataset.get_default_target(),
        train=mp_ds_utils.multi_proc_dataset_opts(task.train_dataset.get_train_dataset()),
        eval_datasets=mp_ds_utils.multi_proc_eval_datasets_opts(task.train_dataset.get_eval_datasets()),
        learning_rate_control_error_measure=train_def.learning_rate_control_error_measure,
        newbob_multi_num_epochs=task.train_epoch_split,
    )
    returnn_train_config_dict.update(config)
    if isinstance(model_def, ModelDefWithCfg):
        returnn_train_config_dict.update(model_def.config)

    max_seq_length_default_target = returnn_train_config_dict.pop("max_seq_length_default_target", None)
    if max_seq_length_default_target is not None:
        max_seq_length = returnn_train_config_dict.setdefault("max_seq_length", {})
        assert isinstance(max_seq_length, dict)
        max_seq_length[task.train_dataset.get_default_target()] = max_seq_length_default_target

    if init_params:
        returnn_train_config_dict["import_model_train_epoch1"] = init_params

    extern_data_raw = task.train_dataset.get_extern_data()
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed(extern_data_raw)

    returnn_train_config = ReturnnConfig(
        returnn_train_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
                    *serialize_model_def(model_def),
                    serialization.Import(_returnn_v2_get_model, import_as="get_model"),
                    serialization.Import(train_def, import_as="_train_def", ignore_import_as_for_hash=True),
                    serialization.Import(_returnn_v2_train_step, import_as="train_step"),
                    serialization.ExplicitHash(
                        {
                            # Increase the version whenever some incompatible change is made in this train() function,
                            # which influences the outcome, but would otherwise not influence the hash.
                            "version": 1,
                            # Whatever the caller provides. This could also include another version,
                            # but this is up to the caller.
                            "extra": extra_hash,
                        }
                    ),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                    serialization.PythonModelineNonhashedCode,
                ]
                + list(epilog)
            )
        ],
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
        sort_config=False,
    )
    if post_config:
        returnn_train_config.post_config.update(post_config)

    for k, v in SharedPostConfig.items():
        if k in returnn_train_config.config or k in returnn_train_config.post_config:
            continue
        returnn_train_config.post_config[k] = v

    # There might be some further functions in the config, e.g. some dataset postprocessing.
    returnn_train_config = get_serializable_config(
        returnn_train_config,
        # The only dim tags we directly have in the config are via extern_data, maybe also model_outputs.
        # All other dim tags are inside functions such as get_model or train_step,
        # so we do not need to care about them here, only about the serialization of those functions.
        # Those dim tags and those functions are already handled above.
        serialize_dim_tags=False,
    )

    kwargs = kwargs.copy()
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

    return ModelWithCheckpoints.from_training_job(definition=model_def, training_job=returnn_train_job)


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


def _returnn_v2_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[default_target_key]
    targets_spatial_dim = targets.get_time_dim_tag()
    train_def: TrainDef = config.typed_value("_train_def")
    train_def(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
    )
