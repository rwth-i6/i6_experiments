"""
helpers for training
"""

from __future__ import annotations

from typing import Optional, Union, Dict, Any, Sequence

import inspect
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.returnn_common import serialization
from returnn_common import nn

from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoints, Checkpoint, AlignmentCollection, \
    ModelT, ModelDef, TrainDef, FramewiseTrainDef
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.recog import SharedPostConfig


def train(prefix_name: str,
          *,
          task: Task,
          config: Dict[str, Any],
          post_config: Optional[Dict[str, Any]] = None,
          epilog: Sequence[serialization.SerializerObject] = (),
          alignment: Optional[AlignmentCollection] = None,  # TODO... metadataset...
          model_def: ModelDef[ModelT],
          train_def: Union[TrainDef[ModelT], FramewiseTrainDef[ModelT]],
          init_params: Optional[Checkpoint] = None,
          extra_hash: Any = None,
          **kwargs
          ) -> ModelWithCheckpoints:
    """
    train

    Note on hash:
    - model_def/train_def: just the module name + function name goes into the hash, not the content!
    - extra_hash: explicitly goes into the hash
    - others just as one would expect
    """
    returnn_train_config_dict = dict(
        use_tensorflow=True,
        behavior_version=model_def.behavior_version,

        # dataset
        default_input=task.train_dataset.get_default_input(),
        target=task.train_dataset.get_default_target(),
        train=task.train_dataset.get_train_dataset(),
        eval_datasets=task.train_dataset.get_eval_datasets(),

        learning_rate_control_error_measure=train_def.learning_rate_control_error_measure,
        newbob_multi_num_epochs=task.train_epoch_split,

        **config
    )

    max_seq_length_default_target = returnn_train_config_dict.pop("max_seq_length_default_target", None)
    if max_seq_length_default_target is not None:
        max_seq_length = returnn_train_config_dict.setdefault("max_seq_length", {})
        assert isinstance(max_seq_length, dict)
        max_seq_length[task.train_dataset.get_default_target()] = max_seq_length_default_target

    if alignment:
        # TODO... metadataset etc...
        # dummy for now, just insert them into the dict, to get the dependency
        returnn_train_config_dict["_alignment_train_TODO"] = alignment.alignments["train"].hdf_files
        from sisyphus import tk
        returnn_train_config_dict["_alignment_train_TODO2"] = tk.Path("<non-existing-file>")

    if init_params:
        returnn_train_config_dict["import_model_train_epoch1"] = init_params

    returnn_train_config = ReturnnConfig(
        returnn_train_config_dict,
        python_epilog=[serialization.Collection(
            [
                serialization.NonhashedCode(
                    nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(
                        task.train_dataset.get_extern_data())),
                serialization.Import(model_def, "_model_def", ignore_import_as_for_hash=True),
                serialization.Import(train_def, "_train_def", ignore_import_as_for_hash=True),
                serialization.Import(_returnn_get_network, "get_network", use_for_hash=False),
                serialization.ExplicitHash({
                    # Increase the version whenever some incompatible change is made in this train() function,
                    # which influences the outcome, but would otherwise not influence the hash.
                    "version": 1,
                    # Whatever the caller provides. This could also include another version,
                    # but this is up to the caller.
                    "extra": extra_hash
                }),
                serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                serialization.PythonCacheManagerFunctionNonhashedCode,
                serialization.PythonModelineNonhashedCode,
            ] + list(epilog)
        )],
        post_config=dict(  # not hashed
            log_batch_size=True,
            tf_log_memory_usage=True,
            tf_session_opts={"gpu_options": {"allow_growth": True}},
            cleanup_old_models=True,
            restart_after_num_net_reinit=1,  # gets OOM after a few reinits due to pretraining...
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # stop_on_nonfinite_train_score = False,
            # flat_net_construction=True,
        ),
        sort_config=False,
    )
    if post_config:
        returnn_train_config.post_config.update(post_config)

    for k, v in SharedPostConfig.items():
        if k in returnn_train_config.config or k in returnn_train_config.post_config:
            continue
        returnn_train_config.post_config[k] = v

    kwargs = kwargs.copy()
    for k, v in dict(
            log_verbosity=5, num_epochs=150,
            time_rqmt=80, mem_rqmt=15, cpu_rqmt=4).items():
        kwargs.setdefault(k, v)
    returnn_train_job = ReturnnTrainingJob(returnn_train_config, **kwargs)
    returnn_train_job.add_alias(prefix_name + "/train")

    return ModelWithCheckpoints.from_training_job(
        definition=model_def,
        training_job=returnn_train_job)


def _returnn_get_network(*, epoch: int, **_kwargs_unused) -> Dict[str, Any]:
    """called from the RETURNN config"""
    from returnn.config import get_global_config
    nn.reset_default_root_name_ctx()
    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("extern_data")
    data = nn.Data(name=default_input_key, **extern_data_dict[default_input_key])
    targets = nn.Data(name=default_target_key, **extern_data_dict[default_target_key])
    data_spatial_dim = data.get_time_dim_tag()
    targets_spatial_dim = targets.get_time_dim_tag()
    data = nn.get_extern_data(data)
    targets = nn.get_extern_data(targets)
    model_def = config.typed_value("_model_def")
    extra_kwargs = {}
    model_def_sig = inspect.signature(model_def)
    if "training" in model_def_sig.parameters:
        extra_kwargs["training"] = True
    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.feature_dim, **extra_kwargs)
    train_def = config.typed_value("_train_def")
    train_def(
        model=model,
        data=data, data_spatial_dim=data_spatial_dim,
        targets=targets, targets_spatial_dim=targets_spatial_dim)
    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    return net_dict
