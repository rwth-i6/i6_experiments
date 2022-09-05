"""
helpers for training
"""

from __future__ import annotations
from typing import Optional, Union, Dict, Any
import numpy
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.returnn_common import serialization
from .model import ModelWithCheckpoint, Checkpoint, AlignmentCollection, ModelT, ModelDef, TrainDef, FramewiseTrainDef
from .task import Task


def train(*,
          task: Task,
          alignment: Optional[AlignmentCollection] = None,  # TODO... metadataset...
          model_def: ModelDef[ModelT],
          train_def: Union[TrainDef[ModelT], FramewiseTrainDef[ModelT]],
          init_params: Optional[Checkpoint] = None,  # TODO...
          extra_hash: Any = None,
          ) -> ModelWithCheckpoint:
    """
    train

    Note on hash:
    - model_def/train_def: just the module name + function name goes into the hash, not the content!
    - extra_hash: explicitly goes into the hash
    - others just as one would expect
    """
    num_epochs = 150

    returnn_train_config_dict = dict(
        use_tensorflow=True,

        # dataset
        default_input=task.train_dataset.get_default_input(),
        target=task.train_dataset.get_default_target(),
        train=task.train_dataset.get_train_dataset(),
        eval_datasets=task.train_dataset.get_eval_datasets(),

        batching="random",
        batch_size=20000,
        max_seqs=200,
        max_seq_length={"classes": 75},

        # gradient_clip=0,
        # gradient_clip_global_norm = 1.0
        optimizer={"class": "nadam", "epsilon": 1e-8},
        # gradient_noise=0.0,
        learning_rate=0.0008,
        learning_rates=[0.0003] * 10 + list(numpy.linspace(0.0003, 0.0008, num=10)),
        learning_rate_control="newbob_multi_epoch",
        # learning_rate_control_error_measure = "dev_score_output"
        learning_rate_control_relative_error_relative_lr=True,
        learning_rate_control_min_num_epochs_per_new_lr=3,
        use_learning_rate_control_always=True,
        newbob_multi_num_epochs=task.train_epoch_split,
        newbob_multi_update_interval=1,
        newbob_learning_rate_decay=0.9,
    )

    returnn_train_config = ReturnnConfig(
        returnn_train_config_dict,
        python_epilog=[serialization.Collection(
            [
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

    return ModelWithCheckpoint(
        definition=model_def,
        checkpoint=returnn_train_job.out_checkpoints[num_epochs])


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
    targets_spatial_dim = targets.get_time_dim_tag()
    model_def = config.typed_value("_model_def")
    model = model_def(epoch=epoch)
    train_def = config.typed_value("_train_def")
    train_def(
        model=model,
        data=data, data_spatial_dim=data_spatial_dim,
        targets=targets, targets_spatial_dim=targets_spatial_dim)
    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    return net_dict
