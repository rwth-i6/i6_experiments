"""
helpers for training
"""

from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.returnn_common import serialization
from .model import ModelWithCheckpoint, ModelDefinition
from .task import Task


def train(task: Task, model_definition: ModelDefinition) -> ModelWithCheckpoint:
    """train"""
    num_epochs = 100
    returnn_train_job = ReturnnTrainingJob(
        _build_train_config(task=task, model_definition=model_definition),
        log_verbosity=5, num_epochs=num_epochs,
        time_rqmt=80, mem_rqmt=15, cpu_rqmt=4)
    return ModelWithCheckpoint(definition=model_definition, checkpoint=returnn_train_job.out_checkpoints[num_epochs])


def _build_train_config(task: Task, model_definition: ModelDefinition):
    import numpy
    returnn_train_config_dict = dict(
        use_tensorflow=True,
        # flat_net_construction=True,

        # TODO dataset...

        # TODO or move all these params to the model definition... ?
        #   although, those are all really train specific
        #   although the model def itself also contains train specific stuff...
        batching="random",
        batch_size=20000,
        max_seqs=200,
        max_seq_length={"classes": 75},

        gradient_clip=0,
        # gradient_clip_global_norm = 1.0
        optimizer={"class": "nadam", "epsilon": 1e-8},
        gradient_noise=0.0,
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
                serialization.ExplicitHash("my_model"),  # TODO
                serialization.PythonEnlargeStackWorkaroundCode,
                serialization.NonhashedCode(model_py_code_str),
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
        ),
        sort_config=False,
    )
    return returnn_train_config
