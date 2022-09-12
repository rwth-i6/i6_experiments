from enum import Enum, auto
from typing import Callable, Optional, Any, List, Dict, Tuple
from i6_core.returnn.config import CodeWrapper


class LearningRateSchedules(Enum):
    Newbob = auto()
    OCLR = auto()


def get_learning_rate_config(
    schedule: LearningRateSchedules = LearningRateSchedules.Newbob, **kwargs
) -> Tuple[Dict, List]:

    config = {}
    extra_python = []

    if schedule == LearningRateSchedules.Newbob:
        config.update(get_newbob_config(**kwargs))
    elif schedule == LearningRateSchedules.OCLR:
        extra_python.append(get_oclr_function(**kwargs))
    else:
        raise NotImplementedError
    config.update(get_nadam_config(**kwargs))

    return config, extra_python


def get_nadam_config(
    epsilon: float = 1e-08,
    **kwargs,
) -> Dict[str, Dict]:
    return {"optimizer": {"class": "nadam", "epsilon": epsilon}}


def get_newbob_config(
    learning_rate: float = 1e-3,
    learning_rates: Optional[List[float]] = None,
    min_learning_rate: float = 1e-5,
    decay: float = 0.9,
    multi_num_epochs: int = 6,
    epochs_per_lr: int = 3,
    combine_errors: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    result = {
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": epochs_per_lr,
        "learning_rate_control_relative_error_relative_lr": True,
        "combine_error_values": combine_errors,
        "newbob_learning_rate_decay": decay,
        "newbob_multi_num_epochs": multi_num_epochs,
        "newbob_multi_update_interval": 1,
    }
    if learning_rates:
        result["learning_rates"] = learning_rates
    return result


def get_oclr_function(
    num_epochs: int,
    n_steps_per_epoch: int,
    peak_lr: float = 1e-3,
    initial_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
    **kwargs,
) -> Callable:
    initial_lr = initial_lr or peak_lr / 10
    final_lr = final_lr or 1e-06
    cycle_epoch = (num_epochs * 9) // 20  # 45% of the training

    return f"""
def dynamic_learning_rate(*,
        global_train_step,
        **kwargs):
    # Increase linearly from initial_lr to peak_lr over the first cycle_epoch epochs
    # Decrease linearly from peak_lr to initial_lr over the next cycle_epoch epochs
    # Decrease linearly from initial_lr to final_lr over the last (total_epochs - 2*cycle_epoch) epochs
    initial_lr = {initial_lr}
    peak_lr = {peak_lr}
    final_lr = {final_lr}
    cycle_epoch = {cycle_epoch}
    total_epochs = {num_epochs}
    n_steps_per_epoch = {n_steps_per_epoch}

    # -- derived -- #
    steps = cycle_epoch * n_steps_per_epoch
    step_size = (peak_lr - initial_lr) / steps
    steps_final = (total_epochs - 2 * cycle_epoch) * n_steps_per_epoch
    step_size_final = (initial_lr - final_lr) / steps_final

    import tensorflow as tf
    n = tf.cast(global_train_step, tf.float32)
    return tf.where(global_train_step <= steps, initial_lr + step_size * n,
               tf.where(global_train_step <= 2*steps, peak_lr - step_size * (n - steps), 
                   tf.maximum(initial_lr - step_size_final * (n - 2*steps), final_lr)))
"""
