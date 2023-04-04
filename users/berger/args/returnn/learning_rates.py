from enum import Enum, auto
from typing import Optional, Any, List, Dict, Tuple
from textwrap import dedent
import numpy as np


class LearningRateSchedules(Enum):
    Newbob = auto()
    NewbobAbs = auto()
    OCLR = auto()
    CONST_DECAY = auto()


class Optimizers(Enum):
    Nadam = auto()
    SGD = auto()


def get_learning_rate_config(
    schedule: LearningRateSchedules = LearningRateSchedules.Newbob,
    optimizer: Optimizers = Optimizers.Nadam,
    **kwargs,
) -> Tuple[Dict, List]:

    config = {}
    extra_python = []

    if schedule == LearningRateSchedules.Newbob:
        config.update(get_newbob_config(**kwargs))
    elif schedule == LearningRateSchedules.NewbobAbs:
        config.update(get_newbob_abs_config(**kwargs))
    elif schedule == LearningRateSchedules.OCLR:
        config.update(get_oclr_config(**kwargs))
    elif schedule == LearningRateSchedules.CONST_DECAY:
        extra_python.append(get_const_decay_function(**kwargs))
    else:
        raise NotImplementedError

    if optimizer == Optimizers.Nadam:
        config.update(get_nadam_config(**kwargs))
    elif optimizer == Optimizers.SGD:
        pass
    else:
        raise NotImplementedError

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


def get_newbob_abs_config(
    learning_rate: float = 1e-3,
    decay: float = 0.9,
    multi_num_epochs: int = 6,
    error_measure: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    result = {
        "learning_rate": learning_rate,
        "learning_rate_control": "newbob_abs",
        "learning_rate_control_relative_error_relative_lr": True,
        "newbob_learning_rate_decay": decay,
        "newbob_relative_error_threshold": 0,
        "newbob_multi_num_epochs": multi_num_epochs,
        "newbob_multi_update_interval": 1,
    }
    if error_measure:
        result["learning_rate_control_error_measure"] = error_measure
    return result


def get_oclr_config(
    num_epochs: int,
    peak_lr: float = 1e-03,
    cycle_epoch: Optional[int] = None,
    initial_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
    **kwargs,
) -> dict:
    initial_lr = initial_lr or peak_lr / 10
    final_lr = final_lr or initial_lr / 5
    cycle_epoch = cycle_epoch or (num_epochs * 9) // 20  # 45% of the training
    lr_list = (
        list(np.linspace(initial_lr, peak_lr, cycle_epoch, endpoint=False))
        + list(np.linspace(peak_lr, initial_lr, cycle_epoch, endpoint=False))
        + list(np.linspace(initial_lr, final_lr, num_epochs - 2 * cycle_epoch))
    )

    return {
        "learning_rates": lr_list,
    }


def get_oclr_function(
    num_epochs: int,
    n_steps_per_epoch: int,
    peak_lr: float = 1e-03,
    cycle_epoch: Optional[int] = None,
    initial_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
    **kwargs,
) -> str:
    initial_lr = initial_lr or peak_lr / 10
    final_lr = final_lr or initial_lr / 5
    cycle_epoch = cycle_epoch or (num_epochs * 9) // 20  # 45% of the training

    return dedent(
        f"""def dynamic_learning_rate(*,
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
                           tf.maximum(initial_lr - step_size_final * (n - 2*steps), final_lr)))"""
    )


def get_const_decay_function(
    num_epochs: int,
    n_steps_per_epoch: int,
    const_lr: float = 1e-03,
    cycle_epoch: Optional[int] = None,
    decay_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
    **kwargs,
) -> str:
    # Adapted OCLR by replacing the increase-part with a const-part
    decay_lr = decay_lr or const_lr / 5
    final_lr = final_lr or const_lr / 50
    cycle_epoch = cycle_epoch or (num_epochs * 9) // 20  # 45% of the training

    return dedent(
        f"""def dynamic_learning_rate(*,
                    global_train_step,
                    **kwargs):
                # Keep const_lr over the first cycle_epoch epochs
                # Decrease linearly from const_lr to decay_lr over the next cycle_epoch epochs
                # Decrease linearly from decay_lr to final_lr over the last (total_epochs - 2*cycle_epoch) epochs
                const_lr = {const_lr}
                decay_lr = {decay_lr}
                final_lr = {final_lr}
                cycle_epoch = {cycle_epoch}
                total_epochs = {num_epochs}
                n_steps_per_epoch = {n_steps_per_epoch}
        
                # -- derived -- #
                steps = cycle_epoch * n_steps_per_epoch
                step_size = (const_lr - decay_lr) / steps
                steps_final = (total_epochs - 2 * cycle_epoch) * n_steps_per_epoch
                step_size_final = (decay_lr - final_lr) / steps_final
        
                import tensorflow as tf
                n = tf.cast(global_train_step, tf.float32)
                return tf.where(global_train_step <= steps, const_lr,
                           tf.where(global_train_step <= 2*steps, const_lr - step_size * (n - steps), 
                               tf.maximum(decay_lr - step_size_final * (n - 2*steps), final_lr)))"""
    )
