from enum import Enum, auto
from typing import Optional, Any, List, Dict, Tuple
from textwrap import dedent
import numpy as np


class LearningRateSchedules(Enum):
    NONE = auto()
    Newbob = auto()
    NewbobAbs = auto()
    OCLR = auto()
    OCLR_STEP = auto()
    OCLR_STEP_TORCH = auto()
    CONST_DECAY = auto()
    CONST_DECAY_STEP = auto()


class Optimizers(Enum):
    Nadam = auto()
    AdamW = auto()
    SGD = auto()


def get_learning_rate_config(
    schedule: LearningRateSchedules = LearningRateSchedules.Newbob,
    optimizer: Optimizers = Optimizers.Nadam,
    **kwargs,
) -> Tuple[Dict, List]:
    config = {}
    extra_python = []

    if schedule == LearningRateSchedules.NONE:
        pass
    elif schedule == LearningRateSchedules.Newbob:
        config.update(get_newbob_config(**kwargs))
    elif schedule == LearningRateSchedules.NewbobAbs:
        config.update(get_newbob_abs_config(**kwargs))
    elif schedule == LearningRateSchedules.OCLR:
        config.update(get_oclr_config(**kwargs))
    elif schedule == LearningRateSchedules.OCLR_STEP:
        extra_python.append(get_oclr_function(**kwargs))
    elif schedule == LearningRateSchedules.OCLR_STEP_TORCH:
        extra_python.append(get_oclr_function_torch(**kwargs))
    elif schedule == LearningRateSchedules.CONST_DECAY:
        config.update(get_const_decay_config(**kwargs))
    elif schedule == LearningRateSchedules.CONST_DECAY_STEP:
        extra_python.append(get_const_decay_function(**kwargs))
    else:
        raise NotImplementedError

    if optimizer == Optimizers.Nadam:
        config.update(get_nadam_config(**kwargs))
    elif optimizer == Optimizers.AdamW:
        config.update(get_adamw_config(**kwargs))
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


def get_adamw_config(
    epsilon: float = 1e-16,
    weight_decay: float = 1e-03,
    **kwargs,
) -> Dict[str, Dict]:
    return {"optimizer": {"class": "adamw", "epsilon": epsilon, "weight_decay": weight_decay}}


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
    decayed_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
    **kwargs,
) -> dict:
    initial_lr = initial_lr or peak_lr / 10
    decayed_lr = decayed_lr or initial_lr
    final_lr = final_lr or initial_lr / 5
    cycle_epoch = cycle_epoch or (num_epochs * 9) // 20  # 45% of the training
    lr_list = (
        list(np.linspace(initial_lr, peak_lr, cycle_epoch, endpoint=False))[1:]
        + list(np.linspace(peak_lr, decayed_lr, cycle_epoch, endpoint=False))
        + list(np.linspace(decayed_lr, final_lr, num_epochs - 2 * cycle_epoch + 1, endpoint=True))
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
    decayed_lr = decayed_lr or initial_lr
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


def get_oclr_function_torch(
    num_epochs: int,
    n_steps_per_epoch: int,
    peak_lr: float = 1e-03,
    inc_epochs: Optional[int] = None,
    dec_epochs: Optional[int] = None,
    initial_lr: Optional[float] = None,
    decayed_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
    **kwargs,
) -> str:
    initial_lr = initial_lr or peak_lr / 10
    decayed_lr = decayed_lr or initial_lr
    final_lr = final_lr or initial_lr / 5
    inc_epochs = inc_epochs or (num_epochs * 9) // 20
    dec_epochs = dec_epochs or inc_epochs

    return dedent(
        f"""def dynamic_learning_rate(*, global_train_step: int, **_):
            # Increase linearly from initial_lr to peak_lr over the first inc_epoch epochs
            # Decrease linearly from peak_lr to decayed_lr over the next dec_epoch epochs
            # Decrease linearly from decayed_lr to final_lr over the remaining epochs
            initial_lr = {initial_lr}
            peak_lr = {peak_lr}
            decayed_lr = {decayed_lr}
            final_lr = {final_lr}
            inc_epochs = {inc_epochs}
            dec_epochs = {dec_epochs}
            total_epochs = {num_epochs}
            n_steps_per_epoch = {n_steps_per_epoch}

            # -- derived -- #
            steps_increase = inc_epochs * n_steps_per_epoch
            steps_decay = dec_epochs * n_steps_per_epoch
            steps_final = (total_epochs - inc_epochs - dec_epochs) * n_steps_per_epoch

            step_size_increase = (peak_lr - initial_lr) / steps_increase
            step_size_decay = (peak_lr - decayed_lr) / steps_decay
            step_size_final = (decayed_lr - final_lr) / steps_final

            if global_train_step <= steps_increase:
                return initial_lr + step_size_increase * global_train_step
            if global_train_step <= steps_increase + steps_decay:
                return peak_lr - step_size_decay * (global_train_step - steps_increase)
            
            return max(
                decayed_lr - step_size_final * (global_train_step - steps_increase - steps_decay),
                final_lr
            )"""
    )


def get_const_decay_config(
    num_epochs: int,
    const_lr: float = 1e-03,
    cycle_epoch: Optional[int] = None,
    decay_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
    **kwargs,
) -> dict:
    decay_lr = decay_lr or const_lr / 5
    final_lr = final_lr or const_lr / 50
    cycle_epoch = cycle_epoch or (num_epochs * 9) // 20  # 45% of the training
    lr_list = (
        [const_lr] * (cycle_epoch - 1)
        + list(np.linspace(const_lr, decay_lr, cycle_epoch, endpoint=False))
        + list(np.linspace(decay_lr, final_lr, num_epochs - 2 * cycle_epoch))
    )

    return {
        "learning_rates": lr_list,
    }


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
