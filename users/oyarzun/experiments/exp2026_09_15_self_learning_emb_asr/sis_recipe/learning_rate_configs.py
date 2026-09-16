from typing import Any, Dict, Optional, Sequence
from functools import cache

from i6_core.returnn.config import ReturnnConfig
from i6_core.serialization import Collection, PartialImport

from returnn.util.math import PiecewiseLinear
from returnn.config import get_global_config


def linear_warmup_cosine_annealing(
    *,
    learning_rate: float,
    epoch_continuous: float,
    num_warmup_epochs: float,
    max_lr: float,
    min_lr: float,
    num_epochs: int,
    num_constant_epochs: int = 0,
    **_kwargs,
) -> float:
    import math

    if epoch_continuous < num_warmup_epochs:
        """Linear warmup."""
        lr = max_lr * epoch_continuous / num_warmup_epochs
        return lr
    if num_warmup_epochs <= epoch_continuous < num_warmup_epochs + num_constant_epochs:
        """Constant learning rate."""
        return max_lr
    epoch_continuous -= num_warmup_epochs + num_constant_epochs
    """Cosine annealing learning rate schedule."""
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch_continuous / num_epochs))
    return lr


def linear_warmup_cosine_annealing_v2(
    *,
    learning_rate: float,
    epoch_continuous: float,
    num_warmup_epochs: float,
    max_lr: float,
    min_lr: float,
    num_epochs: int,
    num_constant_epochs: int = 0,
    **_kwargs,
) -> float:
    """
    Fix the cosine annealing schedule to account for the whole decay phase
    """
    import math

    if epoch_continuous < num_warmup_epochs:
        """Linear warmup."""
        lr = max_lr * epoch_continuous / num_warmup_epochs
        return lr
    if num_warmup_epochs <= epoch_continuous < num_warmup_epochs + num_constant_epochs:
        """Constant learning rate."""
        return max_lr
    num_decay_epochs = num_epochs - num_warmup_epochs - num_constant_epochs
    epoch_continuous -= num_warmup_epochs + num_constant_epochs
    """Cosine annealing learning rate schedule."""
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch_continuous / num_decay_epochs))
    return lr


def get_cosine_annealing_lr_config(**kwargs) -> ReturnnConfig:
    dyn_lr_import = PartialImport(
        code_object_path="speech_llm.prefix_lm.sis_recipe.exp2025_11_06_speech_llms.learning_rate_configs.linear_warmup_cosine_annealing",
        import_as="dynamic_learning_rate",
        hashed_arguments={**kwargs},
        unhashed_arguments={},
        unhashed_package_root=None,
    )
    return ReturnnConfig(
        config={"learning_rate_control": "constant"},
        python_prolog=[dyn_lr_import],
    )


def get_cosine_annealing_lr_v2_config(**kwargs) -> ReturnnConfig:
    dyn_lr_import = PartialImport(
        code_object_path="speech_llm.prefix_lm.sis_recipe.exp2025_11_06_speech_llms.learning_rate_configs.linear_warmup_cosine_annealing_v2",
        import_as="dynamic_learning_rate",
        hashed_arguments={**kwargs},
        unhashed_arguments={},
        unhashed_package_root=None,
    )
    return ReturnnConfig(
        config={"learning_rate_control": "constant"},
        python_prolog=[dyn_lr_import],
    )


def dyn_lr_piecewise_linear(
    *,
    learning_rate: float,
    epoch_continuous: float,
    piecewise_epochs: Sequence[float],
    piecewise_values: Sequence[float],
    **_kwargs,
) -> float:
    """
    Defines a piecewise linear learning rate function based on the `epoch_continuous` value.

    Can be set as `dynamic_learning_rate` in the RETURNN config.

    :param learning_rate: The base learning rate. Set by RETURNN during training.
    :param epoch_continuous: The epoch_continuous value. Set by RETURNN during training.
    :param piecewise_epochs: The epoch values at which the learning rate changes.
        This function uses epoch_continuous, so this value can also fall in between two whole epochs.
        Must have the same length as `piecewise_values`.
        Must be set by the partial import.
    :param piecewise_values: The values of the learning rate at the corresponding steps.
        Must have the same length as `piecewise_epochs`.
        Must be set by the partial import.
    :param _kwargs: Unused additional arguments for forward compatibility.
    """

    config = get_global_config()
    f = config.typed_dict.get("_learning_rate_piecewise_cache")
    if f is None:
        assert len(piecewise_epochs) == len(piecewise_values)
        assert piecewise_epochs[0] == 0, "first entry in piecewise_epochs must define LR from epoch/step 0"
        last_ep = -1
        for ep in piecewise_epochs:
            assert ep > last_ep, "piecewise_epochs must be monotonically increasing"
            last_ep = ep

        f = PiecewiseLinear(dict(zip(piecewise_epochs, piecewise_values)))
        config.typed_dict["_learning_rate_piecewise_cache"] = f

    assert epoch_continuous is not None
    return f(epoch_continuous) * learning_rate


def get_lr_config(func_name: str, **kwargs) -> ReturnnConfig:
    dyn_lr_import = PartialImport(
        code_object_path=f"speech_llm.prefix_lm.sis_recipe.exp2025_11_06_speech_llms.learning_rate_configs.{func_name}",
        import_as="dynamic_learning_rate",
        hashed_arguments={**kwargs},
        unhashed_arguments={},
        unhashed_package_root=None,
    )
    return ReturnnConfig(
        config={"learning_rate_control": "constant"},
        python_prolog=[dyn_lr_import],
    )
