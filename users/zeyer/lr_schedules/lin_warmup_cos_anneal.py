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
        return learning_rate * lr
    if num_warmup_epochs <= epoch_continuous < num_warmup_epochs + num_constant_epochs:
        """Constant learning rate."""
        return learning_rate * max_lr
    epoch_continuous -= num_warmup_epochs + num_constant_epochs
    """Cosine annealing learning rate schedule."""
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch_continuous / num_epochs))
    return learning_rate * lr
