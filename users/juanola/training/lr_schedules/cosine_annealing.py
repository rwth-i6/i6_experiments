def linear_warmup_cosine_annealing(
        *,
        # Returnn params
        learning_rate: float,
        epoch_continuous: float,

        # Set values
        num_epochs: int,
        max_lr: float,
        min_lr: float,
        num_warmup_epochs: float,
        num_constant_epochs: int = 0,
        **_kwargs,
) -> float:
    """
    v2 for robin repo
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
