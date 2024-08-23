from typing import Dict, List


def generate_specaug_params(
    num_epochs: int, base_values: Dict[str, int], schedules: Dict[str, Dict[int, float]], growth_strategy: str
) -> Dict[str, List[int]]:
    """
    Generate specaug parameters for each epoch in advance.

    Args:
        num_epochs (int): Total number of epochs.
        base_values (Dict[str, int]): Base values for each parameter.
        schedules (Dict[str, Dict[int, float]]): Schedules for each parameter.
        growth_strategy (str): Strategy for mask growth over epochs.

    Returns:
        Dict[str, List[int]]: Pre-generated parameter values for each epoch.
    """
    params = {key: [] for key in base_values.keys()}

    for epoch in range(num_epochs):
        for key, base_value in base_values.items():
            schedule = schedules.get(key, {})
            if schedule == {}:
                schedule = {0: 1.0}
            assert 0 in schedule, "Schedule must start at epoch 0"
            param_value = get_mask_param(epoch, base_value, schedule, growth_strategy)
            params[key].append(param_value)

    return params


def get_mask_param(current_epoch: int, base_value: int, schedule: Dict[int, float], growth_strategy: str) -> int:
    """
    Get the mask parameter value for the current epoch.

    Args:
        current_epoch (int): Current epoch number.
        base_value (int): Base value of the parameter.
        schedule (Dict[int, float]): Schedule for the parameter growth.
        growth_strategy (str): Strategy for mask growth over epochs.

    Returns:
        int: Calculated parameter value for the current epoch.
    """
    epochs = sorted(schedule.keys())

    for i in range(len(epochs) - 1):
        if epochs[i] <= current_epoch < epochs[i + 1]:
            lower_epoch, upper_epoch = epochs[i], epochs[i + 1]
            lower_mult, upper_mult = schedule[lower_epoch], schedule[upper_epoch]

            if growth_strategy == "linear":
                progress = (current_epoch - lower_epoch) / (upper_epoch - lower_epoch)
                multiplier = lower_mult + (upper_mult - lower_mult) * progress
            elif growth_strategy == "step":
                multiplier = lower_mult
            else:
                raise ValueError(f"Unknown mask growth strategy: {growth_strategy}")

            return int(base_value * multiplier)

    # If we're past the last epoch in the schedule, use the last multiplier
    last_mult = schedule[epochs[-1]]
    return int(base_value * last_mult)
