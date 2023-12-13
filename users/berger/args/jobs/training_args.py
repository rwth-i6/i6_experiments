from typing import Any, Dict


def get_base_training_args(
    *,
    num_epochs: int,
    num_outputs: int,
    log_verbosity: int = 4,
    time_rqmt: int = 168,
    mem_rqmt: int = 16,
    cpu_rqmt: int = 3,
    partition_epochs: int = 1,
    use_python_control: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    result = {
        "log_verbosity": log_verbosity,
        "num_epochs": num_epochs,
        "num_classes": num_outputs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": time_rqmt,
        "mem_rqmt": mem_rqmt,
        "cpu_rqmt": cpu_rqmt,
        "partition_epochs": {"train": partition_epochs, "dev": 1},
        "use_python_control": use_python_control,
    }
    result.update(**kwargs)

    return result
