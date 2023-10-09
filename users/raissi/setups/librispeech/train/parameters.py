__all__ = ["get_initial_nn_args"]


def get_initial_nn_args_viterbi():
    return {
        "num_input": 50,
        "partition_epochs": {"train": 40, "dev": 1},
        "num_epochs": 600,
        "keep_epochs": None,
        "keep_best_n": None,
    }


def get_initial_nn_args_fullsum():
    return {
        "num_input": 50,
        "partition_epochs": {"train": 20, "dev": 1},
        "num_epochs": 500,
        "keep_epochs": None,
        "keep_best_n": None,
    }
