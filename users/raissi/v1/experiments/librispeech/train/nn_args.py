__all__ = ["get_initial_nn_args"]


def get_initial_nn_args():
    return {
        "num_input": 50,
        "partition_epochs": {"train": 40, "dev": 1},
        "num_epochs": 600,  # this is actually sub epochs
        "keep_epochs": [500, 550, 590, 592, 596, 298, 600],
    }
