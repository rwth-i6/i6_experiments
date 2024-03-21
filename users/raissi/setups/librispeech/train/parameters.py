def get_initial_nn_args_viterbi(train_partition: int = 40, num_epochs: int = 15):
    num_epochs *= train_partition
    return {
        "num_input": 50,
        "partition_epochs": {"train": train_partition, "dev": 1},
        "num_epochs": num_epochs,  # this is actually sub epochs
        "keep_epochs": list(range(num_epochs - 10, num_epochs + 1)),
        "keep_best_n": 3,
    }


def get_initial_nn_args_fullsum(train_partition: int = 20, num_epochs: int = 25):
    num_epochs *= train_partition
    return {
        "num_input": 50,
        "partition_epochs": {"train": train_partition, "dev": 1},
        "num_epochs": num_epochs,  # this is actually sub epochs
        "keep_epochs": list(range(num_epochs - 10, num_epochs + 1)),
        "keep_best_n": 3,
    }


def get_finetune_nn_args_fullsum_mix(train_partition: int = 20, num_epochs: int = 15):
    num_epochs *= train_partition
    return {
        "num_input": 50,
        "partition_epochs": {"train": train_partition, "dev": 1},
        "num_epochs": num_epochs,  # this is actually sub epochs
        "keep_epochs": list(range(num_epochs - 10, num_epochs + 1)) + [num_epochs - (train_partition + 5)],
        "keep_best_n": 3,
    }


def get_finetune_nn_args_fullsum(train_partition: int = 40, num_epochs: int = 15):
    num_epochs *= train_partition
    return {
        "num_input": 50,
        "partition_epochs": {"train": train_partition, "dev": 1},
        "num_epochs": num_epochs,  # this is actually sub epochs
        "keep_epochs": list(range(num_epochs - 10, num_epochs + 1)) + [num_epochs - (train_partition + 5)],
        "keep_best_n": 3,
    }
