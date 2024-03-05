__all__ = ["get_train_params_bw", "get_train_params_ce"]


def get_train_params_bw():
    additional_args = {
        "partition_epochs": {"train": 20, "dev": 1},
        "n_epochs": 25,
        "layers": 6 * [512],
        "l2": 0.0001,
        "lr": 0.0003,
        "dropout": 0.1,
        "min_learning_rate": 0.00001,
        "newbob_learning_rate_decay": 0.9,
        "batch_size": 10000,
        "chunking": "64:32",
        "max_seqs": 128,
        "nadam": True,
        "truncation": -1,
        "gradient_clip": 0,
        "gradient_noise": 0,
        "optimizer_epsilon": 0.00000001,
        "unit_type": "nativelstm2",
        "start_batch": "auto",
        "python_epilog": mask_code_10_simon,
        "learning_rates": get_learning_rates(increase=225, decay=225),  # n_epochs*0.9//2
        "specaugment": True,
    }
    return additional_args


def get_train_params_ce():
    additional_args = {
        "partition_epochs": {"train": 20, "dev": 1},
        "n_epochs": 25,
        "layers": 6 * [512],
        "l2": 0.0001,
        "lr": 0.0003,
        "dropout": 0.1,
        "min_learning_rate": 0.00001,
        "newbob_learning_rate_decay": 0.9,
        "batch_size": 10000,
        "chunking": "64:32",
        "max_seqs": 128,
        "nadam": True,
        "truncation": -1,
        "gradient_clip": 0,
        "gradient_noise": 0,
        "optimizer_epsilon": 0.00000001,
        "unit_type": "nativelstm2",
        "start_batch": "auto",
        "python_epilog": mask_code_10_simon,
        "learning_rates": get_learning_rates(increase=225, decay=225),  # n_epochs*0.9//2
        "specaugment": True,
    }
    return additional_args
