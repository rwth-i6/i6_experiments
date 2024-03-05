__all__ = ["get_train_params_bw", "get_train_params_ce"]


def get_train_params_bw():
    checked = False
    print("you did not check these parameters :(")
    assert checked
    additional_args = {
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
        "specaugment": True,
    }
    return additional_args


def get_train_params_ce(num_layers=6, num_nodes=512, n_epochs=20, train_partition=8, **kwargs):
    chunking = kwargs.pop("chunking") if "chunking" in kwargs else "64:32"

    additional_args = {
        "num_epochs": n_epochs * train_partition,
        "layers": num_layers * [num_nodes],
        "l2": 0.01,
        "lr": 0.0008,
        "dropout": 0.1,
        "partition_epochs": {"train": train_partition, "dev": 1},
        "n_epochs": 20,
        "min_learning_rate": 0.000001,
        "newbob_learning_rate_decay": 0.9,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rate_control_min_num_epochs_per_new_lr": 4,
        "batch_size": 5000,
        "chunking": chunking,
        "max_seqs": int(chunking.split(":")[0]) * 2,
        "nadam": True,
        "truncation": -1,
        "gradient_clip": 0,
        "gradient_noise": 0.1,
        "optimizer_epsilon": 0.00000001,
        "unit_type": "nativelstm2",
        "start_batch": "auto",
        "specaugment": True,
    }
    additional_args.update(kwargs)
    return additional_args
