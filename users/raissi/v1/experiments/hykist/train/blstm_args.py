__all__ = ["get_train_params_ce", "get_train_params_bw"]


def get_train_params_ce(num_layers=6, num_nodes=512, n_epochs=30, train_partition=6, **kwargs):

    chunking = kwargs.pop("chunking") if "chunking" in kwargs else "64:32"
    ce_params = {
        "num_epochs": n_epochs * train_partition,
        "partition_epochs": {"train": train_partition, "dev": 1},
        "layers": num_layers * [num_nodes],
        "l2": 0.01,
        "lr": 0.001,
        "dropout": 0.1,
        "min_learning_rate": 0.00002,
        "newbob_learning_rate_decay": 0.9,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rate_control_min_num_epochs_per_new_lr": train_partition // 2,
        "batch_size": 10000,
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
    ce_params.update(kwargs)

    return ce_params
