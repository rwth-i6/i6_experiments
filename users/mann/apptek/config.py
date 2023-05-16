
config = {
    # probably relevant
    "nadam": True,
    "optimizer_epsilon": 1e-8,
    "gradient_clip": 0.0, # this is the default value in RETURNN
    "gradient_noise": 0.0, # might work with noise, too

    # could be tuned
    "batch_size": 10_000,
    "max_seqs": 128,

    # learning rates
    "learning_rate": 0.00025, # not needed
    "learning_rates": [],
    "min_learning_rate": 1e-6,
    "learning_rate_control": "newbob_multi_epoch",
    "learning_rate_control_error_measure": "dev_score_output_bw",
    "learning_rate_control_relative_error_relative_lr": True,

    # dependent on partition epochs
    "learning_rate_control_min_num_epochs_per_new_lr": 3,
    "newbob_learning_rate_decay": 0.9,
    "newbob_multi_num_epochs": 6,
    "newbob_multi_update_interval": 1,
}

# irrelevant
layer_config = {
    "L2": 0.0001,
    "dropout": 0.1,
}
