default_ilm_config = { # should be used for transcription LM as well
    "class": "FFNN_LM_RF",
    "context_size": 3,
    "symbol_embedding_dim": 128,
    "emebdding_dropout": 0.0,
    "num_ff_layers": 2,
    "ff_hidden_dim": 1000,
    "ff_dropout": 0.0,
    "use_bottleneck": False,
}
