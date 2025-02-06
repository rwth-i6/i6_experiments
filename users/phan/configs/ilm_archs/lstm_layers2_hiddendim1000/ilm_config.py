default_ilm_config = { # should be used for transcription LM as well
    "class": "LSTMLMRF",
    "symbol_embedding_dim": 128,
    "emebdding_dropout": 0.0,
    "num_lstm_layers": 2,
    "lstm_hidden_dim": 1000,
    "lstm_dropout": 0.0,
    "use_bottleneck": False,
}
