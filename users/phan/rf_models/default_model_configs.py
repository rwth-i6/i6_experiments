default_ilm_config = { # should be used for transcription LM as well
    "class": "LSTMLMRF",
    "symbol_embedding_dim": 128,
    "emebdding_dropout": 0.0,
    "num_lstm_layers": 1,
    "lstm_hidden_dim": 1000,
    "lstm_dropout": 0.0,
    "use_bottleneck": False,
}

default_extern_lm_config = {
    "class": "Trafo_LM_Model",
    "layer_out_dim": 1024,
    "layer_ff_dim": 4096,
    "embed_dim": 128,
    "num_layers": 24,
    "att_num_heads": 8,
    "use_pos_enc": True,
    "ff_activation": "relu",
    "pos_enc_diff_pos": True,
}

default_bigram_config = { # almost same as ILM, but one more layer
    "class": "BigramLMRF",
    "symbol_embedding_dim": 128,
    "emebdding_dropout": 0.0,
    "num_ff_layers": 2,
    "ff_hidden_dim": 1000,
    "ff_dropout": 0.0,
    "use_bottleneck": False,
}
