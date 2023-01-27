# SWB-LSTM-BPE-1k
# Config: /u/irie/setups/switchboard/2018-01-23--lmbpe-zeyer/config-train/bpe1k_clean_i256_m2048_m2048.sgd_b16_lr0_cl2.newbobabs.d0.2.config
swb_lstm_bpe1k_model = "/work/asr3/irie/experiments/lm/switchboard/2018-01-23--lmbpe-zeyer/data-train/bpe1k_clean_i256_m2048_m2048.sgd_b16_lr0_cl2.newbobabs.d0.2/net-model/network.023"
swb_lstm_bpe1k_net = {
  "input": {"class": "linear", "n_out": 256, "activation": "identity"},
  "lstm0": {"class": "rnn_cell", "unit": "LSTMBlock", "dropout": 0.2, "n_out": 2048, "unit_opts": {"forget_bias": 0.0}, "from": ["input"]},
  "lstm1": {"class": "rnn_cell", "unit": "LSTMBlock", "dropout": 0.2, "n_out": 2048, "unit_opts": {"forget_bias": 0.0}, "from": ["lstm0"]},
  "output": {"class": "linear", "from": ["lstm1"], "activation": "identity", "dropout": 0.2, "n_out": 1030}
}
swb_lstm_bpe1k_subnet = {
  "class": "subnetwork", "from": ["prev:output"], "load_on_init": swb_lstm_bpe1k_model,
  "subnetwork": swb_lstm_bpe1k_net, "n_out": 1030
}

swb_lstm_bpe500_model = '/work/asr4/zeineldeen/setups-data/switchboard/2021-02-21--lm-bpe/work/crnn/training/CRNNTrainingFromFile.0WtMFNOTbEhO/output/models/epoch.029'
swb_lstm_bpe500_net = {
  "input": {"class": "linear", "n_out": 256, "activation": "identity",
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)"},
  "lstm0": {"class": "rec", "unit": "lstm",
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "n_out": 2048, "dropout": 0.2, "L2": 0.0, "direction": 1, "from": ["input"]},
  "lstm1": {"class": "rec", "unit": "lstm",
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "n_out": 2048, "dropout": 0.2, "L2": 0.0, "direction": 1, "from": ["lstm0"]},
  "output": {"class": "linear", "dropout": 0.2, "activation": "identity", "n_out": 534,
             "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
             "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)", "from": ["lstm1"]}
}

# Librispeech-LSTM-BPE-10k
# config: /u/irie/setups/librispeech/2018-03-05--lmbpe-zeyer/config-train/re_i128_m2048_m2048_m2048_m2048.sgd_b32_lr0_cl2.newbobabs.d0.0.1350
libri_lstm_bpe10k_model = '/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/re_i128_m2048_m2048_m2048_m2048.sgd_b32_lr0_cl2.newbobabs.d0.0.1350/bk-net-model/network.035'
libri_lstm_bpe10k_net = {
  "input": {"class": "linear", "n_out": 128, "activation": "identity"},
  "lstm0": {"class": "rnn_cell", "unit": "LSTMBlock", "dropout": 0.0, "n_out": 2048,
            "unit_opts": {"forget_bias": 0.0}, "from": ["input"]},
  "lstm1": {"class": "rnn_cell", "unit": "LSTMBlock", "dropout": 0.0, "n_out": 2048,
            "unit_opts": {"forget_bias": 0.0}, "from": ["lstm0"]},
  "lstm2": {"class": "rnn_cell", "unit": "LSTMBlock", "dropout": 0.0, "n_out": 2048,
            "unit_opts": {"forget_bias": 0.0}, "from": ["lstm1"]},
  "lstm3": {"class": "rnn_cell", "unit": "LSTMBlock", "dropout": 0.0, "n_out": 2048,
            "unit_opts": {"forget_bias": 0.0}, "from": ["lstm2"]},
  "output": {"class": "linear", "from": ["lstm3"], "activation": "identity", "dropout": 0.0,
             "use_transposed_weights": True, "n_out": 10025}
}
libri_lstm_bpe10k_subnet = {
  "class": "subnetwork", "from": ["prev:output"], "load_on_init": libri_lstm_bpe10k_model,
  "subnetwork": libri_lstm_bpe10k_net, "n_out": 10025
}

libri_100_lstm_bpe2k_model = '/work/asr4/zeineldeen/setups-data/librispeech/2021-02-21--lm-bpe/dependencies/lm_models/lstm/epoch.004'
libri_100_lstm_bpe2k_net = {
  "input": {"class": "linear", "n_out": 128, "activation": "identity",
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)"},
  "lstm0": {"class": "rec", "unit": "lstm",
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "n_out": 2048, "dropout": 0.0, "L2": 0.0, "direction": 1, "from": ["input"]},
  "lstm1": {"class": "rec", "unit": "lstm",
            "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
            "n_out": 2048, "dropout": 0.0, "L2": 0.0, "direction": 1, "from": ["lstm0"]},
  "output": {"class": "linear", "dropout": 0.0, "activation": "identity", "n_out": 2051,
             "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
             "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)", "from": ["lstm1"],
             "use_transposed_weights": True}
}

swb_trans_only_lstm_bpe1k_model = '/work/asr4/zeineldeen/setups-data/switchboard/2021-02-21--lm-bpe/work/crnn/training/CRNNTrainingFromFile.yznBg5IDLMoF/output/models/epoch.024'


def get_swb_lstm_net(num_layers=2, lstm_dim=2048, vocab_size=1030):
  d = {
    "input": {"class": "linear", "n_out": 256, "activation": "identity"},
    "output": {
      "class": "linear", "from": ["lstm%i" % (num_layers - 1)], "activation": "identity", "dropout": 0.2,
      "n_out": vocab_size
    }
  }
  src = 'input'
  for i in range(num_layers):
    d['lstm%i' % i] = {
      "class": "rnn_cell", "unit": "LSTMBlock", "dropout": 0.2, "n_out": lstm_dim, "unit_opts": {"forget_bias": 0.0},
      "from": [src]
    }
    src = 'lstm%i' % i
  return d
