import copy


def get_lm_dict():
  return copy.deepcopy({
    "lm_output": {
      "class": "subnetwork",
      "from": "prev:output",
      "subnetwork": {
        "input": {
          "class": "linear",
          "n_out": 128,
          "activation": "identity"
        },
        "lstm0": {
          "class": "rnn_cell",
          "unit": "LSTMBlock",
          "dropout": 0.0,
          "n_out": 2048,
          "unit_opts": {
            "forget_bias": 0.0
          },
          "from": ["input"]
        },
        "lstm1": {
          "class": "rnn_cell",
          "unit": "LSTMBlock",
          "dropout": 0.0,
          "n_out": 2048,
          "unit_opts": {
            "forget_bias": 0.0
          },
          "from": ["lstm0"]
        },
        "lstm2": {
          "class": "rnn_cell",
          "unit": "LSTMBlock",
          "dropout": 0.0,
          "n_out": 2048,
          "unit_opts": {
            "forget_bias": 0.0
          },
          "from": ["lstm1"]
        },
        "lstm3": {
          "class": "rnn_cell",
          "unit": "LSTMBlock",
          "dropout": 0.0,
          "n_out": 2048,
          "unit_opts": {
            "forget_bias": 0.0
          },
          "from": ["lstm2"]
        },
        "output": {
          "class": "linear",
          "from": ["lstm3"],
          "activation": "identity",
          "dropout": 0.0,
          "use_transposed_weights": True,
          "n_out": 10025
        }
      },
      # "load_on_init": {
      #   "filename": "/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/re_i128_m2048_m2048_m2048_m2048.sgd_b32_lr0_cl2.newbobabs.d0.0.1350/bk-net-model/network.035",
      #   "load_if_prefix": "lm_output/",
      #   "params_prefix": "",
      # },
      "load_on_init": "/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/re_i128_m2048_m2048_m2048_m2048.sgd_b32_lr0_cl2.newbobabs.d0.0.1350/bk-net-model/network.035",
    }
  })
