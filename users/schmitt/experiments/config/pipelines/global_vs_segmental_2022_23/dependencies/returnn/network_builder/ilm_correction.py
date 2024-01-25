from i6_core.returnn.config import CodeWrapper
from typing import Optional, Dict, List
import copy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.network_builder2 import add_is_last_frame_condition


def add_mini_lstm(
        network: Dict,
        rec_layer_name: str,
        train: bool = True
):
  network[rec_layer_name]["unit"].update({
    "mini_att_lstm": {
      "class": "rec",
      "unit": "nativelstm2",
      "n_out": 50,
      "direction": 1,
      "from": "prev:target_embed",
    },
    "mini_att": {
      "class": "linear",
      "activation": None,
      "with_bias": True,
      "from": "mini_att_lstm",
      "n_out": 512,
    },
  })

  if train:
    network[rec_layer_name]["unit"]["readout_in"]["from"] = ["s", "prev:target_embed", "mini_att"]


def add_ilm_correction(network: Dict, rec_layer_name: str, target_num_labels: int, opts: Dict):
  network[rec_layer_name]["unit"].update({
    "prior_s": {
      "class": "rnn_cell",
      "unit": "zoneoutlstm",
      "n_out": 1024,
      "from": ["prev:target_embed", "prev:mini_att"],
      "unit_opts": {
        "zoneout_factor_cell": 0.15,
        "zoneout_factor_output": 0.05,
      },
      "reuse_params": "s_masked/s"
    },
    "prior_readout_in": {
      "class": "linear",
      "activation": None,
      "with_bias": True,
      "from": ["prior_s", "prev:target_embed", "mini_att"],
      "n_out": 1024,
      "reuse_params": "readout_in"
    },
    "prior_readout": {
      "class": "reduce_out",
      "from": ["prior_readout_in"],
      "num_pieces": 2,
      "mode": "max",
    },
    "prior_label_prob": {
      "class": "softmax",
      "from": ["prior_readout"],
      "reuse_params": "label_log_prob",
      "n_out": target_num_labels,
    },
  })

  if rec_layer_name == "output":
    network[rec_layer_name]["unit"].update({
      "label_log_prob_wo_ilm": {
        "class": "eval",
        "eval": f"source(0) - {opts['scale']} * safe_log(source(1))",
        "from": ["label_log_prob", "prior_label_prob"],
      },
    })
    network[rec_layer_name]["unit"]["label_log_prob_plus_emit"]["from"] = ["label_log_prob_wo_ilm", "emit_log_prob"]

  if opts["correct_eos"]:
    add_is_last_frame_condition(network, rec_layer_name)  # adds layer "is_last_frame"
    network[rec_layer_name]["unit"].update({
      "ilm_eos_prob": {
        "class": "gather",
        "from": "prior_label_prob",
        "position": opts["eos_idx"],
        "axis": "f",
      },
      "ilm_eos_log_prob0": {
        "class": "eval",
        "eval": "safe_log(source(0))",
        "from": "ilm_eos_prob",
      },
      "ilm_eos_log_prob": {  # this layer is only non-zero for the last frame
        "class": "switch",
        "condition": "is_last_frame",
        "true_from": "ilm_eos_log_prob0",
        "false_from": 0.0,
      }
    })

    assert network[rec_layer_name]["unit"]["blank_log_prob"]["from"] == "emit_prob0" \
        and network[rec_layer_name]["unit"]["blank_log_prob"]["eval"] == "tf.math.log_sigmoid(-source(0))", (
      "blank_log_prob layer is not as expected"
    )
    # in the last frame, we want to subtract the ilm eos log prob from the blank log prob
    network[rec_layer_name]["unit"]["blank_log_prob"]["from"] = ["emit_prob0", "ilm_eos_log_prob"]
    network[rec_layer_name]["unit"]["blank_log_prob"]["eval"] = "tf.math.log_sigmoid(-source(0)) - source(1)"
