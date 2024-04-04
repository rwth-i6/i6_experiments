import copy
from typing import Optional, Dict, List, Callable

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder import network_builder


def add_lm(
        network: Dict,
        rec_layer_name: str,
        target_num_labels: int,
        opts: Dict,
        label_prob_layer: str,
        get_lm_dict_func: Callable,
        lm_embedding_layer_name: str,
):
  # if segmental model, we need to mask the lm output
  if label_prob_layer == "label_log_prob":
    lm_dict = get_lm_dict_func()
    network[rec_layer_name]["unit"].update({
      "lm_output": {
        "class": "unmask",
        "from": "lm_output_masked",
        "mask": "prev:output_emit",
      },
      "lm_output_masked": {
        "class": "masked_computation",
        "from": "prev:output",
        "mask": "prev:output_emit",
        "unit": {
          "class": "subnetwork",
          "from": "data",
          "subnetwork": lm_dict["lm_output"]["subnetwork"],
          "load_on_init": lm_dict["lm_output"]["load_on_init"],
        },
      }
    })
    network[rec_layer_name]["unit"]["lm_output_masked"]["unit"]["subnetwork"].update({
      "output_non_blank": {
        "class": "reinterpret_data",
        "from": "data",
        "set_sparse_dim": 10025,
      },
    })
    network[rec_layer_name]["unit"]["lm_output_masked"]["unit"]["subnetwork"][lm_embedding_layer_name]["from"] = "output_non_blank"

    load_on_init = network[rec_layer_name]["unit"]["lm_output_masked"]["unit"]["load_on_init"]
    if isinstance(load_on_init, dict) and load_on_init["load_if_prefix"] == "lm_output/":
      network[rec_layer_name]["unit"]["lm_output_masked"]["unit"]["load_on_init"]["load_if_prefix"] = "lm_output_masked/"
    if "param_device" in network[rec_layer_name]["unit"]["lm_output_masked"]["unit"]["subnetwork"][lm_embedding_layer_name]:
      del network[rec_layer_name]["unit"]["lm_output_masked"]["unit"]["subnetwork"][lm_embedding_layer_name]["param_device"]

    if opts["add_lm_eos_last_frame"]:
      network_builder.add_is_last_frame_condition(network, rec_layer_name)  # adds layer "is_last_frame"
      network[rec_layer_name]["unit"].update({
        "lm_eos_log_prob": {
            "class": "switch",
            "condition": "is_last_frame",
            "false_from": 0.0,
            "true_from": "lm_eos_log_prob0",
        },
        "lm_eos_log_prob0": {
            "class": "eval",
            "eval": "safe_log(source(0))",
            "from": "lm_eos_prob",
        },
        "lm_eos_prob": {
          "axis": "f",
          "class": "gather",
          "from": "lm_output_prob",
          "position": 0,
        },
      })
      assert network[rec_layer_name]["unit"]["blank_log_prob"]["from"] == "emit_prob0" \
             or network[rec_layer_name]["unit"]["blank_log_prob"]["from"] == ["emit_prob0", "ilm_eos_log_prob"] \
             and network[rec_layer_name]["unit"]["blank_log_prob"]["eval"].startswith(
        "tf.math.log_sigmoid(-source(0))"), (
        "blank_log_prob layer is not as expected"
      )
      blank_log_prob_layer = network[rec_layer_name]["unit"]["blank_log_prob"]
      if type(blank_log_prob_layer["from"]) is str:
        blank_log_prob_layer["from"] = [blank_log_prob_layer["from"]]

      # in the last frame, we want to add the lm eos log prob to the blank log prob
      blank_log_prob_layer["from"].append("lm_eos_log_prob")
      blank_log_prob_layer["eval"] += f" + source({len(blank_log_prob_layer['from']) - 1})"
  else:
    assert label_prob_layer == "output_prob"
    network[rec_layer_name]["unit"].update(get_lm_dict_func())

  network[rec_layer_name]["unit"].update({
    "lm_output_prob": {
      "class": "activation",
      "activation": "softmax",
      "from": "lm_output",
      "n_out": target_num_labels,
    },
  })

  combo_label_prob_layer = f"combo_{label_prob_layer}"
  if combo_label_prob_layer in network[rec_layer_name]["unit"]:
    network[rec_layer_name]["unit"][combo_label_prob_layer]["from"].append("lm_output_prob")
    network[rec_layer_name]["unit"][combo_label_prob_layer]["eval"] += f" + {opts['scale']} * safe_log(source(2))"
  else:
    network[rec_layer_name]["unit"].update({
      combo_label_prob_layer: {
        "class": "eval",
        "from": [label_prob_layer, "lm_output_prob"],
      },
    })
    if label_prob_layer == "label_log_prob":
      network[rec_layer_name]["unit"][combo_label_prob_layer][
        "eval"] = f"source(0) + {opts['scale']} * safe_log(source(1))"
      network[rec_layer_name]["unit"]["label_log_prob_plus_emit"]["from"] = [combo_label_prob_layer, "emit_log_prob"]
    else:
      assert label_prob_layer == "output_prob"
      network[rec_layer_name]["unit"][combo_label_prob_layer][
        "eval"] = f"safe_log(source(0)) + {opts['scale']} * safe_log(source(1))"
      network[rec_layer_name]["unit"]["output"]["from"] = combo_label_prob_layer
      network[rec_layer_name]["unit"]["output"]["input_type"] = "log_prob"
