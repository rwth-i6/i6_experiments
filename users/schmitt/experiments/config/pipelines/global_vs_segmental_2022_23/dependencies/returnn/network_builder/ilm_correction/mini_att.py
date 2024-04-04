from i6_core.returnn.config import ReturnnConfig
from typing import Optional, Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.ilm_correction.ilm_correction import add_ilm_correction as base_add_ilm_correction


def add_mini_att(
        network: Dict,
        rec_layer_name: str,
        train: bool = True,
        use_mask_layer: bool = False,
):
  assert not (use_mask_layer and train), "mask layer only makes sense for inference"
  network[rec_layer_name]["unit"].update({
    "mini_att" if train else "preload_mini_att": {
      "class": "linear",
      "activation": None,
      "with_bias": True,
      "from": "mini_att_lstm" if train else "preload_mini_att_lstm",
      "n_out": 512,
    },
  })

  mini_att_lstm_dict = {
    "mini_att_lstm" if train else "preload_mini_att_lstm": {
      "class": "rec",
      "unit": "nativelstm2",
      "n_out": 50,
      "direction": 1,
      "from": "prev:target_embed",
    },
  }

  if use_mask_layer:
    network[rec_layer_name]["unit"].update({
      "preload_mini_att_lstm": {
        "class": "unmask",
        "from": "preload_mini_att_lstm_masked",
        "mask": "prev:output_emit",
      },
      "preload_mini_att_lstm_masked": {
        "class": "masked_computation",
        "from": "prev:target_embed",
        "mask": "prev:output_emit",
        "unit": {
          "class": "subnetwork",
          "from": "data",
          "subnetwork": {
            **mini_att_lstm_dict,
            "output": {
              "class": "copy",
              "from": "preload_mini_att_lstm",
            }
          }
        }
      }
    })
    network[rec_layer_name]["unit"]["preload_mini_att_lstm_masked"]["unit"]["subnetwork"]["preload_mini_att_lstm"]["from"] = "data"
    network[rec_layer_name]["unit"]["preload_mini_att_lstm_masked"]["unit"]["subnetwork"]["preload_mini_att_lstm"]["name_scope"] = "/output/rec/preload_mini_att_lstm/rec"
  else:
    network[rec_layer_name]["unit"].update(mini_att_lstm_dict)

  if train:
    network[rec_layer_name]["unit"]["readout_in"]["from"] = ["s", "prev:target_embed", "mini_att"]
    network[rec_layer_name]["unit"]["s"]["from"] = ["prev:target_embed", "prev:mini_att"]


def add_ilm_correction(
        network: Dict,
        rec_layer_name: str,
        target_num_labels: int,
        opts: Dict,
        label_prob_layer: str,
        use_mask_layer: bool = False,
        returnn_config: Optional[ReturnnConfig] = None,
):
  add_mini_att(
    network=network,
    rec_layer_name=rec_layer_name,
    train=False,
    use_mask_layer=use_mask_layer,
  )

  base_add_ilm_correction(
    network=network,
    rec_layer_name=rec_layer_name,
    target_num_labels=target_num_labels,
    opts=opts,
    label_prob_layer=label_prob_layer,
    att_layer_name="preload_mini_att",
    static_att=False,
  )
