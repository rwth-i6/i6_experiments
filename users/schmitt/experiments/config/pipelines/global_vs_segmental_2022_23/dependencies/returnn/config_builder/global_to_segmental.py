from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition, SegmentalLabelDefinition, LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBSprintCorpora, SWBOggZipCorpora
from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict as get_oggzip_dataset_dict
from i6_experiments.users.schmitt.datasets.concat import get_concat_dataset_dict
from i6_experiments.users.schmitt.datasets.extern_sprint import get_dataset_dict as get_extern_sprint_dataset_dict
from i6_experiments.users.schmitt.conformer_pretrain import get_network
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask
from i6_experiments.users.schmitt.augmentation.alignment import shift_alignment_boundaries_func_str
from i6_experiments.users.schmitt.dynamic_lr import dynamic_lr_str
from i6_experiments.users.schmitt.chunking import custom_chunkin_func_str
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn import network_builder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn import custom_construction_algos
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import SegmentalConfigBuilder

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from sisyphus import Path

import os
import re
from abc import abstractmethod, ABC
from typing import Dict, Optional, List
import copy
import numpy as np


class MohammadGlobalAttToSegmentalAttentionMaker:
  @staticmethod
  def make_global_attention_segmental(
          global_net_dict,
          task: str,
          blank_idx: int,
          target_num_labels_w_blank: int,
          target_num_labels_wo_blank: int,
          network_opts: Dict,
          config_dict: Dict,
          python_prolog: List,
  ) -> Dict:
    def _remove_not_needed_layers():
      del seg_net_dict["enc_value"]
      del seg_net_dict["enc_ctx"]
      if task == "train":
        del seg_net_dict["decision"]

      if "ctc_forced_align" in seg_net_dict:
        del seg_net_dict["ctc_forced_align"]
        del seg_net_dict["ctc_forced_align_dump"]

      del seg_net_dict["output"]["unit"]["end"]

    def _add_base_layers():
      seg_net_dict.update({
        "existing_alignment": {
          "class": "reinterpret_data",
          "from": "data:targets",
          "set_sparse": True,
          "set_sparse_dim": target_num_labels_w_blank,
          "size_base": "encoder",
        },
        "is_label": {
          "class": "compare",
          "from": "existing_alignment",
          "kind": "not_equal",
          "value": blank_idx,
        },
        "label_ground_truth_masked": {
          "class": "reinterpret_data",
          "enforce_batch_major": True,
          "from": "label_ground_truth_masked0",
          "register_as_extern_data": "label_ground_truth",
          "set_sparse_dim": target_num_labels_wo_blank,
        },
        "label_ground_truth_masked0": {
          "class": "masked_computation",
          "from": "existing_alignment",
          "mask": "is_label",
          "unit": {"class": "copy", "from": "data"},
        },
        "emit_ground_truth": {
          "class": "reinterpret_data",
          "from": "emit_ground_truth0",
          "is_output_layer": True,
          "register_as_extern_data": "emit_ground_truth",
          "set_sparse": True,
          "set_sparse_dim": 2,
        },
        "emit_ground_truth0": {
          "class": "switch",
          "condition": "is_label",
          "false_from": "const0",
          "true_from": "const1",
        },
        "const0": {"class": "constant", "value": 0, "with_batch_dim": True},
        "const1": {"class": "constant", "value": 1, "with_batch_dim": True},
        "labels_with_blank_ground_truth": {
          "class": "copy",
          "from": "existing_alignment",
          "register_as_extern_data": "targetb",
        },
        "segment_lens_masked": {
          "class": "masked_computation",
          "from": "output/segment_lens",
          "mask": "is_label",
          "register_as_extern_data": "segment_lens_masked",
          "unit": {"class": "copy", "from": "data"},
        },
        "segment_starts_masked": {
          "class": "masked_computation",
          "from": "output/segment_starts",
          "mask": "is_label",
          "register_as_extern_data": "segment_starts_masked",
          "unit": {"class": "copy", "from": "data"},
        },
      })

      seg_net_dict["ctc"]["target"] = "label_ground_truth"

    def _add_label_model_layer():
      seg_net_dict["label_model"] = copy.deepcopy(seg_net_dict["output"])

      seg_net_dict["label_model"]["name_scope"] = "output/rec"
      seg_net_dict["label_model"]["is_output_layer"] = True
      seg_net_dict["label_model"]["target"] = "label_ground_truth"
      del seg_net_dict["label_model"]["max_seq_len"]

      seg_net_dict["label_model"]["unit"]["output"]["target"] = "label_ground_truth"
      seg_net_dict["label_model"]["unit"]["output_prob"]["target"] = "label_ground_truth"

      seg_net_dict["label_model"]["unit"].update({
        "segment_lens": {
          "axis": "t",
          "class": "gather",
          "from": "base:data:segment_lens_masked",
          "position": ":i",
        },
        "segment_starts": {
          "axis": "t",
          "class": "gather",
          "from": "base:data:segment_starts_masked",
          "position": ":i",
        },
      })

    def _add_label_model_att_layers(rec_layer_name: str):
      if "att_weights0" in seg_net_dict[rec_layer_name]["unit"]:
        assert seg_net_dict[rec_layer_name]["unit"]["att_weights0"]["class"] == "softmax_over_spatial"
        att_weights_layer_name = "att_weights0"
      else:
        att_weights_layer_name = "att_weights"
      seg_net_dict[rec_layer_name]["unit"][att_weights_layer_name]["axis"] = "stag:sliced-time:segments"
      seg_net_dict[rec_layer_name]["unit"]["energy_in"]["from"] = [
        "att_ctx",
        "weight_feedback",
        "s_transformed" if task == "train" else "att_query"]
      seg_net_dict[rec_layer_name]["unit"]["att0"]["base"] = "att_val"

      seg_net_dict[rec_layer_name]["unit"].update({
        "segments": {
          "class": "slice_nd",
          "from": "base:encoder",
          "size": "segment_lens",
          "start": "segment_starts"
        },
        "att_ctx": {
          "activation": None,
          "class": "linear",
          "name_scope": "/enc_ctx",
          "from": "segments",
          "n_out": 1024,
          "with_bias": True,
        },
        "att_val": {"class": "copy", "from": "segments"},
      })

    def _add_output_layer():
      if task == "train":
        seg_net_dict["output"]["unit"] = {}
        del seg_net_dict["output"]["max_seq_len"]
        target = "targetb"

        seg_net_dict["output"]["unit"].update({
          "emit_blank_log_prob": {
            "class": "copy",
            "from": ["blank_log_prob", "emit_log_prob"],
          },
          "emit_blank_prob": {
            "activation": "exp",
            "class": "activation",
            "from": "emit_blank_log_prob",
            "loss": "ce",
            "loss_opts": {"focal_loss_factor": 0.0},
            "target": "emit_ground_truth",
          },
          "output": {
            "beam_size": 4,
            "cheating": "exclusive",
            "class": "choice",
            "from": "data",
            "initial_output": 0,
            "input_type": "log_prob",
            "target": target,
          },
        })

      else:
        target = "targets"

        seg_net_dict["output"]["unit"]["output"]["length_normalization"] = False
        seg_net_dict["output"]["unit"]["att_query"] = seg_net_dict["output"]["unit"]["s_transformed"].copy()
        seg_net_dict["output"]["unit"]["att_query"]["name_scope"] = "/output/rec/s_transformed"
        seg_net_dict["output"]["unit"]["att_query"]["from"] = "lm"
        del seg_net_dict["output"]["unit"]["s_transformed"]
        del seg_net_dict["output"]["unit"]["output_prob"]
        seg_net_dict["output"]["unit"]["readout_in"]["from"] = ["lm", "prev:target_embed", "att"]
        seg_net_dict["output"]["unit"].update({
          "label_log_prob0": {
            "class": "linear",
            "activation": "log_softmax",
            "from": "readout",
            "n_out": target_num_labels_wo_blank,
            "name_scope": "/output/rec/output_prob"
          },
          "label_log_prob": {
            "class": "combine",
            "from": ["label_log_prob0", "emit_log_prob"],
            "kind": "add",
          },
          "lm": {"class": "unmask", "from": "lm_masked", "mask": "prev:output_emit"},
          "lm_masked": {
            "class": "masked_computation",
            "from": "prev:target_embed",
            "mask": "prev:output_emit",
            "unit": {
              "class": "subnetwork",
              "from": "data",
              "subnetwork": {
                "lm": {
                  "class": "rnn_cell",
                  "from": ["data", "base:prev:att"],
                  "n_out": 1024,
                  "name_scope": "/output/rec/s/rec",
                  "unit": "zoneoutlstm",
                  "unit_opts": {
                    "zoneout_factor_cell": 0.15,
                    "zoneout_factor_output": 0.05,
                  },
                },
                "output": {"class": "copy", "from": "lm"},
              },
            },
          },
          "output_log_prob": {
            "class": "copy",
            "from": ["label_log_prob", "blank_log_prob"],
          },
          "output": {
            "beam_size": 12,
            "cheating": None,
            "class": "choice",
            "from": "output_log_prob",
            "initial_output": 0,
            "input_type": "log_prob",
            "length_normalization": False,
            "target": "targets",
          },
          "target_embed_masked": {
            "class": "masked_computation",
            "from": "output",
            "mask": "output_emit",
            "initial_output": 0,
            "unit": {
              "class": "subnetwork",
              "from": "data",
              "subnetwork": {
                "output_non_blank": {
                  "class": "reinterpret_data",
                  "from": "data",
                  "set_sparse_dim": target_num_labels_wo_blank
                },
                "target_embed0": copy.deepcopy(seg_net_dict["output"]["unit"]["target_embed0"]),
                "target_embed": copy.deepcopy(seg_net_dict["output"]["unit"]["target_embed"]),
                "output": {"class": "copy", "from": "target_embed"},
              },
            },
          },
          "target_embed": {
            "class": "unmask",
            "from": "target_embed_masked",
            # "mask": "prev:output_emit",
            "mask": "output_emit",
            "initial_output": 0
          },
        })

        del seg_net_dict["output"]["unit"]["target_embed0"]
        seg_net_dict["output"]["unit"]["target_embed_masked"]["unit"]["subnetwork"]["target_embed0"]["from"] = "output_non_blank"
        seg_net_dict["output"]["unit"]["target_embed_masked"]["unit"]["subnetwork"]["target_embed0"]["name_scope"] = "/output/rec/target_embed0"

        seg_net_dict.update({
          "output_non_blank": {
            "class": "compare",
            "from": "output",
            "kind": "not_equal",
            "value": blank_idx,
          },
          "output_wo_b": {
            "class": "masked_computation",
            "from": "output",
            "mask": "output_non_blank",
            "unit": {"class": "copy"},
          },
        })
        seg_net_dict["decision"]["from"] = "output_wo_b"

      seg_net_dict["output"].update({
        "back_prop": True if task == "train" else False,
        "class": "rec",
        "from": "encoder",
        "include_eos": True,
        "size_target": "targetb" if task == "train" else None,
        "target": target,
      })

      seg_net_dict["output"]["unit"].update(
        {
          "am": {"class": "copy", "from": "data:source"},
          "blank_log_prob": {
            "class": "eval",
            "eval": "tf.math.log_sigmoid(-source(0))",
            "from": "emit_prob0",
          },
          "const1": {"class": "constant", "value": 1},
          "emit_log_prob": {
            "activation": "log_sigmoid",
            "class": "activation",
            "from": "emit_prob0",
          },
          "emit_prob0": {
            "activation": None,
            "class": "linear",
            "from": "s",
            "is_output_layer": True,
            "n_out": 1,
          },
          "output_emit": {
            "class": "compare",
            "from": "output",
            "initial_output": True,
            "kind": "not_equal",
            "value": blank_idx,
          },
          "prev_out_embed": {
            "activation": None,
            "class": "linear",
            "from": "prev:output",
            "n_out": 128,
          },
          "s": {
            "L2": 0.0001,
            "class": "rec",
            "dropout": 0.3,
            "from": ["am", "prev_out_embed"],
            "n_out": 128,
            "unit": "nativelstm2",
            "unit_opts": {"rec_weight_dropout": 0.3},
          },
        },
      )

      length_scale = network_opts.get("length_scale")
      if type(length_scale) == float and length_scale != 1.0:
        # scale both blank and emit log prob by a constant factor
        seg_net_dict["output"]["unit"].update({
          "blank_log_prob": {
            "class": "eval",
            "eval": "tf.math.log_sigmoid(-source(0)) * %f" % length_scale,
            "from": "emit_prob0",
          },
          "emit_log_prob": {
            "class": "eval",
            "eval": "tf.math.log_sigmoid(source(0)) * %f" % length_scale,
            "from": "emit_prob0",
          },
        })

      blank_penalty = network_opts.get("blank_penalty")
      if type(blank_penalty) == float and blank_penalty != 0.0:
        # add constant penalty to blank log prob
        seg_net_dict["output"]["unit"]["blank_log_prob"]["eval"] += (" - %f" % blank_penalty)

    def _add_att_weight_aux_train_loss(rec_layer_name: str):
      raise NotImplementedError
    
    def _add_att_weight_recog_penalty(rec_layer_name: str, opts):
      mult_weight = opts["mult_weight"]
      exp_weight = opts["exp_weight"]

      network_builder.add_center_positions(network=seg_net_dict)

      seg_net_dict[rec_layer_name]["unit"].update({
        "segment_abs_positions0": {
          "class": "range_in_axis",
          "from": "att_weights",
          "axis": "stag:sliced-time:segments",
        },
        "segment_abs_positions": {
          "class": "eval",
          "from": ["segment_abs_positions0", "segment_starts"],
          "eval": "source(0) + source(1)"
        },
        "weighted_segment_abs_positions": {
          "class": "eval",
          "from": ["segment_abs_positions", "att_weights"],
          "eval": "tf.cast(source(0), tf.float32) * source(1)"
        },
        "att_weights_center_of_gravity": {
          "class": "reduce",
          "mode": "sum",
          "from": "weighted_segment_abs_positions",
          "axis": "stag:sliced-time:segments"
        },
        "att_weight_penalty": {
          "class": "eval",
          "from": ["att_weights_center_of_gravity", "center_positions"],
          "eval": "-%f * tf.math.abs(source(0) - tf.cast(source(1), tf.float32)) ** %f" % (mult_weight, exp_weight)
        },
      })

      if rec_layer_name == "label_model":
        # raise NotImplementedError
        assert "output_prob" in seg_net_dict[rec_layer_name]["unit"]
        seg_net_dict[rec_layer_name]["unit"]["output_log_prob0"] = network_builder.get_output_log_prob("output_prob")
        seg_net_dict[rec_layer_name]["unit"]["output_log_prob"] = {
          "class": "eval",
          "from": ["output_log_prob", "att_weight_penalty"],
          "eval": "source(0) + source(1)"
        }
      else:
        seg_net_dict[rec_layer_name]["unit"].update({
          "label_log_prob_w_penalty": {
            "class": "eval",
            "from": ["label_log_prob", "att_weight_penalty"],
            "eval": "source(0) + source(1)"
          },
        })
        seg_net_dict[rec_layer_name]["unit"]["output_log_prob"]["from"] = ["label_log_prob_w_penalty", "blank_log_prob"]

    def _add_gaussian_att_weight_interpolation(rec_layer_name: str):
      raise NotImplementedError

    def _add_weight_feedback(rec_layer_name: str):
      if network_opts["use_weight_feedback"]:
        config_dict["att_t_dim_tag"] = SegmentalConfigBuilder.get_att_t_dim_tag_code_wrapper()
        config_dict["att_t_overlap_dim_tag"] = SegmentalConfigBuilder.get_att_t_overlap_dim_tag_code_wrapper()
        config_dict["accum_att_weights_dim_tag"] = SegmentalConfigBuilder.get_accum_att_weights_dim_tag_code_wrapper(
          window_size=network_opts["segment_center_window_size"]
        )

        python_prolog.append("from returnn.tensor import batch_dim")

        if task == "train":
          prev_segment_starts_name = "prev:segment_starts"
          prev_segment_lens_name = "prev:segment_lens"
          prev_accum_att_weights_name = "prev:accum_att_weights"
        else:
          prev_segment_starts_name = "prev_segment_starts"
          prev_segment_lens_name = "prev_segment_lens"
          prev_accum_att_weights_name = "prev_accum_att_weights_masked"

          seg_net_dict[rec_layer_name]["unit"].update({
            "prev_segment_starts": {
              "class": "unmask", "from": "prev_segment_starts_masked", "mask": "prev:output_emit"},
            "prev_segment_starts_masked": {
              "class": "masked_computation",
              "from": "prev:segment_starts",
              "mask": "prev:output_emit",
              "unit": {
                "class": "subnetwork",
                "from": "data",
                "subnetwork": {
                  "output": {"class": "copy", "from": "data"},
                },
              },
            },
            "prev_segment_lens": {"class": "unmask", "from": "prev_segment_lens_masked", "mask": "prev:output_emit"},
            "prev_segment_lens_masked": {
              "class": "masked_computation",
              "from": "prev:segment_lens",
              "mask": "prev:output_emit",
              "unit": {
                "class": "subnetwork",
                "from": "data",
                "subnetwork": {
                  "output": {"class": "copy", "from": "data"},
                },
              },
            },
            # "prev_accum_att_weights": {
            #   "class": "unmask", "from": "prev_accum_att_weights_masked", "mask": "prev:output_emit"},
            "prev_accum_att_weights_masked": {
              "class": "masked_computation",
              "from": "prev:accum_att_weights",
              "mask": "prev:output_emit",
              "unit": {
                "class": "subnetwork",
                "from": "data",
                "subnetwork": {
                  "output": {"class": "copy", "from": "data"},
                },
              },
            },
          })

        seg_net_dict[rec_layer_name]["unit"].update({
          "overlap_len0": {
            "class": "eval",
            "from": [prev_segment_starts_name, prev_segment_lens_name, "segment_starts"],
            "eval": "source(0) + source(1) - source(2)"
          },
          "overlap_mask": {
            "class": "compare",
            "from": "overlap_len0",
            "value": 0,
            "kind": "less"
          },
          "overlap_len": {
            "class": "switch",
            "condition": "overlap_mask",
            "true_from": 0,
            "false_from": "overlap_len0"
          },
          "overlap_range": {
            "class": "range_in_axis",
            "from": "overlap_accum_weights",
            "axis": CodeWrapper("att_t_overlap_dim_tag"),
          },
          "att_weights_range": {
            "class": "range_in_axis",
            "from": "att_weights",
            "axis": CodeWrapper("att_t_dim_tag"),
          },
          "overlap_start": {
            "class": "combine",
            "from": [prev_segment_lens_name, "overlap_len"],
            "kind": "sub"
          },
          "overlap_accum_weights": {
            "class": "slice_nd",
            "from": prev_accum_att_weights_name,
            "start": "overlap_start",
            "size": "overlap_len",
            "axis": CodeWrapper("accum_att_weights_dim_tag"),
            "out_spatial_dim": CodeWrapper("att_t_overlap_dim_tag"),
            "initial_output": 0.
          },
          "accum_att_weights_scattered0": {
            "class": "scatter_nd",
            "from": "overlap_accum_weights",
            "position": "overlap_range",
            "position_axis": CodeWrapper("att_t_overlap_dim_tag"),
            "out_spatial_dim": CodeWrapper("accum_att_weights_dim_tag"),
          },
          "accum_att_weights_scattered": {
            "class": "reinterpret_data",
            "from": "accum_att_weights_scattered0",
            "enforce_batch_major": True
          },
          "att_weights_scattered": {
            "class": "scatter_nd",
            "from": "att_weights",
            "position": "att_weights_range",
            "position_axis": CodeWrapper("att_t_dim_tag"),
            "out_spatial_dim": CodeWrapper("accum_att_weights_dim_tag"),
          },
          "inv_fertility_scattered": {
            "class": "scatter_nd",
            "from": "inv_fertility",
            "position": "att_weights_range",
            "position_axis": CodeWrapper("att_t_dim_tag"),
            "out_spatial_dim": CodeWrapper("accum_att_weights_dim_tag"),
          },
          "inv_fertility": {
            "class": "slice_nd",
            "from": "base:inv_fertility",
            "start": "segment_starts",
            "size": "segment_lens",
            "out_spatial_dim": CodeWrapper("att_t_dim_tag")
          },
          'accum_att_weights0': {
            'class': 'eval',
            'eval': 'source(0) + source(1) * source(2) * 0.5',
            'from': ['accum_att_weights_scattered', 'att_weights_scattered', 'inv_fertility_scattered'],
            "initial_output": "base:initial_output_layer"
          },
          "accum_att_weights": {
            "class": "reinterpret_data",
            "from": "accum_att_weights0",
            "enforce_batch_major": True,
          },
          "prev_accum_att_weights_sliced": {
            "class": "slice_nd",
            "from": "accum_att_weights_scattered",
            "start": 0,
            "size": "segment_lens",
            "axis": CodeWrapper("accum_att_weights_dim_tag"),
            "out_spatial_dim": CodeWrapper("att_t_dim_tag")
          },
          "weight_feedback": {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": "prev_accum_att_weights_sliced",
            "n_out": 1024,
          },
        })

        seg_net_dict[rec_layer_name]["unit"]["segments"]["out_spatial_dim"] = CodeWrapper("att_t_dim_tag")

        seg_net_dict.update({
          "initial_output_layer": {
            "class": "constant",
            "value": 0.,
            "shape": [
              CodeWrapper("batch_dim"),
              CodeWrapper("accum_att_weights_dim_tag"),
            ]
          },
        })

      else:
        seg_net_dict[rec_layer_name]["unit"].update({
          "weight_feedback": {
            "class": "constant",
            "value": 0,
            "dtype": "float32",
            "with_batch_dim": True
          }
        })

    def _add_segments():
      seg_net_dict["output"]["unit"].update(
        network_builder.get_segment_starts_and_lengths(network_opts["segment_center_window_size"]))

    def _add_positional_embedding(rec_layer_name: str):
      if task == "train":
        seg_net_dict.update({
          'segment_starts1_masked': {
            'class': 'masked_computation',
            'from': 'output/segment_starts1',
            'mask': 'is_label',
            'register_as_extern_data': 'segment_starts1_masked',
            'unit': {
              'class': 'copy',
              'from': 'data'
            }
          },
        })
        seg_net_dict["label_model"]["unit"].update({
          'segment_starts1': {
            'axis': 't',
            'class': 'gather',
            'from': 'base:data:segment_starts1_masked',
            'position': ':i',
            "initial_output": -1
          },
        })

      seg_net_dict[rec_layer_name]["unit"].update({
        "segment_start_delta0": {
          "class": "combine",
          "from": ["segment_starts1", "prev:segment_starts1"],
          "kind": "sub"
        },
        "segment_start_delta_max_mask": {
          "class": "compare",
          "from": "segment_start_delta0",
          "value": 20,
          "kind": "greater"
        },
        "segment_start_delta1": {
          "class": "switch",
          "condition": "segment_start_delta_max_mask",
          "true_from": 20,
          "false_from": "segment_start_delta0"
        },
        "segment_start_delta2": {
          "class": "eval",
          "from": ["segment_start_delta1"],
          "eval": "source(0) - 1"
        },
        "segment_start_delta": {
          "class": "reinterpret_data",
          "from": "segment_start_delta2",
          "set_sparse_dim": 20,
          "set_sparse": True
        },
        'segment_start_delta_embed': {
          'class': 'linear',
          'activation': None,
          'with_bias': False,
          'from': 'segment_start_delta',
          'n_out': 128,
          'initial_output': 0
        },
      })

      if task == "train":
        seg_net_dict["label_model"]["unit"]["s"]["from"].append("segment_start_delta_embed")
      else:
        seg_net_dict["output"]["unit"]["lm_masked"]["unit"]["subnetwork"]["lm"]["from"].append("base:segment_start_delta_embed")

    seg_net_dict = copy.deepcopy(global_net_dict)

    if task == "train":
      rec_layer_name = "label_model"
      _remove_not_needed_layers()
      _add_base_layers()
      _add_label_model_layer()
      _add_output_layer()
      _add_label_model_att_layers(rec_layer_name)
      _add_weight_feedback(rec_layer_name)
      _add_segments()
    else:
      rec_layer_name = "output"
      _remove_not_needed_layers()
      _add_label_model_att_layers(rec_layer_name)
      _add_weight_feedback(rec_layer_name)
      _add_output_layer()
      _add_segments()

    if network_opts.get("use_positional_embedding"):
      _add_positional_embedding(rec_layer_name)

    if network_opts.get("use_att_weight_aux_train_loss"):
      _add_att_weight_aux_train_loss(rec_layer_name)

    att_weights_recog_penalty_opts = network_opts.get("att_weight_recog_penalty_opts")
    if att_weights_recog_penalty_opts:
      _add_att_weight_recog_penalty(rec_layer_name, att_weights_recog_penalty_opts)

    if network_opts.get("use_gaussian_att_weight_interpolation"):
      _add_gaussian_att_weight_interpolation(rec_layer_name)

    return seg_net_dict
