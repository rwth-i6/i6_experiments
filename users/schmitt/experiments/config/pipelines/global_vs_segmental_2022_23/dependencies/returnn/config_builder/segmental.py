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
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import ConfigBuilder, SWBBlstmConfigBuilder, SwbConformerConfigBuilder, LibrispeechConformerConfigBuilder, ConformerConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from sisyphus import Path

import os
import re
from abc import abstractmethod, ABC
from typing import Dict, Optional, List
import copy
import numpy as np


class SegmentalConfigBuilder(ConfigBuilder, ABC):
  def __init__(self, dependencies: SegmentalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  def get_train_config(self, opts: Dict, python_epilog: Optional[Dict] = None):
    python_epilog = copy.deepcopy(self.python_epilog if python_epilog is None else python_epilog)

    hdf_targets = self.dependencies.hdf_targets
    if "dataset_opts" in opts:
      hdf_targets = opts["dataset_opts"].get("hdf_targets", hdf_targets)

    assert hdf_targets != {} and "train" in hdf_targets and "cv" in hdf_targets and "devtrain" in hdf_targets, (
      "You need to provide HDF targets to train a segmental model"
    )

    if opts.get("chunking", None):
      chunk_size_targets = opts["chunking"]["chunk_size_targets"]
      chunk_step_targets = opts["chunking"]["chunk_step_targets"]
      chunk_size_data = opts["chunking"]["chunk_size_data"]
      chunk_step_data = opts["chunking"]["chunk_step_data"]
      python_epilog += [
        "from returnn.util.basic import NumbersDict",
        "chunk_size = NumbersDict({'targets': %s, 'data': %s})" % (chunk_size_targets, chunk_size_data),
        "chunk_step = NumbersDict({'targets': %s, 'data': %s})" % (chunk_step_targets, chunk_step_data),
        custom_chunkin_func_str.format(blank_idx=self.dependencies.model_hyperparameters.blank_idx)
      ]

    return super().get_train_config(opts=opts, python_epilog=python_epilog)

  def edit_network_only_train_length_model(self, net_dict: Dict):
    if "class" in net_dict:
      net_dict["trainable"] = False

    for item in net_dict:
      if type(net_dict[item]) == dict and item != "output":
        self.edit_network_only_train_length_model(net_dict[item])

  def get_dump_scores_config(self, corpus_key: str, opts: Dict):
    returnn_config = self.get_eval_config(eval_corpus_key=corpus_key, opts=opts)

    if "output_log_prob" not in returnn_config.config["network"]["label_model"]["unit"]:
      returnn_config.config["network"]["label_model"]["unit"]["output_log_prob"] = network_builder.get_output_log_prob(output_prob_layer_name="output_prob")

    if "att_weight_penalty" in returnn_config.config["network"]["label_model"]["unit"]:
      output_log_prob_layer_name = "output_log_prob0"
    else:
      output_log_prob_layer_name = "output_log_prob"

    returnn_config.config["network"]["label_model"]["unit"].update({
      "gather_output_log_prob": {
        "class": "gather",
        "from": output_log_prob_layer_name,
        "axis": "f",
        "position": "base:data:label_ground_truth",
        "is_output_layer": True
      },
    })

    returnn_config.config["network"]["output"]["unit"].update({
      "gather_emit_blank_log_prob": {
        "class": "gather",
        "from": "emit_blank_log_prob",
        "axis": "f",
        "position": "base:data:emit_ground_truth",
        "is_output_layer": True
      },
    })

    hdf_filenames = opts["hdf_filenames"]

    returnn_config.config["network"].update({
      "label_model_log_scores_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["label_model_log_scores"],
        "from": "label_model/gather_output_log_prob",
        "is_output_layer": True,
      },
      "length_model_log_scores_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["length_model_log_scores"],
        "from": "output/gather_emit_blank_log_prob",
        "is_output_layer": True,
      },
      "targets_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["targets"],
        "from": "data:targetb",
        "is_output_layer": True,
      },
    })

    if "att_weight_penalty_scores" in hdf_filenames:
      assert "att_weight_penalty" in returnn_config.config["network"]["label_model"]["unit"]
      returnn_config.config["network"]["label_model"]["unit"]["att_weight_penalty"]["is_output_layer"] = True
      returnn_config.config["network"].update({
        "att_weight_penalty_scores_dump": {
          "class": "hdf_dump",
          "filename": hdf_filenames["att_weight_penalty_scores"],
          "from": "label_model/att_weight_penalty",
          "is_output_layer": True,
        },
      })

    returnn_config.config["forward_batch_size"] = CodeWrapper("batch_size")

    return returnn_config

  def get_dump_att_weight_config(self, corpus_key: str, opts: Dict):
    returnn_config = self.get_eval_config(eval_corpus_key=corpus_key, opts=opts)

    assert opts.get("use_train_net", False), "Use train net to dump att weights!"
    if opts.get("use_train_net", False):
      rec_layer_name = "label_model"
    else:
      rec_layer_name = "output"

    returnn_config.config["network"][rec_layer_name]["unit"].update({
      "segment_range0": {
        "class": "range_in_axis",
        "from": "att_weights",
        "axis": "stag:sliced-time:segments"
      },
      "segment_range": {
        "class": "eval",
        "from": ["segment_range0", "segment_starts"],
        "eval": "source(0) + source(1)"
      },
      "att_weights_scattered_into_encoder": {
        "class": "scatter_nd",
        "from": "att_weights",
        "position": "segment_range",
        "position_axis": "stag:sliced-time:segments",
        "output_dim_via_time_from": "base:encoder",
        "is_output_layer": True
      },
    })

    hdf_filenames = opts["hdf_filenames"]

    returnn_config.config["network"].update({
      "att_weights_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["att_weights"],
        "from": "%s/att_weights_scattered_into_encoder" % rec_layer_name,
        "is_output_layer": True,
      },
      "targets_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["targets"],
        "from": "data:targetb",
        "is_output_layer": True,
      },
      "seg_starts_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["seg_starts"],
        "from": "segment_starts_masked",
        "is_output_layer": True,
      },
      "seg_lens_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["seg_lens"],
        "from": "segment_lens_masked",
        "is_output_layer": True,
      },
    })

    if "center_positions" in hdf_filenames:
      # additionally dump the center positions into hdf
      network_builder.add_center_positions(network=returnn_config.config["network"])
      returnn_config.config["network"].update({
        "center_positions_dump": {
          "class": "hdf_dump",
          "filename": hdf_filenames["center_positions"],
          "from": "center_positions_masked",
          "is_output_layer": True,
        },
      })

    if "att_weight_penalty" in hdf_filenames:
      assert "att_weight_penalty" in returnn_config.config["network"][rec_layer_name]["unit"], "att_weight_penalty does not exist in network!"

      returnn_config.config["network"][rec_layer_name]["unit"]["att_weight_penalty"]["is_output_layer"] = True
      returnn_config.config["network"].update({
        "att_weight_penalty_dump": {
          "class": "hdf_dump",
          "filename": hdf_filenames["att_weight_penalty"],
          "from": "label_model/att_weight_penalty",
          "is_output_layer": True,
        },
      })

    returnn_config.config["forward_batch_size"] = CodeWrapper("batch_size")

    return returnn_config

  def get_recog_config_for_forward_job(self, opts: Dict):
    forward_recog_config = self.get_recog_config(opts)

    forward_recog_config.config.update({
      "forward_use_search": True,
      "forward_batch_size": CodeWrapper("batch_size")
    })
    forward_recog_config.config["network"]["dump_decision"] = {
      "class": "hdf_dump",
      "from": "decision",
      "is_output_layer": True,
      "filename": "search_out.hdf"
    }
    del forward_recog_config.config["task"]
    forward_recog_config.config["eval"] = self.get_search_dataset_dict(
      corpus_key=opts["search_corpus_key"],
      dataset_opts=opts.get("dataset_opts", {})
    )
    del forward_recog_config.config["search_data"]
    forward_recog_config.config["network"]["output_w_beam"] = copy.deepcopy(
      forward_recog_config.config["network"]["output"])
    forward_recog_config.config["network"]["output_w_beam"]["name_scope"] = "output/rec"
    del forward_recog_config.config["network"]["output"]
    forward_recog_config.config["network"]["output"] = copy.deepcopy(
      forward_recog_config.config["network"]["decision"])
    forward_recog_config.config["network"]["output"]["from"] = "output_w_beam"
    forward_recog_config.config["network"]["output_non_blank"]["from"] = "output_w_beam"
    forward_recog_config.config["network"]["output_wo_b"]["from"] = "output_w_beam"

    return forward_recog_config

  def add_align_augment(self, net_dict, networks_dict, python_prolog):
    python_prolog.append(shift_alignment_boundaries_func_str.format(
      blank_idx=self.dependencies.model_hyperparameters.blank_idx,
      max_shift=2
    ))

    def _add_align_augment_layer(network_dict):
      network_dict.update({
        "existing_alignment0": copy.deepcopy(network_dict["existing_alignment"]),
        "existing_alignment": {
          "class": "eval",
          "from": "existing_alignment0",
          "eval": "self.network.get_config().typed_value('shift_alignment_boundaries')(source(0, as_data=True), network=self.network)"
        }
      })

    if net_dict is not None:
      _add_align_augment_layer(net_dict)
    if networks_dict is not None:
      for net_dict in networks_dict:
        _add_align_augment_layer(net_dict)

  @staticmethod
  def get_att_t_dim_tag_code_wrapper():
    return CodeWrapper(
      'DimensionTag(kind=DimensionTag.Types.Spatial, description="sliced-time:segments", dimension=None)')

  @staticmethod
  def get_att_t_overlap_dim_tag_code_wrapper():
    return CodeWrapper(
      'DimensionTag(kind=DimensionTag.Types.Spatial, description="sliced-time:overlap_accum_weights", dimension=None)')

  @staticmethod
  def get_accum_att_weights_dim_tag_code_wrapper(window_size: int):
    return CodeWrapper(
      'DimensionTag(kind=DimensionTag.Types.Spatial, description="accum_att_weights", dimension=%d)' % window_size)


class SWBBlstmSegmentalAttentionConfigBuilder(SegmentalConfigBuilder, SWBBlstmConfigBuilder, ConfigBuilder):
  def get_default_lr_opts(self):
    return {
      "type": "newbob",
      "learning_rate_control_error_measure": "dev_error_label_model/label_prob"
    }

  def get_net_dict(self, task: str, config_dict, python_prolog):
    net_dict = {}
    net_dict.update(network_builder.get_info_layer(global_att=False))
    net_dict.update(network_builder.get_source_layers(from_layer="data:data"))
    net_dict.update(network_builder.get_blstm_encoder())
    if task == "train":
      net_dict.update(network_builder.get_existing_alignment_layer())
      net_dict.update(network_builder.get_is_label_layer())
      net_dict.update(network_builder.get_label_ground_truth_target())
      net_dict.update(network_builder.get_targetb_target())
      net_dict.update(network_builder.get_emit_ground_truth_target())
      net_dict.update(network_builder.get_masked_segment_starts_and_lengths())
      net_dict.update(network_builder.get_ctc_loss(global_att=False))
      net_dict["output"] = {
        "class": "rec",
        "from": "encoder",
        "include_eos": True,
        "size_target": "targetb",
        "target": "targetb",
        "unit": network_builder.get_length_model_unit_dict(task="train")
      }
      net_dict["label_model"] = {
        "class": "rec",
        "is_output_layer": True,
        "from": "data:label_ground_truth",
        "name_scope": "output/rec",
        "include_eos": True,
        "unit": network_builder.get_label_model_unit_dict(global_attention=False, task="train")
      }
    else:
      net_dict["output"] = {
        "class": "rec",
        "from": "encoder",
        "include_eos": True,
        "size_target": None,
        "target": "targets",
        "unit": network_builder.get_length_model_unit_dict(task=task)
      }
      net_dict["output"]["unit"].update({
        network_builder.get_label_model_unit_dict(global_attention=False, task=task)
      })
      net_dict.update(network_builder.get_decision_layer(global_att=False))

    net_dict["output"]["unit"].update(
      network_builder.get_segment_starts_and_lengths(
        segment_center_window_size=self.variant_params["network"]["segment_center_window_size"]))

    return copy.deepcopy(net_dict)

  def get_networks_dict(self, task: str, config_dict, python_prolog):
    return None

  def get_custom_construction_algo(self, config_dict, python_prolog):
    python_prolog.append(custom_construction_algos.custom_construction_algo_segmental_att)
    config_dict["pretrain"] = {
      "construction_algo": CodeWrapper("custom_construction_algo_segmental_att"),
      "copy_param_mode": "subset",
      "repetitions": 1
    }


class SWBConformerSegmentalAttentionConfigBuilder(SwbConformerConfigBuilder, SegmentalConfigBuilder, ConformerConfigBuilder, ConfigBuilder):
  def get_net_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      return None
    else:
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.network_builder.mohammad_conformer.networks_11_4 import networks_dict
      return MohammadGlobalAttToSegmentalAttentionMaker.make_global_attention_segmental(
        copy.deepcopy(networks_dict[22]),
        task=task,
        blank_idx=self.dependencies.model_hyperparameters.blank_idx,
        target_num_labels_w_blank=self.dependencies.model_hyperparameters.target_num_labels,
        target_num_labels_wo_blank=self.dependencies.model_hyperparameters.target_num_labels_wo_blank,
        network_opts=self.variant_params["network"],
        config_dict=config_dict,
        python_prolog=python_prolog
      )

  def get_networks_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.network_builder.mohammad_conformer.networks_11_4 import networks_dict

      new_networks_dict = {}
      for i, net_dict in networks_dict.items():
        new_networks_dict[i] = MohammadGlobalAttToSegmentalAttentionMaker.make_global_attention_segmental(
          copy.deepcopy(net_dict),
          task=task,
          blank_idx=self.dependencies.model_hyperparameters.blank_idx,
          target_num_labels_w_blank=self.dependencies.model_hyperparameters.target_num_labels,
          target_num_labels_wo_blank=self.dependencies.model_hyperparameters.target_num_labels_wo_blank,
          network_opts=self.variant_params["network"],
          config_dict=config_dict,
          python_prolog=python_prolog
        )
      return new_networks_dict
    else:
      return None


class LibrispeechConformerSegmentalAttentionConfigBuilder(SegmentalConfigBuilder, ConformerConfigBuilder, LibrispeechConformerConfigBuilder, ConfigBuilder):
  def __init__(self, dependencies: SegmentalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  def get_net_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      return None
    else:
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict
      return MohammadGlobalAttToSegmentalAttentionMaker.make_global_attention_segmental(
        copy.deepcopy(networks_dict[36]),
        task=task,
        blank_idx=self.dependencies.model_hyperparameters.blank_idx,
        target_num_labels_w_blank=self.dependencies.model_hyperparameters.target_num_labels,
        target_num_labels_wo_blank=self.dependencies.model_hyperparameters.target_num_labels_wo_blank,
        network_opts=self.variant_params["network"],
        config_dict=config_dict,
        python_prolog=python_prolog
      )

  def get_networks_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict

      new_networks_dict = {}
      for i, net_dict in networks_dict.items():
        new_networks_dict[i] = MohammadGlobalAttToSegmentalAttentionMaker.make_global_attention_segmental(
          copy.deepcopy(net_dict),
          task=task,
          blank_idx=self.dependencies.model_hyperparameters.blank_idx,
          target_num_labels_w_blank=self.dependencies.model_hyperparameters.target_num_labels,
          target_num_labels_wo_blank=self.dependencies.model_hyperparameters.target_num_labels_wo_blank,
          network_opts=self.variant_params["network"],
          config_dict=config_dict,
          python_prolog=python_prolog
        )
      return new_networks_dict
    else:
      return None


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

    def _add_output_layer(length_model_opts: Dict):
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
        seg_net_dict["output"]["unit"]["target_embed_masked"]["unit"]["subnetwork"]["target_embed0"][
          "from"] = "output_non_blank"
        seg_net_dict["output"]["unit"]["target_embed_masked"]["unit"]["subnetwork"]["target_embed0"][
          "name_scope"] = "/output/rec/target_embed0"

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
            "from": ["am"],
            "n_out": 128,
            "unit": "nativelstm2",
            "unit_opts": {"rec_weight_dropout": 0.3},
          },
        },
      )

      if length_model_opts["use_embedding"]:
        assert "embedding_size" in length_model_opts
        seg_net_dict["output"]["unit"].update({
            "prev_out_embed": {
              "activation": None,
              "class": "linear",
              "from": "prev:output",
              "n_out": length_model_opts["embedding_size"],
            },
          })
        seg_net_dict["output"]["unit"]["s"]["from"].append("prev_out_embed")

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
      network_builder.add_att_weights_center_of_gravity(network=seg_net_dict, rec_layer_name=rec_layer_name)

      seg_net_dict[rec_layer_name]["unit"].update({
        "att_weight_penalty": {
          "class": "eval",
          "from": ["att_weights_center_of_gravity", "center_positions"],
          "eval": "-%f * tf.math.abs(source(0) - tf.cast(source(1), tf.float32)) ** %f" % (mult_weight, exp_weight)
        },
      })

      if opts.get("use_as_loss"):
        assert "loss_scale" in opts
        seg_net_dict[rec_layer_name]["unit"].update({
          "att_weight_penalty_loss": {
            "class": "eval",
            "from": "att_weight_penalty",
            "eval": "-source(0)",  # penalty is negative, but we want to minimize loss, there -
            "loss": "as_is",
            "loss_opts": {"scale": opts["loss_scale"]}
          }
        })

      if rec_layer_name == "label_model":
        # raise NotImplementedError
        assert "output_prob" in seg_net_dict[rec_layer_name]["unit"]
        seg_net_dict[rec_layer_name]["unit"]["output_log_prob0"] = network_builder.get_output_log_prob("output_prob")
        seg_net_dict[rec_layer_name]["unit"]["output_log_prob"] = {
          "class": "eval",
          "from": ["output_log_prob0", "att_weight_penalty"],
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

    def _add_gaussian_att_weight_interpolation(rec_layer_name: str, opts: Dict):
      # just to make sure the network looks as we expect
      assert seg_net_dict[rec_layer_name]["unit"]["att_weights"]["class"] == "softmax_over_spatial"
      assert "att_weights0" not in seg_net_dict[rec_layer_name]["unit"]
      assert network_opts["segment_center_window_size"] is not None

      network_builder.add_center_positions(network=seg_net_dict)

      tf_gauss_str = "1.0 / ({std} * tf.sqrt(2 * 3.141592)) * tf.exp(-0.5 * ((tf.cast({range} - {mean}, tf.float32)) / {std}) ** 2)".format(
        std=opts["std"], mean="source(1)", range="source(0)"
      )
      gaussian_clip_window_size = 3

      seg_net_dict[rec_layer_name]["unit"].update({
        "att_weights0":  copy.deepcopy(seg_net_dict[rec_layer_name]["unit"]["att_weights"]),
        "gaussian_mask": {  # true, only in (gaussian_clip_window_size * 2 - 1) frames around center
          "class": "compare",
          "from": ["gaussian_start", "gaussian_range", "gaussian_end"],
          "kind": "less"
        },
        "gaussian_start": {
          "class": "eval",
          "from": "center_positions",
          "eval": "source(0) - %d" % gaussian_clip_window_size
        },
        "gaussian_end": {
          "class": "eval",
          "from": "center_positions",
          "eval": "source(0) + %d" % gaussian_clip_window_size
        },
        "gaussian_range0": {
          "class": "range_in_axis",
          "from": "att_weights0",
          "axis": "stag:sliced-time:segments",
        },
        "gaussian_range": {
          "class": "eval",
          "from": ["gaussian_range0", "segment_starts"],
          "eval": "source(0) + source(1)"
        },
        "gaussian0": {
          "class": "eval",
          "from": ["gaussian_range", "center_positions"],
          "eval": tf_gauss_str,
          "out_type": {"dtype": "float32"}
        },
        "gaussian1": {
          "class": "switch",
          "condition": "gaussian_mask",
          "true_from": "gaussian0",
          "false_from": CodeWrapper('float("-inf")')
        },
        "gaussian": {
          "class": "softmax_over_spatial",
          "axis": "stag:sliced-time:segments",
          "from": "gaussian1"
        },
        "att_weights": {
          "class": "eval",
          "from": ["att_weights0", "gaussian"],
          "eval": "{gauss_scale} * source(1) + (1 - {gauss_scale}) * source(0)".format(
            gauss_scale=opts["gauss_scale"])
        },
      })

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
        seg_net_dict["output"]["unit"]["lm_masked"]["unit"]["subnetwork"]["lm"]["from"].append(
          "base:segment_start_delta_embed")

    seg_net_dict = copy.deepcopy(global_net_dict)

    if task == "train":
      rec_layer_name = "label_model"
      _remove_not_needed_layers()
      _add_base_layers()
      _add_label_model_layer()
      _add_output_layer(length_model_opts=network_opts["length_model_opts"])
      _add_label_model_att_layers(rec_layer_name)
      _add_weight_feedback(rec_layer_name)
      _add_segments()
    else:
      rec_layer_name = "output"
      _remove_not_needed_layers()
      _add_label_model_att_layers(rec_layer_name)
      _add_weight_feedback(rec_layer_name)
      _add_output_layer(length_model_opts=network_opts["length_model_opts"])
      _add_segments()

    if network_opts.get("use_positional_embedding"):
      _add_positional_embedding(rec_layer_name)

    if network_opts.get("use_att_weight_aux_train_loss"):
      _add_att_weight_aux_train_loss(rec_layer_name)

    att_weights_recog_penalty_opts = network_opts.get("att_weight_recog_penalty_opts")
    if att_weights_recog_penalty_opts:
      _add_att_weight_recog_penalty(rec_layer_name, att_weights_recog_penalty_opts)

    gaussian_att_weight_interpolation_opts = network_opts.get("gaussian_att_weight_interpolation_opts")
    if gaussian_att_weight_interpolation_opts:
      _add_gaussian_att_weight_interpolation(rec_layer_name, gaussian_att_weight_interpolation_opts)

    return seg_net_dict

