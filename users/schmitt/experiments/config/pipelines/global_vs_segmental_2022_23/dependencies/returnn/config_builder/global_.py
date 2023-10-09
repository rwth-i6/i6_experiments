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
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_to_segmental import MohammadGlobalAttToSegmentalAttentionMaker
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from sisyphus import Path

import os
import re
from abc import abstractmethod, ABC
from typing import Dict, Optional, List
import copy
import numpy as np


class GlobalConfigBuilder(ConfigBuilder, ABC):
  def __init__(self, dependencies: SegmentalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  def get_dump_att_weight_config(self, corpus_key: str, opts: Dict):
    returnn_config = self.get_eval_config(eval_corpus_key=corpus_key, opts=opts)

    returnn_config.config["network"]["output"]["unit"]["att_weights"]["is_output_layer"] = True

    hdf_filenames = opts["hdf_filenames"]
    returnn_config.config["network"].update({
      "att_weights_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["att_weights"],
        "from": "output/att_weights",
        "is_output_layer": True,
      },
      "targets_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["targets"],
        "from": "data:targets",
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
    # forward_recog_config.config["network"]["dump_decision"] = {
    #   "class": "hdf_dump",
    #   "from": "decision",
    #   "is_output_layer": True,
    #   "filename": "search_out.hdf"
    # }
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
    forward_recog_config.config["network"]["decision"]["from"] = "output_w_beam"

    return forward_recog_config

  def get_dump_scores_config(self, corpus_key: str, opts: Dict):
    returnn_config = self.get_eval_config(eval_corpus_key=corpus_key, opts=opts)

    if "output_log_prob" not in returnn_config.config["network"]["output"]["unit"]:
      returnn_config.config["network"]["output"]["unit"]["output_log_prob"] = network_builder.get_output_log_prob(output_prob_layer_name="output_prob")

    returnn_config.config["network"]["output"]["unit"].update({
      "gather_output_log_prob": {
        "class": "gather",
        "from": "output_log_prob",
        "axis": "f",
        "position": "base:data:targets",
        "is_output_layer": True
      },
    })

    hdf_filenames = opts["hdf_filenames"]

    returnn_config.config["network"].update({
      "label_model_log_scores_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["label_model_log_scores"],
        "from": "output/gather_output_log_prob",
        "is_output_layer": True,
      },
      "targets_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["targets"],
        "from": "data:targets",
        "is_output_layer": True,
      },
    })

    returnn_config.config["forward_batch_size"] = CodeWrapper("batch_size")

    return returnn_config

  def edit_network_only_train_length_model(self, net_dict: Dict):
    raise NotImplementedError


class SWBBlstmGlobalAttentionConfigBuilder(GlobalConfigBuilder, SWBBlstmConfigBuilder, ConfigBuilder):
  def get_default_lr_opts(self):
    return {
      "type": "newbob",
      "learning_rate_control_error_measure": "dev_error_output/label_prob"
    }

  def get_net_dict(self, task: str, config_dict, python_prolog):
    net_dict = {}
    net_dict.update(network_builder.get_info_layer(global_att=True))
    net_dict.update(network_builder.get_source_layers(from_layer="data:data"))
    net_dict.update(network_builder.get_blstm_encoder())
    net_dict.update(network_builder.get_enc_ctx_and_val())
    net_dict["output"] = {
      "class": "rec",
      "max_seq_len": "max_len_from('base:encoder')",
      "target": "target_w_eos" if task == "train" else "targets",
      "unit": network_builder.get_label_model_unit_dict(global_attention=True, task=task)
    }
    if task == "train":
      net_dict.update(network_builder.get_target_w_eos())
      net_dict.update(network_builder.get_ctc_loss(global_att=True))
    else:
      net_dict.update(network_builder.get_decision_layer(global_att=True))

    return copy.deepcopy(net_dict)

  def get_networks_dict(self, task: str, config_dict, python_prolog):
    return None

  def get_custom_construction_algo(self, config_dict, python_prolog):
    python_prolog.append(custom_construction_algos.custom_construction_algo_global_att)
    config_dict["pretrain"] = {
      "construction_algo": CodeWrapper("custom_construction_algo_global_att"),
      "copy_param_mode": "subset",
      "repetitions": 1
    }


class SWBConformerGlobalAttentionConfigBuilder(SwbConformerConfigBuilder, GlobalConfigBuilder, ConformerConfigBuilder, ConfigBuilder):
  def get_net_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      return None
    else:
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.network_builder.mohammad_conformer.networks_11_4 import networks_dict

      network_dict = copy.deepcopy(networks_dict[22])
      return network_dict

  def get_networks_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.network_builder.mohammad_conformer.networks_11_4 import networks_dict
      return copy.deepcopy(networks_dict)
    else:
      return None


class LibrispeechConformerGlobalAttentionConfigBuilder(GlobalConfigBuilder, LibrispeechConformerConfigBuilder, ConformerConfigBuilder, ConfigBuilder):
  def get_net_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      return None
    else:
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict

      network_dict = copy.deepcopy(networks_dict[36])
      if not self.variant_params["network"].get("use_weight_feedback", True):
        # replace the weight feedback with a 0 constant: same as if there was no weight feedback
        network_dict["output"]["unit"]["weight_feedback"] = {
          "class": "constant",
          "dtype": "float32",
          "value": 0,
          "with_batch_dim": True,
        }

      return network_dict

  def get_networks_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict
      return copy.deepcopy(networks_dict)
    else:
      return None
