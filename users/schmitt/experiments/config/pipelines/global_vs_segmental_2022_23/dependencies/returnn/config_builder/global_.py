from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder import network_builder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.lm import lm_irie, lstm_bpe_10k
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.lm import base as lm_base
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.ilm_correction import mini_att as mini_att_ilm_correction
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.ilm_correction import zero_att as zero_att_ilm_correction
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn import custom_construction_algos
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import ConfigBuilder, SWBBlstmConfigBuilder, SwbConformerConfigBuilder, LibrispeechConformerConfigBuilder, ConformerConfigBuilder

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from sisyphus import Path

import os
import re
from abc import abstractmethod, ABC
from typing import Dict, Optional, List
import copy
import numpy as np


class GlobalConfigBuilder(ConfigBuilder, ABC):
  def __init__(self, dependencies: GlobalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  def get_train_config(self, opts: Dict, python_epilog: Optional[Dict] = None):
    train_config = super().get_train_config(opts=opts, python_epilog=python_epilog)

    if opts.get("train_mini_lstm_opts") is not None:  # need to check for None because it can be {}
      mini_att_ilm_correction.add_mini_att(
        network=train_config.config["network"],
        rec_layer_name="output",
        train=True,
      )
      self.edit_network_freeze_layers_excluding(
        train_config.config["network"],
        layers_to_exclude=["mini_att_lstm", "mini_att"]
      )
      train_config.config["network"]["output"]["trainable"] = True
      train_config.config["network"]["output"]["unit"]["att"]["is_output_layer"] = True

      if opts["train_mini_lstm_opts"].get("use_se_loss", False):
        mini_att_ilm_correction.add_se_loss(
          network=train_config.config["network"],
          rec_layer_name="output",
        )

    return train_config

  def get_recog_config(self, opts: Dict):
    recog_config = super().get_recog_config(opts)

    ilm_correction_opts = opts.get("ilm_correction_opts", None)
    if ilm_correction_opts is not None:
      if ilm_correction_opts.get("type", "mini_att") == "mini_att":
        ilm_correction = mini_att_ilm_correction
        recog_config.config["preload_from_files"] = {
          "mini_lstm": {
            "filename": ilm_correction_opts["mini_att_checkpoint"],
            "prefix": "preload_",
          }
        }
      elif ilm_correction_opts.get("type", "mini_att") == "zero_att":
        ilm_correction = zero_att_ilm_correction
      else:
        raise NotImplementedError

      ilm_correction.add_ilm_correction(
        network=recog_config.config["network"],
        rec_layer_name="output",
        target_num_labels=self.dependencies.model_hyperparameters.target_num_labels,
        opts=ilm_correction_opts,
        label_prob_layer="output_prob",
        use_mask_layer=False,
        returnn_config=recog_config
      )

    lm_opts = opts.get("lm_opts", None)
    if lm_opts is not None:
      if lm_opts.get("type", "trafo") == "trafo":
        get_lm_dict_func = lm_irie.get_lm_dict
        lm_embedding_layer_name = "target_embed_raw"
      else:
        assert lm_opts["type"] == "lstm"
        get_lm_dict_func = lstm_bpe_10k.get_lm_dict
        lm_embedding_layer_name = "input"

      lm_base.add_lm(
        network=recog_config.config["network"],
        rec_layer_name="output",
        target_num_labels=self.dependencies.model_hyperparameters.target_num_labels,
        opts=lm_opts,
        label_prob_layer="output_prob",
        get_lm_dict_func=get_lm_dict_func,
        lm_embedding_layer_name=lm_embedding_layer_name
      )

    return recog_config

  def get_compile_tf_graph_config(self, opts: Dict):
    returnn_config = self.get_recog_config(opts)
    del returnn_config.config["network"]["output"]["max_seq_len"]

    return returnn_config

  def get_dump_att_weight_config(self, corpus_key: str, opts: Dict):
    returnn_config = self.get_eval_config(eval_corpus_key=corpus_key, opts=opts)

    returnn_config.config["network"]["output"]["unit"]["att_weights"]["is_output_layer"] = True

    hdf_filenames = opts["hdf_filenames"]
    returnn_config.config["network"].update({
      # att weights
      "att_weights_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["att_weights"],
        "from": "output/att_weights",
        "is_output_layer": True,
      },
      # output labels
      "targets_dump": {
        "class": "hdf_dump",
        "filename": hdf_filenames["targets"],
        "from": "data:targets",
        "is_output_layer": True,
      },
    })
    # ctc alignment
    if "ctc_alignment" in hdf_filenames:
      returnn_config.config["network"].update(network_builder.get_ctc_forced_align_hdf_dump(
        align_target="data:targets",
        filename=hdf_filenames["ctc_alignment"]
      ))

    returnn_config.config["forward_batch_size"] = CodeWrapper("batch_size")

    return returnn_config

  def get_dump_length_model_probs_config(self, corpus_key: str, opts: Dict):
    raise NotImplementedError

  def get_recog_config_for_forward_job(self, opts: Dict):
    forward_recog_config = self.get_recog_config(opts)

    forward_recog_config.config.update({
      "forward_use_search": True,
      "forward_batch_size": CodeWrapper("batch_size")
    })

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

  def edit_network_modify_decoder(self, version: int, net_dict: Dict, train: bool, target_num_labels: int):
    network_builder.modify_decoder(version, net_dict, "output", target_num_labels, False, train)


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

  def get_networks_dict(self, task: str, config_dict, python_prolog, use_get_global_config: bool = False):
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
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.network_dicts.zeineldeen_swb_global_att_11_4 import networks_dict

      network_dict = copy.deepcopy(networks_dict[22])
      return network_dict

  def get_networks_dict(self, task: str, config_dict, python_prolog, use_get_global_config: bool = False):
    if task == "train":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.network_dicts.zeineldeen_swb_global_att_11_4 import networks_dict
      return copy.deepcopy(networks_dict)
    else:
      return None


class LibrispeechConformerGlobalAttentionConfigBuilder(GlobalConfigBuilder, LibrispeechConformerConfigBuilder, ConformerConfigBuilder, ConfigBuilder):
  def get_net_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      return None
    else:
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.network_dicts.zeineldeen_ls_global_att_5_6 import networks_dict

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

  def get_networks_dict(self, task: str, config_dict, python_prolog, use_get_global_config: bool = False):
    if task == "train":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder.network_dicts.zeineldeen_ls_global_att_5_6 import networks_dict
      return copy.deepcopy(networks_dict)
    else:
      return None
