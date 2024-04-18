from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import ConfigBuilder, LibrispeechConformerConfigBuilder, ConformerConfigBuilder

from abc import ABC
from typing import Dict, Optional
import copy


class CtcConfigBuilder(ConfigBuilder, ABC):
  def __init__(self, dependencies: GlobalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  def get_train_config(self, opts: Dict, python_epilog: Optional[Dict] = None):
    train_config = super().get_train_config(opts=opts, python_epilog=python_epilog)

    # remove import_model_train_epoch1 from config and replace it with preload_from_files, where we ignore ctc params
    if opts.get("preload_ignore_ctc", False):
      assert "import_model_train_epoch1" in train_config.config
      if "preload_from_files" not in train_config.config:
        train_config.config["preload_from_files"] = {}
      train_config.config["preload_from_files"]["existing_model"] = {
        "filename": train_config.config["import_model_train_epoch1"],
        "init_for_train": True,
        "ignore_missing": True,
        "ignore_params_prefixes": ["ctc"],
      }
      del train_config.config["import_model_train_epoch1"]

    return train_config

  def edit_network_only_train_length_model(self, net_dict: Dict):
    raise NotImplementedError

  def get_recog_config_for_forward_job(self, opts: Dict):
    raise NotImplementedError

  def get_dump_att_weight_config(self, corpus_key: str, opts: Dict):
    raise NotImplementedError

  def get_dump_length_model_probs_config(self, corpus_key: str, opts: Dict):
    raise NotImplementedError

  def get_dump_scores_config(self, corpus_key: str, opts: Dict):
    raise NotImplementedError


class LibrispeechConformerCtcConfigBuilder(CtcConfigBuilder, LibrispeechConformerConfigBuilder, ConformerConfigBuilder, ConfigBuilder):
  @staticmethod
  def _network_remove_not_needed_layers(net_dict: Dict):
    net_dict = copy.deepcopy(net_dict)
    for layer in ("output", "decision", "enc_ctx", "enc_value", "inv_fertility"):
      del net_dict[layer]
    net_dict["output"] = {"class": "copy", "from": "ctc"}
    return net_dict

  def get_net_dict(self, task: str, config_dict, python_prolog):
    if task == "train":
      return None
    else:
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict

      network_dict = self._network_remove_not_needed_layers(networks_dict[36])

      return network_dict

  def get_networks_dict(self, task: str, config_dict, python_prolog, use_get_global_config: bool = False):
    if task == "train":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict
      nets_dict = {}
      for k, v in networks_dict.items():
        nets_dict[k] = self._network_remove_not_needed_layers(v)
      return nets_dict
    else:
      return None
