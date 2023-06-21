from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition, SegmentalLabelDefinition, LabelDefinition
from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict
from i6_experiments.users.schmitt.conformer_pretrain import get_network
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask
from i6_experiments.users.schmitt.dynamic_lr import dynamic_lr_str

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from sisyphus import Path

import os
import re
from abc import abstractmethod, ABC
from typing import Dict, Optional
import copy


class ConfigBuilder(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          variant_params: Dict,
          initial_lr=None,
          import_model=None,
          import_model_train_epoch1=None,
  ):
    self.dependencies = dependencies
    self.variant_params = variant_params

    self.post_config_dict = dict(
      cleanup_old_models=True
    )

    self.config_dict = dict(
      use_tensorflow=True,
      log_batch_size=True,
      truncation=-1,
      tf_log_memory_usage=True,
      debug_print_layer_output_template=True
    )

    self.config_dict.update(dict(
      batching="random",
      max_seqs=200,
      max_seq_length={"targets": 75},
    ))

    if import_model is not None:
      self.config_dict["load"] = import_model
    if import_model_train_epoch1 is not None:
      self.config_dict["import_model_train_epoch1"] = import_model_train_epoch1

    self.config_dict.update(dict(
      gradient_clip=0.0,
      gradient_noise=0.0,
      adam=True,
      optimizer_epsilon=1e-8,
      accum_grad_multiple_step=2,
      learning_rate=initial_lr if initial_lr is not None else 0.0009,
      learning_rate_control="constant",
      learning_rates=[8e-5] * 35
    ))

    self.python_prolog = [
      "from returnn.tf.util.data import DimensionTag",
      "import os",
      "import numpy as np",
      "from subprocess import check_output, CalledProcessError",
      "import sys",
      "sys.setrecursionlimit(4000)",
      _mask,
      random_mask,
      transform,
    ]

    if variant_params["config"].get("speed_pert"):
      self.python_prolog.append(speed_pert)

    self.python_epilog = [
      dynamic_lr_str.format(
        initial_lr=8.999999999999999e-05,
        peak_lr=0.0009,
        final_lr=1e-06,
        cycle_ep=915,
        total_ep=2035,
        n_step=1350
      )
    ]

  @abstractmethod
  def get_train_dataset_dict(self):
    pass

  @abstractmethod
  def get_cv_dataset_dict(self):
    pass

  @abstractmethod
  def get_devtrain_dataset_dict(self):
    pass

  @abstractmethod
  def get_search_dataset_dict(self, corpus_key: str):
    pass

  @abstractmethod
  def get_eval_dataset_dict(self, corpus_key: str):
    pass

  @abstractmethod
  def get_extern_data_dict(self):
    return dict(data=dict(
      dim=1,
      same_dim_tags_as=dict(
        t=CodeWrapper("DimensionTag(kind=DimensionTag.Types.Spatial, description='time', dimension=None)")
      )
    ))

  @abstractmethod
  def get_net_dict(self, task: str):
    pass

  @abstractmethod
  def get_networks_dict(self, task: str):
    pass

  def get_train_datasets(self):
    return dict(
      extern_data=self.get_extern_data_dict(),
      train=self.get_train_dataset_dict(),
      dev=self.get_cv_dataset_dict(),
      eval_datasets={"devtrain": self.get_devtrain_dataset_dict()}
    )

  def get_search_dataset(self, search_corpus_key: str):
    return dict(
      extern_data=self.get_extern_data_dict(),
      search_data=self.get_search_dataset_dict(corpus_key=search_corpus_key)
    )

  def get_eval_dataset(self, eval_corpus_key: str):
    return dict(
      extern_data=self.get_extern_data_dict(),
      eval=self.get_eval_dataset_dict(corpus_key=eval_corpus_key)
    )

  def get_train_config(self):
    self.config_dict.update(
      dict(
        task="train",
        batch_size=2400000
      )
    )

    self.config_dict.update(self.get_train_datasets())

    net_dict = self.get_net_dict("train")
    if net_dict is not None:
      self.config_dict["network"] = net_dict

    return ReturnnConfig(
      config=self.config_dict,
      post_config=self.post_config_dict,
      python_prolog=self.python_prolog,
      python_epilog=self.python_epilog,
      staged_network_dict=self.get_networks_dict("train")
    )

  def get_recog_config(self, search_corpus_key: str):
    self.config_dict.update(dict(
      task="search",
      batch_size=2400000,
      search_output_layer="decision"
    ))

    self.config_dict.update(self.get_search_dataset(search_corpus_key=search_corpus_key))

    net_dict = self.get_net_dict("search")
    if net_dict is not None:
      self.config_dict["network"] = net_dict

    return ReturnnConfig(
      config=self.config_dict,
      post_config=self.post_config_dict,
      python_prolog=self.python_prolog,
      python_epilog=self.python_epilog,
      staged_network_dict=self.get_networks_dict("search"))

  def get_eval_config(self, eval_corpus_key: str):
    self.config_dict.update(dict(
      batch_size=2400000,
    ))

    self.config_dict.update(self.get_eval_dataset(eval_corpus_key=eval_corpus_key))

    net_dict = self.get_net_dict("search")
    if net_dict is not None:
      self.config_dict["network"] = net_dict

    return ReturnnConfig(
      config=self.config_dict,
      post_config=self.post_config_dict,
      python_prolog=self.python_prolog,
      python_epilog=self.python_epilog,
      staged_network_dict=self.get_networks_dict("search"))


class GlobalConfigBuilder(ConfigBuilder):
  def __init__(
          self,
          dependencies: GlobalLabelDefinition,
          variant_params: Dict,
          use_eos: bool = True
  ):
    super().__init__(dependencies=dependencies, variant_params=variant_params)

    self.use_eos = use_eos

  def get_net_dict(self, task: str):
    if task == "train":
      return None
    elif task == "search":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict
      return networks_dict[36]
    else:
      raise NotImplementedError

  def get_networks_dict(self, task: str):
    if task == "train":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict
      return networks_dict
    elif task == "search":
      return None
    else:
      raise NotImplementedError

  def get_default_dataset_opts(self, corpus_key):
    return dict(
      oggzip_path_list=self.dependencies.oggzip_paths[corpus_key],
      bpe_file=self.dependencies.bpe_codes_path,
      vocab_file=self.dependencies.vocab_path,
      segment_file=self.dependencies.segment_paths[corpus_key],
      seq_postfix=self.dependencies.model_hyperparameters.sos_idx if self.use_eos else None
    )

  def get_train_dataset_dict(self):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=20,
      pre_process=CodeWrapper("speed_pert") if self.variant_params["config"].get("speed_pert") else None,
      seq_ordering="laplace:.1000",
      epoch_wise_filter={(1, 5): {"max_mean_len": 1000}},
      **self.get_default_dataset_opts("train")
    )

  def get_cv_dataset_dict(self):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts("cv")
    )

  def get_devtrain_dataset_dict(self):
    return get_dataset_dict(
      fixed_random_subset=3000,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts("devtrain")
    )

  def get_search_dataset_dict(self, corpus_key: str):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts(corpus_key)
    )

  def get_eval_dataset_dict(self, corpus_key: str):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts(corpus_key)
    )

  def get_extern_data_dict(self):
    extern_data_dict = super().get_extern_data_dict()
    extern_data_dict.update(dict(
      targets=dict(dim=self.dependencies.model_hyperparameters.target_num_labels, sparse=True)
    ))

    return extern_data_dict


class SegmentalConfigBuilder(ConfigBuilder):
  def __init__(
          self,
          dependencies: SegmentalLabelDefinition,
          variant_params: Dict,
          alignments: Optional[Dict[str, Path]] = None
  ):
    super().__init__(dependencies=dependencies, variant_params=variant_params)

    self.dependencies = dependencies
    self.alignments = alignments

  def get_net_dict(self, task: str):
    if task == "train":
      return None
    elif task == "search":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict
      return networks_dict[22]
    else:
      raise NotImplementedError

  def get_networks_dict(self, task: str):
    if task == "train":
      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.network_builder.networks import networks_dict

      new_networks_dict = {}
      for i, net_dict in networks_dict.items():
        new_networks_dict[i] = self._make_global_attention_segmental(net_dict)

      return new_networks_dict
    elif task == "search":
      return None
    else:
      raise NotImplementedError

  def get_default_dataset_opts(self, corpus_key):
    return dict(
      oggzip_path_list=self.dependencies.oggzip_paths[corpus_key],
      bpe_file=self.dependencies.bpe_codes_path,
      vocab_file=self.dependencies.vocab_path,
      segment_file=self.dependencies.segment_paths.get(corpus_key, None),
      hdf_targets=self.alignments.get(corpus_key, None)
    )

  def get_train_dataset_dict(self):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=6,
      pre_process=CodeWrapper("speed_pert") if self.variant_params["config"].get("speed_pert") else None,
      seq_ordering="laplace:6000",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts("train")
    )

  def get_cv_dataset_dict(self):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts("cv")
    )

  def get_devtrain_dataset_dict(self):
    return get_dataset_dict(
      fixed_random_subset=3000,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts("devtrain")
    )

  def get_search_dataset_dict(self, corpus_key: str):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts(corpus_key)
    )

  def get_eval_dataset_dict(self, corpus_key: str):
    return get_dataset_dict(
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      **self.get_default_dataset_opts(corpus_key)
    )

  def get_extern_data_dict(self):
    extern_data_dict = super().get_extern_data_dict()
    extern_data_dict.update(dict(
      targets=dict(dim=self.dependencies.model_hyperparameters.target_num_labels, sparse=True)
    ))

    return extern_data_dict

  def _make_global_attention_segmental(self, global_net_dict) -> Dict:
    seg_net_dict = copy.deepcopy(global_net_dict)

    del seg_net_dict["inv_fertility"]
    del seg_net_dict["enc_ctx"]
    del seg_net_dict["decision"]

    seg_net_dict["ctc"]["target"] = "label_ground_truth"

    seg_net_dict.update({
      "existing_alignment": {
        "class": "reinterpret_data",
        "from": "data:targets",
        "set_sparse": True,
        "set_sparse_dim": self.dependencies.model_hyperparameters.target_num_labels,
        "size_base": "encoder",
      },
      "is_label": {
        "class": "compare",
        "from": "existing_alignment",
        "kind": "not_equal",
        "value": self.dependencies.model_hyperparameters.blank_idx,
      },
      "label_ground_truth_masked": {
        "class": "reinterpret_data",
        "enforce_batch_major": True,
        "from": "label_ground_truth_masked0",
        "register_as_extern_data": "label_ground_truth",
        "set_sparse_dim": self.dependencies.model_hyperparameters.target_num_labels_wo_blank,
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
        # "out_spatial_dim": CodeWrapper('DimensionTag(kind=DimensionTag.Types.Spatial, description="label-axis")'),
        "register_as_extern_data": "segment_lens_masked",
        "unit": {"class": "copy", "from": "data"},
      },
      "segment_starts_masked": {
        "class": "masked_computation",
        "from": "output/segment_starts",
        "mask": "is_label",
        # "out_spatial_dim": CodeWrapper('DimensionTag(kind=DimensionTag.Types.Spatial, description="label-axis")'),
        "register_as_extern_data": "segment_starts_masked",
        "unit": {"class": "copy", "from": "data"},
      },
    })

    del seg_net_dict["output"]["unit"]["end"]
    del seg_net_dict["output"]["unit"]["accum_att_weights"]
    del seg_net_dict["output"]["unit"]["weight_feedback"]

    seg_net_dict["label_model"] = copy.deepcopy(seg_net_dict["output"])
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
      # "att_ctx": {
      #   "class": "reinterpret_data",
      #   "from": "att_ctx0",
      #   "set_dim_tags": {
      #     "stag:sliced-time:att_ctx": CodeWrapper('DimensionTag(kind=DimensionTag.Types.Spatial, description="att_t")')
      #   },
      # },
      "att_ctx": {
        "class": "slice_nd",
        "from": "base:enc_ctx",
        "size": "segment_lens",
        "start": "segment_starts",
      },
      # "att_val": {
      #   "class": "reinterpret_data",
      #   "from": "att_val0",
      #   "set_dim_tags": {
      #     "stag:sliced-time:att_val": CodeWrapper('DimensionTag(kind=DimensionTag.Types.Spatial, description="att_t")')
      #   },
      # },
      "att_val": {
        "class": "slice_nd",
        "from": "base:encoder",
        "size": "segment_lens",
        "start": "segment_starts",
      },
    })

    seg_net_dict["label_model"]["unit"]["energy_in"]["from"] = ["att_ctx", "s_transformed"]
    seg_net_dict["label_model"]["unit"]["att_weights"]["axis"] = "stag:sliced-time:att_ctx"

    seg_net_dict["output"] = {
      "back_prop": True,
      "class": "rec",
      "from": "encoder",
      "include_eos": True,
      "size_target": "targetb",
      "target": "targetb",
      "unit": {
        "am": {"class": "copy", "from": "data:source"},
        "blank_log_prob": {
          "class": "eval",
          "eval": "tf.math.log_sigmoid(-source(0))",
          "from": "emit_prob0",
        },
        "const1": {"class": "constant", "value": 1},
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
        "output": {
          "beam_size": 4,
          "cheating": "exclusive",
          "class": "choice",
          "from": "data",
          "initial_output": 0,
          "input_type": "log_prob",
          "target": "targetb",
        },
        "output_emit": {
          "class": "compare",
          "from": "output",
          "initial_output": True,
          "kind": "not_equal",
          "value": self.dependencies.model_hyperparameters.blank_idx,
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
        "segment_lens": {
          "class": "combine",
          "from": ["segment_lens0", "const1"],
          "is_output_layer": True,
          "kind": "add",
        },
        "segment_lens0": {
          "class": "combine",
          "from": [":i", "segment_starts"],
          "kind": "sub",
        },
        "segment_starts": {
          "class": "switch",
          "condition": "prev:output_emit",
          "false_from": "prev:segment_starts",
          "initial_output": 0,
          "is_output_layer": True,
          "true_from": ":i",
        },
      },
    }

    return seg_net_dict
