from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.general import SegmentalLabelDefinition

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.corpora.corpora import SWBCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.exes import RasrExecutables

from i6_experiments.users.schmitt.experiments.swb.transducer.config import SegmentalSWBExtendedConfig

from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig

import copy

from typing import Dict, Optional


def get_train_config(
        dependencies: SegmentalLabelDefinition, variant_params: Dict, load: Optional[Checkpoint], length_scale: float
  ) -> ReturnnConfig:
  data_opts = {}
  for corpus_key in SWBCorpora.train_corpus_keys:
    data_opts[corpus_key] = {
      "data": corpus_key, "rasr_config_path": dependencies.rasr_config_paths["feature_extraction"][corpus_key],
      "segment_file": dependencies.segment_paths[corpus_key], "alignment": dependencies.alignment_paths[corpus_key],
      "rasr_nn_trainer_exe": RasrExecutables.nn_trainer_path
    }
    if corpus_key == "train":
      data_opts[corpus_key].update({
        "epoch_split": variant_params["config"]["epoch_split"],
        "correct_concat_ep_split": False,
        "concat_seqs": False,
        "concat_seq_tags": None
      })

  # add specific label parameters to model parameters
  variant_params["config"].update({
    "sos_idx": dependencies.model_hyperparameters.sos_idx,
    "target_num_labels": dependencies.model_hyperparameters.target_num_labels,
    "vocab": dependencies.vocab_path,
    "sil_idx": dependencies.model_hyperparameters.sil_idx,
    "targetb_blank_idx": dependencies.model_hyperparameters.blank_idx
  })

  config_params = copy.deepcopy(variant_params["config"])

  # these parameters are not needed for the config class
  del config_params["label_type"]
  del config_params["model_type"]

  return SegmentalSWBExtendedConfig(
    task="train",
    train_data_opts=data_opts["train"],
    cv_data_opts=data_opts["cv"],
    devtrain_data_opts=data_opts["devtrain"],
    import_model=load,
    length_scale=length_scale,
    **config_params).get_config()


def get_recog_config(
        dependencies: SegmentalLabelDefinition, variant_params: Dict, corpus_key: str, dump_output: bool, length_scale: float
) -> ReturnnConfig:
  # create params for the dataset creation in RETURNN
  data_opts = {
    "data": corpus_key, "rasr_config_path": dependencies.rasr_config_paths["feature_extraction"][corpus_key],
    "rasr_nn_trainer_exe": RasrExecutables.nn_trainer_path, "vocab": dependencies.vocab_dict
  }

  config_params = copy.deepcopy(variant_params["config"])

  # these parameters are not needed for the config class
  del config_params["label_type"]
  del config_params["model_type"]
  config = SegmentalSWBExtendedConfig(
    task="search", search_data_opts=data_opts, target="bpe", search_use_recomb=False, dump_output=dump_output,
    beam_size=12, length_scale=length_scale, **config_params)
  return config.get_config()


def get_compile_config(
        variant_params: Dict, length_scale: float
) -> ReturnnConfig:

  config_params = copy.deepcopy(variant_params["config"])

  # these parameters are not needed for the config class
  del config_params["label_type"]
  del config_params["model_type"]

  return SegmentalSWBExtendedConfig(
    task="eval", feature_stddev=3., length_scale=length_scale, **config_params).get_config()
