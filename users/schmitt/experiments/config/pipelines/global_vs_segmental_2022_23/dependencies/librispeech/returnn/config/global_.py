from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables

# from i6_experiments.users.schmitt.experiments.swb.global_enc_dec.config import GlobalEncoderDecoderConfig
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.librispeech.returnn.config_builder.legacy_v1.global_ import GlobalEncoderDecoderConfig

from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig

import copy

from typing import Dict, Optional


def get_train_config(
        dependencies: GlobalLabelDefinition,
        variant_params: Dict,
        load: Optional[Checkpoint],
        import_model_train_epoch1: Optional[Checkpoint] = None,
        initial_lr: Optional[float] = None
  ) -> ReturnnConfig:
  data_opts = {}
  for corpus_key in LibrispeechCorpora.train_corpus_keys:
    if variant_params["config"]["features"] == "gammatone":
      segment_path = dependencies.segment_paths[corpus_key]
    else:
      segment_path = dependencies.raw_audio_train_segment_paths[corpus_key]
    data_opts[corpus_key] = {
      "data": corpus_key, "rasr_config_path": dependencies.rasr_config_paths["feature_extraction"][corpus_key],
      "segment_file": segment_path,
      "label_hdf": dependencies.label_paths[corpus_key],
      "rasr_nn_trainer_exe": RasrExecutables.nn_trainer_path, "label_name": "bpe",
      "raw_audio_path": dependencies.oggzip_paths[corpus_key], "features": variant_params["config"]["features"],
    }
    if corpus_key == "train":
      data_opts[corpus_key].update({
        "epoch_split": variant_params["config"]["epoch_split"],
        "concat_seqs": False,
        "concat_seq_tags": None
      })

  # add specific label parameters to model parameters
  variant_params["config"].update({
    "sos_idx": dependencies.model_hyperparameters.sos_idx,
    "target_num_labels": dependencies.model_hyperparameters.target_num_labels,
    "vocab": dependencies.vocab_dict,
    "sil_idx": dependencies.model_hyperparameters.sil_idx,
  })

  config_params = copy.deepcopy(variant_params["config"])

  # these parameters are not needed for the config class
  del config_params["label_type"]
  del config_params["model_type"]

  config_builder = config_params["config_builder"]
  del config_params["config_builder"]

  returnn_config = GlobalEncoderDecoderConfig(
    task="train",
    train_data_opts=data_opts["train"],
    cv_data_opts=data_opts["cv"],
    devtrain_data_opts=data_opts["devtrain"],
    import_model=load,
    import_model_train_epoch1=import_model_train_epoch1,
    initial_lr=initial_lr,
    **config_params).get_config()

  return returnn_config


def get_recog_config(
        dependencies: GlobalLabelDefinition,
        variant_params: Dict,
        corpus_key: str,
        dump_output: bool,
        beam_size: int
) -> ReturnnConfig:
  # create params for the dataset creation in RETURNN
  data_opts = {
    "data": corpus_key, "rasr_config_path": dependencies.rasr_config_paths["feature_extraction"][corpus_key],
    "rasr_nn_trainer_exe": RasrExecutables.nn_trainer_path, "vocab": dependencies.vocab_dict,
    "features": variant_params["config"]["features"], "raw_audio_path": dependencies.oggzip_paths[corpus_key]
  }

  config_params = copy.deepcopy(variant_params["config"])

  # these parameters are not needed for the config class
  del config_params["label_type"]
  del config_params["model_type"]
  del config_params["config_builder"]

  returnn_config = GlobalEncoderDecoderConfig(
    task="search", search_data_opts=data_opts, dump_output=dump_output,
    beam_size=beam_size, **config_params).get_config()

  returnn_config = update_global_att_config_to_match_seg_att_config(returnn_config)

  return returnn_config
