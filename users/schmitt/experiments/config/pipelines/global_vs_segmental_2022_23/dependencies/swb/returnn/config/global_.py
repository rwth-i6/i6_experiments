from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import GlobalLabelDefinition

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.corpora.corpora import SWBCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables

# from i6_experiments.users.schmitt.experiments.swb.global_enc_dec.config import GlobalEncoderDecoderConfig
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config_builder.legacy_v1.global_ import GlobalEncoderDecoderConfig

from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig

import copy

from typing import Dict, Optional


def update_global_att_config_to_match_seg_att_config(returnn_config: ReturnnConfig):
  """
  This function changes some layer names in the returnn config of my global attention model in order to match the names
  in the config of my segmental attention model in order to be able to import the global attention model checkpoint
  into my segmental model. This is only meant as a temporary fix.
  :param returnn_config: the global attention config which is meant to be changed.
  :return:
  """
  # in the segmental model, the enc_ctx is inside the recurrent loop (because it depends on the predicted segments)
  returnn_config.config["network"]["enc_ctx"]["name_scope"] = "output/rec/att_ctx"

  # these two are simple renamings
  returnn_config.config["network"]["output"]["unit"]["energy"]["name_scope"] = "att_energy0"
  returnn_config.config["network"]["output"]["unit"]["label_prob"]["name_scope"] = "label_log_prob0"

  # here, in case the state vector is used in the internal LM, the ordering needs to be changed from
  # ["target_embed", "prev:att"] to ["prev:att", "target_embed"] to match the segmental model
  if "prev:att" in returnn_config.config["network"]["output"]["unit"]["lm"]["from"]:
    returnn_config.config["network"]["output"]["unit"]["lm"]["from"] = ["prev:att", "target_embed"]

  # in the segmental model, during recognition, the LM is masked because it does not depend on the blank ctx
  returnn_config.config["network"]["output"]["unit"]["lm"]["name_scope"] = "lm_masked/lm"

  # the embedding behavior in the segmental model is a bit different than for the global att model
  returnn_config.config["network"]["output"]["unit"]["prev_out_non_blank"] = {
      "class": "reinterpret_data",
      "from": "prev:output",
      "set_sparse": True,
      "set_sparse_dim": 1031,
  }
  returnn_config.config["network"]["output"]["unit"]["target_embed"]["name_scope"] = "prev_non_blank_embed"
  returnn_config.config["network"]["output"]["unit"]["target_embed"]["from"] = "prev_out_non_blank"
  del returnn_config.config["network"]["output"]["unit"]["target_embed"]["initial_output"]

  return returnn_config


def get_train_config(
        dependencies: GlobalLabelDefinition,
        variant_params: Dict,
        load: Optional[Checkpoint],
        import_model_train_epoch1: Optional[Checkpoint] = None
  ) -> ReturnnConfig:
  data_opts = {}
  for corpus_key in SWBCorpora.train_corpus_keys:
    data_opts[corpus_key] = {
      "data": corpus_key, "rasr_config_path": dependencies.rasr_config_paths["feature_extraction"][corpus_key],
      "segment_file": dependencies.segment_paths[corpus_key], "label_hdf": dependencies.label_paths[corpus_key],
      "rasr_nn_trainer_exe": RasrExecutables.nn_trainer_path, "label_name": "bpe",
      "raw_audio_path": dependencies.raw_audio_paths[corpus_key], "features": variant_params["config"]["features"],
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
  del config_params["returnn_python_exe"]
  del config_params["returnn_root"]

  config_builder = config_params["config_builder"]
  del config_params["config_builder"]

  returnn_config = GlobalEncoderDecoderConfig(
    task="train",
    train_data_opts=data_opts["train"],
    cv_data_opts=data_opts["cv"],
    devtrain_data_opts=data_opts["devtrain"],
    import_model=load,
    import_model_train_epoch1=import_model_train_epoch1,
    **config_params).get_config()

  returnn_config = update_global_att_config_to_match_seg_att_config(returnn_config)

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
    "features": variant_params["config"]["features"], "raw_audio_path": dependencies.raw_audio_paths[corpus_key]
  }

  config_params = copy.deepcopy(variant_params["config"])

  # these parameters are not needed for the config class
  del config_params["label_type"]
  del config_params["model_type"]
  del config_params["returnn_python_exe"]
  del config_params["returnn_root"]
  del config_params["config_builder"]

  returnn_config = GlobalEncoderDecoderConfig(
    task="search", search_data_opts=data_opts, dump_output=dump_output,
    beam_size=beam_size, **config_params).get_config()

  returnn_config = update_global_att_config_to_match_seg_att_config(returnn_config)

  return returnn_config
