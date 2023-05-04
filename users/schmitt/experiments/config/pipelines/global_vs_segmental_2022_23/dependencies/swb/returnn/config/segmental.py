from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import SegmentalLabelDefinition

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.corpora.corpora import SWBCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config_builder.legacy_v1.segmental import SegmentalSWBExtendedConfig

from i6_experiments.users.schmitt.chunking import chunking as chunking_func

from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig

import copy

from typing import Dict, Optional
from sisyphus import Path


def remove_length_model_from_network(returnn_config: ReturnnConfig):
  returnn_config.config["network"]["output"]["unit"].update({
    "emit_prob0_0": {
      "class": "constant", "value": 1., "with_batch_dim": True},
    "emit_prob0": {
      "class": "expand_dims", "from": "emit_prob0_0", "axis": "f"},
  })

  return returnn_config


def get_train_config(
        dependencies: SegmentalLabelDefinition,
        alignments: Optional[Dict[str, Path]],
        variant_params: Dict,
        load: Optional[Checkpoint],
        length_scale: float,
        import_model_train_epoch1: Optional[Checkpoint] = None,
  ) -> ReturnnConfig:
  data_opts = {}
  for corpus_key in SWBCorpora.train_corpus_keys:
    data_opts[corpus_key] = {
      "data": corpus_key, "rasr_config_path": dependencies.rasr_config_paths["feature_extraction"][corpus_key],
      "segment_file": dependencies.segment_paths[corpus_key],
      "alignment": dependencies.alignment_paths[corpus_key] if alignments is None else alignments[corpus_key],
      "rasr_nn_trainer_exe": RasrExecutables.nn_trainer_path, "features": variant_params["config"]["features"],
      "raw_audio_path": dependencies.raw_audio_paths[corpus_key]
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
    "target_num_labels": dependencies.model_hyperparameters.target_num_labels_wo_blank,
    "vocab": dependencies.vocab_path,
    "sil_idx": dependencies.model_hyperparameters.sil_idx,
    "targetb_blank_idx": dependencies.model_hyperparameters.blank_idx
  })

  config_params = copy.deepcopy(variant_params["config"])

  # these parameters are not needed for the config class
  del config_params["label_type"]
  del config_params["model_type"]
  del config_params["returnn_python_exe"]
  del config_params["returnn_root"]
  do_chunk_fix = config_params.pop("do_chunk_fix")

  train_config_obj = SegmentalSWBExtendedConfig(
    task="train",
    train_data_opts=data_opts["train"],
    cv_data_opts=data_opts["cv"],
    devtrain_data_opts=data_opts["devtrain"],
    import_model=load,
    import_model_train_epoch1=import_model_train_epoch1,
    length_scale=length_scale,
    **config_params).get_config()

  # TODO: put this into the SegmentalSWBExtendedConfig
  """
  This is a custom chunking implementation: it only yields a chunk, if it does not only consist of blanks.
  This is done because the label-sync training loop breaks, if a sequence only contains blanks.
  """
  if "chunking" in train_config_obj.config and do_chunk_fix:
    chunk_size_data = train_config_obj.config["chunking"][0]["data"]
    chunk_size_align = train_config_obj.config["chunking"][0]["alignment"]
    chunk_step_data = train_config_obj.config["chunking"][1]["data"]
    chunk_step_align = train_config_obj.config["chunking"][1]["alignment"]
    del train_config_obj.config["chunking"]
    train_config_obj.python_prolog += [
      "from returnn.util.basic import NumbersDict",
      "chunk_size = NumbersDict({'alignment': %s, 'data': %s})" % (chunk_size_align, chunk_size_data),
      "chunk_step = NumbersDict({'alignment': %s, 'data': %s})" % (chunk_step_align, chunk_step_data),
      chunking_func]

  return train_config_obj


def get_recog_config(
        dependencies: SegmentalLabelDefinition,
        variant_params: Dict,
        corpus_key: str,
        dump_output: bool,
        length_scale: float,
        beam_size: int,
        use_recomb: bool
) -> ReturnnConfig:
  # create params for the dataset creation in RETURNN
  data_opts = {
    "data": corpus_key, "rasr_config_path": dependencies.rasr_config_paths["feature_extraction"][corpus_key],
    "rasr_nn_trainer_exe": RasrExecutables.nn_trainer_path, "vocab": dependencies.vocab_dict,
    "features": variant_params["config"]["features"], "raw_audio_path": dependencies.raw_audio_paths[corpus_key]
  }

  config_params = copy.deepcopy(variant_params["config"])

  # quick fix: the vocab variable in the config needs to be set to the vocab_dict when using recomb
  # TODO: fix
  if use_recomb:
    config_params["vocab"] = dependencies.vocab_dict

  # these parameters are not needed for the config class
  del config_params["label_type"]
  del config_params["model_type"]
  del config_params["returnn_python_exe"]
  del config_params["returnn_root"]
  del config_params["do_chunk_fix"]
  config = SegmentalSWBExtendedConfig(
    task="search", search_data_opts=data_opts, target="bpe", search_use_recomb=use_recomb, dump_output=dump_output,
    beam_size=beam_size, length_scale=length_scale, **config_params)
  return config.get_config()


def get_compile_config(
        variant_params: Dict,
        length_scale: float,
        remove_length_model: bool = False
) -> ReturnnConfig:

  config_params = copy.deepcopy(variant_params["config"])

  # these parameters are not needed for the config class
  del config_params["label_type"]
  del config_params["model_type"]
  del config_params["returnn_python_exe"]
  del config_params["returnn_root"]
  del config_params["do_chunk_fix"]

  returnn_config = SegmentalSWBExtendedConfig(
    task="eval", feature_stddev=3., length_scale=length_scale, **config_params).get_config()

  if remove_length_model:
    returnn_config = remove_length_model_from_network(returnn_config)

  return returnn_config
