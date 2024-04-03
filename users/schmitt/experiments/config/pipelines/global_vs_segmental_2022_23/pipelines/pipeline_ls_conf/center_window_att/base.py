from typing import Dict, Optional, List, Any, Tuple, Union
import copy

from sisyphus import Path

from i6_core.returnn.training import Checkpoint

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog import ReturnnDecodingExperimentV2, RasrDecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.realignment import RasrRealignmentExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment

default_import_model_name = "glob.conformer.mohammad.5.6"
default_train_opts = {
  "chunking_opts": None,
  "align_targets": None,
  "import_model_train_epoch1": external_checkpoints[default_import_model_name],
  "num_epochs": 10,
  "const_lr": 1e-4,
  "only_train_length_model": False,
  "no_ctc_loss": False,
  "lr_opts": None,
  "train_mini_lstm_opts": None,
  "cleanup_old_models": None
}
default_returnn_recog_opts = {
  "corpus_key": "dev-other",
  "concat_num": None,
  "search_rqmt": None,
  "batch_size": None,
  "load_ignore_missing_vars": False,
  "lm_opts": None,
  "ilm_correction_opts": None,
}
default_rasr_recog_opts = {
  "corpus_key": "dev-other",
  "search_rqmt": None,
  "max_segment_len": -1,
  "lm_opts": None,
  "lm_lookahead_opts": None,
  "open_vocab": True,
  "segment_list": None,
  "concurrent": 1,
  "native_lstm2_so_path": Path("/work/asr3/zeyer/schmitt/dependencies/tf_native_libraries/lstm2/simon/CompileNativeOpJob.Q1JsD9yc8hfb/output/NativeLstm2.so"),
  "ilm_correction_opts": None,
  "label_pruning": 12.0,
  "label_pruning_limit": 12,
  "word_end_pruning": 12.0,
  "word_end_pruning_limit": 12,
  "simple_beam_search": True,
  "full_sum_decoding": False,
  "allow_recombination": False,
}
default_rasr_realign_opts = {
  "corpus_key": "dev-other",
  "search_rqmt": None,
  "max_segment_len": -1,
  "segment_list": None,
  "concurrent": 1,
  "native_lstm2_so_path": Path("/work/asr3/zeyer/schmitt/dependencies/tf_native_libraries/lstm2/simon/CompileNativeOpJob.Q1JsD9yc8hfb/output/NativeLstm2.so"),
  "label_pruning": 12.0,
  "label_pruning_limit": 5000,
}


def get_center_window_att_config_builder(
        win_size: int,
        use_weight_feedback: bool = True,
        use_positional_embedding: bool = False,
        att_weight_recog_penalty_opts: Optional[Dict] = None,
        length_model_opts: Optional[Dict] = None,
        length_scale: float = 1.0,
        blank_penalty: Union[float, str] = 0.0,
        gaussian_att_weight_interpolation_opts: Optional[Dict] = None,
        expected_position_aux_loss_opts: Optional[Dict] = None,
        pos_pred_att_weight_interpolation_opts: Optional[Dict] = None,
        search_remove_eos: bool = False,
        use_old_global_att_to_seg_att_maker: bool = False,
):
  model_type = "librispeech_conformer_seg_att"
  variant_name = "seg.conformer.like-global"
  variant_params = copy.deepcopy(models[model_type][variant_name])
  variant_params["network"]["segment_center_window_size"] = win_size
  variant_params["network"]["use_weight_feedback"] = use_weight_feedback
  variant_params["network"]["use_positional_embedding"] = use_positional_embedding
  variant_params["network"]["att_weight_recog_penalty_opts"] = att_weight_recog_penalty_opts
  variant_params["network"]["gaussian_att_weight_interpolation_opts"] = gaussian_att_weight_interpolation_opts
  variant_params["network"]["pos_pred_att_weight_interpolation_opts"] = pos_pred_att_weight_interpolation_opts
  variant_params["network"]["expected_position_aux_loss_opts"] = expected_position_aux_loss_opts
  variant_params["network"]["length_scale"] = length_scale
  variant_params["network"]["blank_penalty"] = blank_penalty
  variant_params["network"]["search_remove_eos"] = search_remove_eos

  if length_model_opts:
    # make sure that we do not add any params which are not present in the defaults
    assert set(length_model_opts.keys()).issubset(set(variant_params["network"]["length_model_opts"].keys()))
    variant_params["network"]["length_model_opts"].update(length_model_opts)

  config_builder = LibrispeechConformerSegmentalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
    use_old_global_att_to_seg_att_maker=use_old_global_att_to_seg_att_maker
  )

  return config_builder


def returnn_recog_center_window_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        checkpoint_alias: str = "last",
        recog_opts: Optional[Dict[str, Any]] = None,
        analyse: bool = True,
):
  _recog_opts = copy.deepcopy(default_returnn_recog_opts)
  if recog_opts is not None:
    _recog_opts.update(recog_opts)

  recog_exp = ReturnnDecodingExperimentV2(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    checkpoint_alias=checkpoint_alias,
    **_recog_opts
  )
  recog_exp.run_eval()

  if analyse:
    if _recog_opts["concat_num"] is not None:
      raise NotImplementedError

    recog_exp.run_analysis(
      ground_truth_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[_recog_opts["corpus_key"]],
      att_weight_ref_alignment_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[_recog_opts["corpus_key"]],
      att_weight_ref_alignment_blank_idx=10025,
      att_weight_seq_tags=None,
    )


def rasr_recog_center_window_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        checkpoint_alias: str = "last",
        recog_opts: Optional[Dict[str, Any]] = None,
        analyse: bool = True,
):
  _recog_opts = copy.deepcopy(default_rasr_recog_opts)
  if recog_opts is not None:
    _recog_opts.update(recog_opts)

  recog_exp = RasrDecodingExperiment(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    corpus_key=_recog_opts["corpus_key"],
    search_rqmt=_recog_opts["search_rqmt"],
    length_norm=False,
    label_pruning=_recog_opts["label_pruning"],
    label_pruning_limit=_recog_opts["label_pruning_limit"],
    word_end_pruning=_recog_opts["word_end_pruning"],
    word_end_pruning_limit=_recog_opts["word_end_pruning_limit"],
    simple_beam_search=_recog_opts["simple_beam_search"],
    full_sum_decoding=_recog_opts["full_sum_decoding"],
    allow_recombination=_recog_opts["allow_recombination"],
    max_segment_len=_recog_opts["max_segment_len"],
    concurrent=_recog_opts["concurrent"],
    reduction_factor=960,
    reduction_subtrahend=399,
    lm_opts=_recog_opts["lm_opts"],
    lm_lookahead_opts=_recog_opts["lm_lookahead_opts"],
    open_vocab=_recog_opts["open_vocab"],
    segment_list=_recog_opts["segment_list"],
    native_lstm2_so_path=_recog_opts["native_lstm2_so_path"],
    ilm_correction_opts=_recog_opts["ilm_correction_opts"],
    checkpoint_alias=checkpoint_alias,
  )
  recog_exp.run_eval()

  if analyse:
    raise NotImplementedError


def rasr_realign_center_window_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        realign_opts: Optional[Dict[str, Any]] = None,
):
  _realign_opts = copy.deepcopy(default_rasr_realign_opts)
  if realign_opts is not None:
    _realign_opts.update(realign_opts)

  realign_exp = RasrRealignmentExperiment(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    corpus_key=_realign_opts["corpus_key"],
    search_rqmt=_realign_opts["search_rqmt"],
    length_norm=False,
    label_pruning=_realign_opts["label_pruning"],
    label_pruning_limit=_realign_opts["label_pruning_limit"],
    word_end_pruning=12.0,  # dummy value, not used
    word_end_pruning_limit=12,  # dummy value, not used
    simple_beam_search=False,  # dummy value, not used
    full_sum_decoding=False,  # dummy value, not used
    allow_recombination=False,  # dummy value, not used
    max_segment_len=_realign_opts["max_segment_len"],
    concurrent=_realign_opts["concurrent"],
    reduction_factor=960,
    reduction_subtrahend=399,
    lm_opts=None,  # dummy value, not used
    lm_lookahead_opts=None,  # dummy value, not used
    open_vocab=False,  # dummy value, not used
    segment_list=_realign_opts["segment_list"],
    native_lstm2_so_path=_realign_opts["native_lstm2_so_path"],
    ilm_correction_opts=None,  # dummy value, not used
  )
  realign_exp.do_realignment()


def train_center_window_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        train_opts: Optional[Dict[str, Any]] = None,
):
  _train_opts = copy.deepcopy(default_train_opts)
  if train_opts is not None:
    _train_opts.update(train_opts)

  train_exp = SegmentalTrainExperiment(
    config_builder=config_builder,
    alias=alias,
    n_epochs=_train_opts["num_epochs"],
    import_model_train_epoch1=_train_opts["import_model_train_epoch1"],
    align_targets=_train_opts["align_targets"],
    lr_opts={
      "type": "const_then_linear",
      "const_lr": _train_opts["const_lr"],
      "const_frac": 1 / 3,
      "final_lr": 1e-6,
      "num_epochs": _train_opts["num_epochs"]
    } if _train_opts["lr_opts"] is None else _train_opts["lr_opts"],
    chunking_opts=_train_opts["chunking_opts"],
    only_train_length_model=_train_opts["only_train_length_model"],
    no_ctc_loss=_train_opts["no_ctc_loss"],
    train_mini_lstm_opts=_train_opts["train_mini_lstm_opts"],
    cleanup_old_models=_train_opts["cleanup_old_models"],
  )
  return train_exp.run_train()


def standard_train_recog_center_window_att_import_global(
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        alias: str,
        train_opts: Optional[Dict[str, Any]] = None,
        recog_opts: Optional[Dict[str, Any]] = None,
        analyse: bool = True,
):
  _train_opts = train_opts
  train_mini_lstm_opts = _train_opts.pop("train_mini_lstm_opts", None)

  checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
    alias=alias,
    config_builder=config_builder,
    train_opts=_train_opts,
  )

  recog_checkpoints = config_builder.get_recog_checkpoints(
    model_dir=model_dir,
    learning_rates=learning_rates,
    key="dev_score_label_model/output_prob",
    checkpoints=checkpoints,
    n_epochs=_train_opts["num_epochs"]
  )

  if train_mini_lstm_opts is not None:
    if train_mini_lstm_opts.get("use_eos", False):
      align_targets = ctc_aligns.global_att_ctc_align.ctc_alignments_with_eos(
        segment_paths=config_builder.dependencies.segment_paths,
        blank_idx=config_builder.dependencies.model_hyperparameters.blank_idx,
        eos_idx=config_builder.dependencies.model_hyperparameters.sos_idx,
      )
      _train_opts["align_targets"] = align_targets

    train_mini_att_num_epochs = train_mini_lstm_opts.pop("num_epochs")
    _train_opts.update({
      "train_mini_lstm_opts": train_mini_lstm_opts,
      "import_model_train_epoch1": recog_checkpoints["last"],
      "num_epochs": 10,
      "lr_opts": {
        "type": "newbob",
        "learning_rate": 1e-4,
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 3,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rate_control_error_measure": "dev_error_label_model/output_prob"
      }
    })
    mini_att_checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
      alias=alias,
      config_builder=config_builder,
      train_opts=_train_opts,
    )
    mini_att_checkpoint = mini_att_checkpoints[train_mini_att_num_epochs]

  _recog_opts = copy.deepcopy(recog_opts)

  if isinstance(_recog_opts, dict) and "ilm_correction_opts" in _recog_opts and _recog_opts["ilm_correction_opts"] is not None:
    _recog_opts["ilm_correction_opts"]["mini_att_checkpoint"] = mini_att_checkpoint

  for checkpoint_alias, checkpoint in recog_checkpoints.items():
    if checkpoint_alias != "last":
      analyse = False

    # only do lm recog for the specified checkpoint alias
    if isinstance(_recog_opts, dict) and "lm_opts" in _recog_opts and isinstance(_recog_opts["lm_opts"], dict):
      if _recog_opts["lm_opts"]["checkpoint_alias"] != checkpoint_alias:
        continue

    if _recog_opts is None or _recog_opts.pop("returnn_recog", True):
      returnn_recog_center_window_att_import_global(
        alias=alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        recog_opts=_recog_opts,
        checkpoint_alias=checkpoint_alias,
        analyse=analyse,
      )
    else:
      rasr_recog_center_window_att_import_global(
        alias=alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        recog_opts=_recog_opts,
        checkpoint_alias=checkpoint_alias,
        analyse=analyse,
      )
