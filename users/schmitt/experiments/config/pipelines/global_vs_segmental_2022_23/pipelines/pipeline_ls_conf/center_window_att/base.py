from typing import Dict, Optional, List, Any, Tuple
import copy

from i6_core.returnn.training import Checkpoint

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog import ReturnnDecodingExperimentV2, RasrDecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment

default_import_model_name = "glob.conformer.mohammad.5.6"


def get_center_window_att_config_builder(
        win_size: int,
        use_weight_feedback: bool = True,
        use_positional_embedding: bool = False,
        att_weight_recog_penalty_opts: Optional[Dict] = None,
        length_model_opts: Optional[Dict] = None,
        length_scale: float = 1.0,
        blank_penalty: float = 0.0,
        gaussian_att_weight_interpolation_opts: Optional[Dict] = None,
        expected_position_aux_loss_opts: Optional[Dict] = None,
        pos_pred_att_weight_interpolation_opts: Optional[Dict] = None,
        search_remove_eos: bool = False,
        use_old_global_att_to_seg_att_maker: bool = True,
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


def recog_center_window_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        search_corpus_key: str,
        concat_num: Optional[int] = None,
        search_rqmt: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        analyse: bool = False,
        att_weight_seq_tags: Optional[List[str]] = None,
        load_ignore_missing_vars: bool = False,
):
  recog_exp = ReturnnDecodingExperimentV2(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    corpus_key=search_corpus_key,
    concat_num=concat_num,
    search_rqmt=search_rqmt,
    batch_size=batch_size,
    load_ignore_missing_vars=load_ignore_missing_vars,
  )
  recog_exp.run_eval()

  if analyse:
    if concat_num is not None:
      raise NotImplementedError

    recog_exp.run_analysis(
      ground_truth_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[search_corpus_key],
      att_weight_ref_alignment_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[search_corpus_key],
      att_weight_ref_alignment_blank_idx=10025,
      att_weight_seq_tags=att_weight_seq_tags,
    )


def rasr_recog_center_window_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        search_corpus_key: str,
        concat_num: Optional[int] = None,
        search_rqmt: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        analyse: bool = False,
        att_weight_seq_tags: Optional[List[str]] = None,
        load_ignore_missing_vars: bool = False,
        max_segment_len: int = 20,
):
  recog_exp = RasrDecodingExperiment(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    corpus_key=search_corpus_key,
    search_rqmt=search_rqmt,
    length_norm=False,
    label_pruning=12.0,
    label_pruning_limit=12,
    word_end_pruning=12.0,
    word_end_pruning_limit=12,
    simple_beam_search=True,
    full_sum_decoding=False,
    allow_recombination=False,
    max_segment_len=max_segment_len,
    concurrent=4,
    reduction_factor=960,
    reduction_subtrahend=399
  )
  recog_exp.run_eval()


def standard_train_recog_center_window_att_import_global(
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        alias: str,
        n_epochs: int,
        const_lr: float,
        chunking_opts: Optional[Dict[str, int]] = None,
        att_weight_seq_tags: Optional[List[str]] = None,
):
  train_exp = SegmentalTrainExperiment(
    config_builder=config_builder,
    alias=alias,
    n_epochs=n_epochs,
    import_model_train_epoch1=external_checkpoints[default_import_model_name],
    align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
    lr_opts={
      "type": "const_then_linear",
      "const_lr": const_lr,
      "const_frac": 1 / 3,
      "final_lr": 1e-6,
      "num_epochs": n_epochs
    },
    chunking_opts=chunking_opts,
  )
  checkpoints, model_dir, learning_rates = train_exp.run_train()

  recog_center_window_att_import_global(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoints[n_epochs],
    analyse=True,
    search_corpus_key="dev-other",
    att_weight_seq_tags=att_weight_seq_tags,
  )
