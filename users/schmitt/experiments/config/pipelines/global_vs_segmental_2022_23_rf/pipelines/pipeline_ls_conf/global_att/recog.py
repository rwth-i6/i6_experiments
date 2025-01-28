from typing import Tuple, Optional, List, Union, Dict

from i6_core.returnn.training import Checkpoint
from sisyphus import Path

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import ConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.recog import _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.recog import model_recog, model_recog_pure_torch
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnGlobalAttDecodingPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.gmm_alignments import LIBRISPEECH_GMM_WORD_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import lm_checkpoints


def global_att_returnn_label_sync_beam_search(
        alias: str,
        config_builder: ConfigBuilderRF,
        checkpoint: Union[Checkpoint, Dict],
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        mini_lstm_const_lr: Optional[float] = None,
        lm_alias: Optional[str] = "kazuki-10k",
        lm_checkpoint: Optional[Checkpoint] = lm_checkpoints["kazuki-10k"],
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
        analyze_gradients: bool = False,
        plot_att_weights: bool = True,
        att_weight_seq_tags: Optional[List] = (
          "dev-other/3660-6517-0005/3660-6517-0005",
          "dev-other/6467-62797-0001/6467-62797-0001",
          "dev-other/6467-62797-0002/6467-62797-0002",
          "dev-other/7697-105815-0015/7697-105815-0015",
          "dev-other/7697-105815-0051/7697-105815-0051",
        ),
        corpus_keys: Tuple[str, ...] = ("dev-other",),
        batch_size: Optional[int] = None,
        concat_num: Optional[int] = None,
        only_do_analysis: bool = False,
        analysis_do_forced_align_on_gradients: bool = False,
        analysis_plot_encoder_gradient_graph: bool = False,
        analysis_dump_gradients: bool = False,
        analysis_dump_self_att: bool = False,
        analysis_ref_alignment_opts: Optional[Dict] = None,
        analysis_dump_gradients_input_layer_name: str = "encoder_input",
        analysis_analyze_gradients_plot_encoder_layers: bool = False,
        analsis_analyze_gradients_plot_log_gradients: bool = False,
        analsis_analyze_gradients_search: bool = False,
        behavior_version: Optional[int] = None,
        length_normalization_exponent: float = 1.0,
        external_aed_opts: Optional[Dict] = None,
        base_scale: float = 1.0,
        sbatch_args: Optional[List[str]] = None,
):
  if lm_type is not None:
    assert len(checkpoint_aliases) == 1, "Do LM recog only for the best checkpoint"

  ilm_opts = {"type": ilm_type}
  if ilm_type == "mini_att":
    ilm_opts.update({
      "use_se_loss": False,
      "const_lr": 1e-4 if mini_lstm_const_lr is None else mini_lstm_const_lr,
    })

  recog_opts = {
    "recog_def": model_recog,
    "forward_step_func": _returnn_v2_forward_step,
    "forward_callback": _returnn_v2_get_forward_callback,
    "behavior_version": behavior_version,
    "length_normalization_exponent": length_normalization_exponent,
    "external_aed_opts": external_aed_opts,
    "base_model_scale": base_scale,
  }
  if concat_num is not None:
    recog_opts["dataset_opts"] = {"concat_num": concat_num}
  if batch_size is not None:
    recog_opts["batch_size"] = batch_size

  if run_analysis:
    assert len(corpus_keys) == 1, "Only one corpus key is supported for analysis"

    if analysis_ref_alignment_opts is None:
      assert corpus_keys[0] in ("train", "dev-other")
      analysis_ref_alignment_opts = {}
      if corpus_keys[0] == "train":
        analysis_ref_alignment_opts["ref_alignment_hdf"] = LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"]
        analysis_ref_alignment_opts["ref_alignment_blank_idx"] = LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx
        analysis_ref_alignment_opts["ref_alignment_vocab_path"] = LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path
      else:
        analysis_ref_alignment_opts["ref_alignment_hdf"] = LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["dev-other"]
        analysis_ref_alignment_opts["ref_alignment_blank_idx"] = LibrispeechBPE10025_CTC_ALIGNMENT.model_hyperparameters.blank_idx
        analysis_ref_alignment_opts["ref_alignment_vocab_path"] = LibrispeechBPE10025_CTC_ALIGNMENT.vocab_path

    analysis_opts = {
      "att_weight_seq_tags": list(att_weight_seq_tags) if att_weight_seq_tags is not None else None,
      "analyze_gradients": analyze_gradients,
      "plot_att_weights": plot_att_weights,
      "ref_alignment_hdf": analysis_ref_alignment_opts.get(
        "ref_alignment_hdf", LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths[corpus_keys[0]]),
      "ref_alignment_blank_idx": analysis_ref_alignment_opts.get(
        "ref_alignment_blank_idx", LibrispeechBPE10025_CTC_ALIGNMENT.model_hyperparameters.blank_idx),
      "ref_alignment_vocab_path": analysis_ref_alignment_opts.get(
        "ref_alignment_vocab_path", LibrispeechBPE10025_CTC_ALIGNMENT.vocab_path),
      "do_forced_align_on_gradients": analysis_do_forced_align_on_gradients,
      "plot_encoder_gradient_graph": analysis_plot_encoder_gradient_graph,
      "dump_gradients": analysis_dump_gradients,
      "dump_gradients_input_layer_name": analysis_dump_gradients_input_layer_name,
      "analyze_gradients_plot_encoder_layers": analysis_analyze_gradients_plot_encoder_layers,
      "analyze_gradients_plot_log_gradients": analsis_analyze_gradients_plot_log_gradients,
      "dump_self_att": analysis_dump_self_att,
      "analyze_gradients_search": analsis_analyze_gradients_search,
    }
  else:
    analysis_opts = None

  if sbatch_args is not None:
    recog_rqmt = {"sbatch_args": sbatch_args}
  else:
    recog_rqmt = {}

  pipeline = ReturnnGlobalAttDecodingPipeline(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    checkpoint_aliases=checkpoint_aliases,
    beam_sizes=beam_size_list,
    lm_scales=lm_scale_list,
    lm_opts={"type": lm_type, "add_lm_eos_last_frame": True, "alias": lm_alias, "checkpoint": lm_checkpoint},
    ilm_scales=ilm_scale_list,
    ilm_opts=ilm_opts,
    run_analysis=run_analysis,
    analysis_opts=analysis_opts,
    recog_opts=recog_opts,
    search_alias=f'returnn_decoding',
    corpus_keys=corpus_keys,
    only_do_analysis=only_do_analysis,
    search_rqmt=recog_rqmt,
  )
  pipeline.run()

  return pipeline
