from typing import Tuple, Optional, List, Union, Dict

from i6_core.returnn.training import Checkpoint
from sisyphus import Path

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import ConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.recog import _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.recog import model_recog, model_recog_pure_torch
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnGlobalAttDecodingPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT


def global_att_returnn_label_sync_beam_search(
        alias: str,
        config_builder: ConfigBuilderRF,
        checkpoint: Union[Checkpoint, Dict],
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
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
):
  if lm_type is not None:
    assert len(checkpoint_aliases) == 1, "Do LM recog only for the best checkpoint"

  ilm_opts = {"type": ilm_type}
  if ilm_type == "mini_att":
    ilm_opts.update({
      "use_se_loss": False,
    })

  recog_opts = {
    "recog_def": model_recog,
    "forward_step_func": _returnn_v2_forward_step,
    "forward_callback": _returnn_v2_get_forward_callback,
  }
  if concat_num is not None:
    recog_opts["dataset_opts"] = {"concat_num": concat_num}
  if batch_size is not None:
    recog_opts["batch_size"] = batch_size

  if run_analysis:
    assert len(corpus_keys) == 1, "Only one corpus key is supported for analysis"
    analysis_opts = {
      "att_weight_seq_tags": list(att_weight_seq_tags) if att_weight_seq_tags is not None else None,
      "analyze_gradients": analyze_gradients,
      "plot_att_weights": plot_att_weights,
      "ref_alignment_hdf": LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths[corpus_keys[0]],
      "ref_alignment_blank_idx": LibrispeechBPE10025_CTC_ALIGNMENT.model_hyperparameters.blank_idx,
      "ref_alignment_vocab_path": LibrispeechBPE10025_CTC_ALIGNMENT.vocab_path,
      "do_forced_align_on_gradients": analysis_do_forced_align_on_gradients,
      "plot_encoder_gradient_graph": analysis_plot_encoder_gradient_graph,
      "dump_gradients": analysis_dump_gradients,
    }
  else:
    analysis_opts = None

  ReturnnGlobalAttDecodingPipeline(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    checkpoint_aliases=checkpoint_aliases,
    beam_sizes=beam_size_list,
    lm_scales=lm_scale_list,
    lm_opts={"type": lm_type, "add_lm_eos_last_frame": True},
    ilm_scales=ilm_scale_list,
    ilm_opts=ilm_opts,
    run_analysis=run_analysis,
    analysis_opts=analysis_opts,
    recog_opts=recog_opts,
    search_alias=f'returnn_decoding',
    corpus_keys=corpus_keys,
    only_do_analysis=only_do_analysis
  ).run()
