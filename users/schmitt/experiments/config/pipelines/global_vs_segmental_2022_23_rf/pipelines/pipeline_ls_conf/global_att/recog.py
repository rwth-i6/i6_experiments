from typing import Tuple, Optional, List, Union, Dict

from i6_core.returnn.training import Checkpoint
from sisyphus import Path

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import ConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.recog import _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.recog import model_recog, model_recog_pure_torch
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnGlobalAttDecodingPipeline


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
        att_weight_seq_tags: Optional[List] = None,
        corpus_keys: Tuple[str, ...] = ("dev-other",),
        batch_size: Optional[int] = None,
):
  if lm_type is not None:
    assert len(checkpoint_aliases) == 1, "Do LM recog only for the best checkpoint"

  ilm_opts = {"type": ilm_type}
  if ilm_type == "mini_att":
    ilm_opts.update({
      "use_se_loss": False,
      "correct_eos": True,
    })

  recog_opts = {
    "recog_def": model_recog,
    "forward_step_func": _returnn_v2_forward_step,
    "forward_callback": _returnn_v2_get_forward_callback,
  }
  if batch_size is not None:
    recog_opts["batch_size"] = batch_size

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
    analysis_opts={"att_weight_seq_tags": att_weight_seq_tags},
    recog_opts=recog_opts,
    search_alias=f'returnn_decoding',
    corpus_keys=corpus_keys,
  ).run()
