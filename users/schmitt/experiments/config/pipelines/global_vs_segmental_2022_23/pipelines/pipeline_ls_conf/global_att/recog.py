from typing import Tuple, Optional, List, Union, Dict

from i6_core.returnn.training import Checkpoint
from sisyphus import Path

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import LibrispeechConformerGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnGlobalAttDecodingPipeline, ReturnnGlobalAttDecodingExperiment


def global_att_returnn_label_sync_beam_search(
        alias: str,
        config_builder: LibrispeechConformerGlobalAttentionConfigBuilder,
        checkpoint: Union[Checkpoint, Dict],
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
        att_weight_seq_tags: Optional[List] = None
):
  ilm_opts = {"type": ilm_type}
  if ilm_type == "mini_att":
    ilm_opts.update({
      "use_se_loss": False,
      "correct_eos": True,
    })
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
    analysis_opts={"att_weight_seq_tags": att_weight_seq_tags}
  ).run()


def global_att_returnn_label_sync_beam_search_wav_no_peak_norm_same_static_padding(
        alias: str,
        config_builder: LibrispeechConformerGlobalAttentionConfigBuilder,
        checkpoint: Union[Checkpoint, Dict],
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
        att_weight_seq_tags: Optional[List] = None
):
  ilm_opts = {"type": ilm_type}
  if ilm_type == "mini_att":
    ilm_opts.update({
      "use_se_loss": False,
      "correct_eos": True,
    })
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
    recog_opts={
      "dataset_opts": {
        "peak_normalization": False,
        "oggzip_paths": {
          "dev-other": [config_builder.variant_params["dataset"]["corpus"].oggzip_paths_wav["dev-other"].out_ogg_zip]
        }
      },
      "use_same_static_padding": True,
    },
    search_alias="returnn_decoding_wav_no-peak-norm_same-static-padding"
  ).run()


def global_att_returnn_label_sync_beam_search_concat_recog(
        alias: str,
        config_builder: LibrispeechConformerGlobalAttentionConfigBuilder,
        checkpoint: Union[Checkpoint, Dict],
        concat_nums: Tuple[int, ...],
        checkpoint_alias: str,
):
  # TODO: i restarted the search jobs and something seems to have changed in RETURNN since the last time, because
  # it now throws an error:
  """
    File "/u/schmitt/src/returnn_new/returnn/datasets/audio.py", line 346, in OggZipDataset._get_ref_seq_idx
    line: return self._seq_order[seq_idx]
    locals:
      self = <local> <OggZipDataset 'dataset_id140515693976880_subdataset_zip_dataset' epoch=None>
      self._seq_order = <local> None
      seq_idx = <local> 0
  """
  for concat_num in concat_nums:
    ReturnnGlobalAttDecodingExperiment(
      alias=alias,
      config_builder=config_builder,
      checkpoint=checkpoint,
      checkpoint_alias=checkpoint_alias,
      recog_opts={
        "dataset_opts": {"concat_num": concat_num}
      }
    ).run_eval()
