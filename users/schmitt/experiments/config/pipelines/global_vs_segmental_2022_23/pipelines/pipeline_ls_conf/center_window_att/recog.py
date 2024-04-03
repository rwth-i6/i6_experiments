from typing import Tuple, Optional, List, Union, Dict

from i6_core.returnn.training import Checkpoint
from sisyphus import Path

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingPipeline, RasrSegmentalAttDecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.realignment_new import RasrRealignmentExperiment


def center_window_returnn_frame_wise_beam_search(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
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
      "correct_eos": False,
    })
  ReturnnSegmentalAttDecodingPipeline(
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


def center_window_returnn_frame_wise_beam_search_use_global_att_ilm(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Union[Checkpoint, Dict],
        lm_scale_list: Tuple[float, ...],
        lm_type: str,
        ilm_scale_list: Tuple[float, ...],
        ilm_correct_eos: bool = False,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
        att_weight_seq_tags: Optional[List] = None
):
  ilm_opts = {
    "type": "mini_att",
    "use_se_loss": False,
    "correct_eos": ilm_correct_eos,
    "mini_att_checkpoint": Checkpoint(Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2021_22/i6_core/returnn/training/ReturnnTrainingJob.YOiZDUR2V7S6/output/models/epoch.010.index"))
  }
  ReturnnSegmentalAttDecodingPipeline(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    checkpoint_aliases=checkpoint_aliases,
    beam_sizes=beam_size_list,
    lm_scales=lm_scale_list,
    lm_opts={
      "type": lm_type,
      "add_lm_eos_last_frame": True
    },
    ilm_scales=ilm_scale_list,
    ilm_opts=ilm_opts,
    run_analysis=run_analysis,
    analysis_opts={
      "att_weight_seq_tags": att_weight_seq_tags
    },
    search_alias="returnn_decoding_use-global-att-ilm"
  ).run()


def center_window_returnn_frame_wise_beam_search_like_rasr(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
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
  ilm_opts = {
    "type": ilm_type
  }
  if ilm_type == "mini_att":
    ilm_opts.update({
      "use_se_loss": False,
      "correct_eos": False,
    })
  ReturnnSegmentalAttDecodingPipeline(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    checkpoint_aliases=checkpoint_aliases,
    beam_sizes=beam_size_list,
    lm_scales=lm_scale_list,
    lm_opts={
      "type": lm_type,
      "add_lm_eos_last_frame": True
    },
    ilm_scales=ilm_scale_list,
    ilm_opts=ilm_opts,
    run_analysis=run_analysis,
    analysis_opts={
      "att_weight_seq_tags": att_weight_seq_tags
    },
    recog_opts={
      "dataset_opts": {
        "peak_normalization": False,
        "oggzip_paths": {
          "dev-other": [config_builder.variant_params["dataset"]["corpus"].oggzip_paths_wav["dev-other"].out_ogg_zip]
        }
      },
      "max_seqs": 1
    },
    search_rqmt={
      "gpu": 0,
      "time": 4,
      "cpu": 3
    },
    search_alias="returnn_decoding_like-rasr"
  ).run()


def center_window_returnn_frame_wise_beam_search_wav_no_peak_norm_same_static_padding(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
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
  ilm_opts = {
    "type": ilm_type
  }
  if ilm_type == "mini_att":
    ilm_opts.update({
      "use_se_loss": False,
      "correct_eos": False,
    })
  ReturnnSegmentalAttDecodingPipeline(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    checkpoint_aliases=checkpoint_aliases,
    beam_sizes=beam_size_list,
    lm_scales=lm_scale_list,
    lm_opts={
      "type": lm_type,
      "add_lm_eos_last_frame": True
    },
    ilm_scales=ilm_scale_list,
    ilm_opts=ilm_opts,
    run_analysis=run_analysis,
    analysis_opts={
      "att_weight_seq_tags": att_weight_seq_tags
    },
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


def center_window_rasr_frame_wise_beam_search(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Union[Checkpoint, Dict],
        max_segment_len_list: Tuple[int, ...],
):
  for max_segment_len in max_segment_len_list:
    if max_segment_len == -1:
      search_rqmt = {"mem": 4, "time": 1, "gpu": 0}
      concurrent = 100
    elif max_segment_len == 20:
      search_rqmt = {"mem": 12, "time": 12, "gpu": 0}
      concurrent = 200
    elif max_segment_len == 30:
      search_rqmt = {"mem": 16, "time": 15, "gpu": 0}
      concurrent = 300
    else:
      assert max_segment_len == 40
      search_rqmt = {"mem": 16, "time": 15, "gpu": 0}
      concurrent = 400

    for open_vocab in (True,):
      RasrSegmentalAttDecodingExperiment(
        alias=alias,
        search_rqmt=search_rqmt,
        reduction_factor=960,
        reduction_subtrahend=399,
        max_segment_len=max_segment_len,
        concurrent=concurrent,
        open_vocab=open_vocab,
        checkpoint=checkpoint,
        pruning_preset="simple-beam-search",
        checkpoint_alias="last",
        config_builder=config_builder,
      ).run_eval()


def center_window_rasr_realignment(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        checkpoint_alias: str
):
  RasrRealignmentExperiment(
    alias=alias,
    reduction_factor=960,
    reduction_subtrahend=399,
    job_rqmt={
      "mem": 4,
      "time": 1,
      "gpu": 0
    },
    concurrent=100,
    checkpoint=checkpoint,
    checkpoint_alias=checkpoint_alias,
    config_builder=config_builder,
  ).get_realignment()
