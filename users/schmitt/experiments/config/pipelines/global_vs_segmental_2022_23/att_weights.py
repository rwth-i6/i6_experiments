import copy

# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder import ConfigBuilder, GlobalConfigBuilder, SegmentalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import \
  ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import \
  SegmentalConfigBuilder

from i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJobV2

from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.training import Checkpoint
from i6_core.text.processing import WriteToTextFileJob

from sisyphus import Path, tk

from typing import Dict, List, Optional, Callable

write_cv_segments_to_file_job = WriteToTextFileJob(
  content=[
    "dev-clean/1919-142785-0048/1919-142785-0048",
    "dev-other/8288-274162-0005/8288-274162-0005",
    "dev-other/6467-56885-0015/6467-56885-0015"
  ]
)

write_dev_other_segments_to_file_job = WriteToTextFileJob(
  content=[
    "dev-other/116-288045-0003/116-288045-0003",
    "dev-other/116-288045-0006/116-288045-0006",
    "dev-other/1701-141760-0007/1701-141760-0007",
  ]
)

segment_files = {
  "cv": write_cv_segments_to_file_job.out_file,
  "dev-other": write_dev_other_segments_to_file_job.out_file
}


def dump_att_weights(
        config_builder: ConfigBuilder,
        variant_params: Dict,
        checkpoint: Checkpoint,
        hdf_targets: Path,
        corpus_key: str,
        hdf_alias: str,
        ref_alignment: Path,
        ref_alignment_blank_idx: int,
        alias: str,
        seq_tags_to_analyse: Optional[List[str]] = None,
):
  def _get_dump_att_weight_config(corpus_key, opts):
    return config_builder.get_dump_att_weight_config(corpus_key=corpus_key, opts=opts)

  dump_frame_probs(
    config_builder=config_builder,
    variant_params=variant_params,
    checkpoint=checkpoint,
    hdf_targets=hdf_targets,
    corpus_key=corpus_key,
    hdf_alias=hdf_alias,
    ref_alignment=ref_alignment,
    ref_alignment_blank_idx=ref_alignment_blank_idx,
    alias=alias,
    seq_tags_to_analyse=seq_tags_to_analyse,
    probs_name="att_weights",
    get_dump_frame_probs_config_func=_get_dump_att_weight_config
  )


def dump_frame_probs(
        config_builder: ConfigBuilder,
        variant_params: Dict,
        checkpoint: Checkpoint,
        hdf_targets: Path,
        corpus_key: str,
        hdf_alias: str,
        ref_alignment: Path,
        ref_alignment_blank_idx: int,
        alias: str,
        probs_name: str,
        get_dump_frame_probs_config_func: Callable,
        seq_tags_to_analyse: Optional[List[str]] = None,
):
  if seq_tags_to_analyse is None:
    if corpus_key == "cv":
      seq_tags_to_analyse = [
        "dev-clean/1919-142785-0048/1919-142785-0048",
        "dev-other/8288-274162-0005/8288-274162-0005",
        "dev-other/6467-56885-0015/6467-56885-0015"
      ]
    elif corpus_key == "dev-other":
      seq_tags_to_analyse = [
        "dev-other/3660-6517-0005/3660-6517-0005",
        "dev-other/6467-62797-0001/6467-62797-0001",
        "dev-other/6467-62797-0002/6467-62797-0002",
        "dev-other/7697-105815-0015/7697-105815-0015",
        "dev-other/7697-105815-0051/7697-105815-0051",
      ]
    else:
      raise ValueError("No default seq tags for this corpus key!")

  segment_file = WriteToTextFileJob(
    content=seq_tags_to_analyse
  )

  dump_frame_probs_opts = {
    "dataset_opts": {
      "segment_paths": {corpus_key: segment_file.out_file},
      "hdf_targets": {corpus_key: hdf_targets}
    },
  }

  hdf_filenames = {
    probs_name: "%s.hdf" % probs_name,
    "targets": "targets.hdf"
  }

  if isinstance(config_builder, SegmentalConfigBuilder):
    dump_frame_probs_opts["use_train_net"] = True

    hdf_filenames.update({
      "seg_starts": "seg_starts.hdf",
      "seg_lens": "seg_lens.hdf"
    })

    if config_builder.variant_params["network"].get("segment_center_window_size") is not None:
      # in this case, we also want to dump the center positions
      hdf_filenames.update({
        "center_positions": "center_positions.hdf"
      })

  dump_frame_probs_opts["hdf_filenames"] = hdf_filenames

  dump_frame_probs_config = get_dump_frame_probs_config_func(
    corpus_key=corpus_key,
    opts=dump_frame_probs_opts
  )

  forward_job = ReturnnForwardJob(
    model_checkpoint=checkpoint,
    returnn_config=dump_frame_probs_config,
    returnn_root=variant_params["returnn_root"],
    returnn_python_exe=variant_params["returnn_python_exe"],
    hdf_outputs=list(hdf_filenames.values()),
    eval_mode=True
  )
  forward_job.add_alias("%s/analysis/%s/%s/%s/dump_hdf" % (alias, probs_name, corpus_key, hdf_alias))

  if isinstance(config_builder, SegmentalConfigBuilder):
    target_blank_idx = config_builder.dependencies.model_hyperparameters.blank_idx
  else:
    target_blank_idx = None

  plot_frame_probs_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=forward_job.out_hdf_files["%s.hdf" % probs_name],
    targets_hdf=forward_job.out_hdf_files["targets.hdf"],
    seg_lens_hdf=forward_job.out_hdf_files.get(hdf_filenames.get("seg_lens")),
    seg_starts_hdf=forward_job.out_hdf_files.get(hdf_filenames.get("seg_starts")),
    center_positions_hdf=forward_job.out_hdf_files.get(hdf_filenames.get("center_positions")),
    target_blank_idx=target_blank_idx,
    ref_alignment_blank_idx=ref_alignment_blank_idx,
    ref_alignment_hdf=ref_alignment,
    json_vocab_path=config_builder.dependencies.vocab_path,
  )
  plot_frame_probs_job.add_alias("%s/analysis/%s/%s/%s/plot" % (alias, probs_name, corpus_key, hdf_alias))
  tk.register_output(("%s/analysis/%s/%s/%s/plot" % (alias, probs_name, corpus_key, hdf_alias)),
                     plot_frame_probs_job.out_plot_dir)
