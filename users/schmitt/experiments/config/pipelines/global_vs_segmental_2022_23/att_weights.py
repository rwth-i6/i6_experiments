import copy

# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder import ConfigBuilder, GlobalConfigBuilder, SegmentalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import \
  ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import \
  SegmentalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import \
  GlobalConfigBuilder

from i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJobV2, PlotCtcProbsJob
from i6_experiments.users.schmitt.alignment.att_weights import AttentionWeightStatisticsJob, ScatterAttentionWeightMonotonicityAgainstWERJob

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
        sclite_report_dir: Path,
        seq_tags_to_analyse: Optional[List[str]] = None,
        plot_energies: bool = False,
        dump_ctc: bool = True,
        calc_att_weight_stats: bool = False,
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
      "segment_paths": {corpus_key: None if calc_att_weight_stats else segment_file.out_file},
      "hdf_targets": {corpus_key: hdf_targets},
    },
  }

  hdf_filenames = {
    "att_weights": "att_weights.hdf",
    "targets": "targets.hdf"
  }

  if plot_energies:
    hdf_filenames["att_energies"] = "att_energies.hdf"

  if isinstance(config_builder, SegmentalConfigBuilder):
    if hdf_alias == "ground_truth":
      assert hdf_targets == ref_alignment
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

  elif isinstance(config_builder, GlobalConfigBuilder) and dump_ctc:
    hdf_filenames.update({
      "ctc_alignment": "ctc_alignment.hdf",
    })

  dump_frame_probs_opts["hdf_filenames"] = hdf_filenames

  dump_frame_probs_config = config_builder.get_dump_att_weight_config(corpus_key=corpus_key, opts=dump_frame_probs_opts)
  forward_job = ReturnnForwardJob(
    model_checkpoint=checkpoint,
    returnn_config=dump_frame_probs_config,
    returnn_root=variant_params["returnn_root"],
    returnn_python_exe=variant_params["returnn_python_exe"],
    hdf_outputs=list(hdf_filenames.values()),
    eval_mode=True
  )
  forward_job.add_alias("%s/analysis/%s/%s/%s/dump_hdf" % (alias, "att_weights", corpus_key, hdf_alias))

  if isinstance(config_builder, SegmentalConfigBuilder):
    target_blank_idx = config_builder.dependencies.model_hyperparameters.blank_idx
  else:
    target_blank_idx = None

  att_weight_hdfs = [forward_job.out_hdf_files["att_weights.hdf"]]
  job_aliases = ["att_weights"]
  if plot_energies:
    att_weight_hdfs.append(forward_job.out_hdf_files["att_energies.hdf"])
    job_aliases.append("att_energies")

  for job_alias, att_weight_hdf in zip(job_aliases, att_weight_hdfs):
    plot_att_weights_job = PlotAttentionWeightsJobV2(
      att_weight_hdf=att_weight_hdf,
      targets_hdf=forward_job.out_hdf_files["targets.hdf"],
      seg_lens_hdf=forward_job.out_hdf_files.get(hdf_filenames.get("seg_lens")),
      seg_starts_hdf=forward_job.out_hdf_files.get(hdf_filenames.get("seg_starts")),
      center_positions_hdf=forward_job.out_hdf_files.get(hdf_filenames.get("center_positions")),
      target_blank_idx=target_blank_idx,
      ref_alignment_blank_idx=ref_alignment_blank_idx,
      ref_alignment_hdf=ref_alignment,
      json_vocab_path=config_builder.dependencies.vocab_path,
      ctc_alignment_hdf=forward_job.out_hdf_files.get(hdf_filenames.get("ctc_alignment")),
      segment_whitelist=seq_tags_to_analyse,
    )
    plot_att_weights_job.add_alias(f"{alias}/analysis/att_weights/{corpus_key}/{hdf_alias}/plot_{job_alias}")
    tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)

  if calc_att_weight_stats:
    att_weight_stats_job = AttentionWeightStatisticsJob(
      att_weights_hdf=forward_job.out_hdf_files["att_weights.hdf"],
      bpe_vocab=config_builder.dependencies.vocab_path,
      ctc_alignment_hdf=forward_job.out_hdf_files.get("ctc_alignment.hdf"),
      segment_file=None,
      ctc_blank_idx=ref_alignment_blank_idx,
      remove_last_label=True if hdf_alias == "ground_truth" else False,  # in search we do not dump EOS
    )
    att_weight_stats_job.add_alias(f"{alias}/analysis/att_weights/{corpus_key}/{hdf_alias}/monotonicity_stats/statistics")
    tk.register_output(att_weight_stats_job.get_one_alias(), att_weight_stats_job.out_statistics)

    scatter_att_weight_monotonicity_against_wer_job = ScatterAttentionWeightMonotonicityAgainstWERJob(
      seq_tag_to_non_monotonicity_map_file=att_weight_stats_job.out_seq_tag_to_non_monotonic_argmax,
      sclite_report_dir=sclite_report_dir,
      dataset="tedlium2"
    )
    scatter_att_weight_monotonicity_against_wer_job.add_alias(
      f"{alias}/analysis/att_weights/{corpus_key}/{hdf_alias}/scatter_att_weight_monotonicity_against_wer/statistics")
    tk.register_output(
      scatter_att_weight_monotonicity_against_wer_job.get_one_alias(),
      scatter_att_weight_monotonicity_against_wer_job.out_scatter,
    )


def dump_ctc_probs(
        config_builder: GlobalConfigBuilder,
        variant_params: Dict,
        checkpoint: Checkpoint,
        hdf_targets: Path,
        corpus_key: str,
        hdf_alias: str,
        alias: str,
        seq_tags_to_analyse: List[str],
):
  segment_file = WriteToTextFileJob(
    content=seq_tags_to_analyse
  )

  dump_ctc_probs_opts = {
    "dataset_opts": {
      "segment_paths": {corpus_key: segment_file.out_file},
      "hdf_targets": {corpus_key: hdf_targets},
    },
    "hdf_filenames": {
      "ctc_probs": "ctc_probs.hdf",
      "targets": "targets.hdf",
      "ctc_alignment": "ctc_alignment.hdf",
    }
  }

  dump_ctc_probs_config = config_builder.get_dump_ctc_probs_config(corpus_key=corpus_key, opts=dump_ctc_probs_opts)
  forward_job = ReturnnForwardJob(
    model_checkpoint=checkpoint,
    returnn_config=dump_ctc_probs_config,
    returnn_root=variant_params["returnn_root"],
    returnn_python_exe=variant_params["returnn_python_exe"],
    hdf_outputs=list(dump_ctc_probs_opts["hdf_filenames"].values()),
    eval_mode=True
  )
  forward_job.add_alias("%s/analysis/ctc_probs/%s/%s/dump_hdf" % (alias, corpus_key, hdf_alias))

  plot_ctc_probs_job = PlotCtcProbsJob(
    ctc_probs_hdf=forward_job.out_hdf_files["ctc_probs.hdf"],
    targets_hdf=forward_job.out_hdf_files["targets.hdf"],
    ctc_alignment_hdf=forward_job.out_hdf_files["ctc_alignment.hdf"],
    ctc_blank_idx=config_builder.dependencies.model_hyperparameters.target_num_labels,
    json_vocab_path=config_builder.dependencies.vocab_path,
    segment_whitelist=seq_tags_to_analyse,
  )
  plot_ctc_probs_job.add_alias(f"{alias}/analysis/ctc_probs/{corpus_key}/{hdf_alias}/plot_ctc_probs")
  tk.register_output(plot_ctc_probs_job.get_one_alias(), plot_ctc_probs_job.out_plot_dir)
