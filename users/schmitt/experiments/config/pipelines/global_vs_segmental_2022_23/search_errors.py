import copy

# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder import ConfigBuilder, SegmentalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import SegmentalConfigBuilder
from i6_experiments.users.schmitt.recognition.search_errors import CalcSearchErrorJobV2

from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.training import Checkpoint

from sisyphus import Path, tk

from typing import Dict, Any, Optional


def calc_search_errors(
        config_builder: ConfigBuilder,
        variant_params: Dict,
        checkpoint: Checkpoint,
        ground_truth_hdf_targets: Path,
        search_hdf_targets: Path,
        corpus_key: str,
        alias: str,
        segment_file: Optional[Path] = None,
):
  forward_jobs = {}

  hdf_filenames = {
    "label_model_log_scores": "label_model_log_scores.hdf",
    "targets": "targets.hdf"
  }

  dump_scores_opts_ground_truth = {
    "dataset_opts": {
      "hdf_targets": {corpus_key: ground_truth_hdf_targets}
    },
  }  # type: Dict[str, Any]
  dump_scores_opts_search = {
    "dataset_opts": {
      "hdf_targets": {corpus_key: search_hdf_targets}
    },
  }  # type: Dict[str, Any]

  if segment_file:
    dump_scores_opts_ground_truth["dataset_opts"]["segment_paths"] = {corpus_key: segment_file}
    dump_scores_opts_search["dataset_opts"]["segment_paths"] = {corpus_key: segment_file}

  if isinstance(config_builder, SegmentalConfigBuilder):
    dump_scores_opts_ground_truth["use_train_net"] = True
    dump_scores_opts_search["use_train_net"] = True
    hdf_filenames["length_model_log_scores"] = "length_model_log_scores.hdf"
    if config_builder.variant_params["network"].get("att_weight_recog_penalty_opts"):
      # in this case, we also want to dump the att weight penalty scores
      hdf_filenames["att_weight_penalty_scores"] = "att_weight_penalty_scores.hdf"

  dump_scores_opts_ground_truth["hdf_filenames"] = hdf_filenames
  dump_scores_opts_search["hdf_filenames"] = hdf_filenames

  for opts_alias, opts in (
          ("ground_truth", dump_scores_opts_ground_truth),
          ("search", dump_scores_opts_search),
  ):

    dump_scores_config = config_builder.get_dump_scores_config(
      corpus_key=corpus_key,
      opts=opts
    )

    # dump_scores_config.config.update({
    #   "forward_use_search": False,
    #   "forward_batch_size": CodeWrapper("batch_size")
    # })

    forward_job = ReturnnForwardJob(
      model_checkpoint=checkpoint,
      returnn_config=dump_scores_config,
      returnn_root=variant_params["returnn_root"],
      returnn_python_exe=variant_params["returnn_python_exe"],
      hdf_outputs=list(opts["hdf_filenames"].values()),
      eval_mode=True
    )
    forward_job.add_alias("%s/analysis/search_errors/%s/dump_%s_scores_hdf" % (alias, corpus_key, opts_alias))

    forward_jobs[opts_alias] = forward_job

  if isinstance(config_builder, SegmentalConfigBuilder):
    blank_idx = config_builder.dependencies.model_hyperparameters.blank_idx
  else:
    blank_idx = None

  label_sync_scores_ground_truth_hdf = {
    "label_model": forward_jobs["ground_truth"].out_hdf_files["label_model_log_scores.hdf"]
  }
  att_weight_penalty_scores_ground_truth_hdf = forward_jobs["ground_truth"].out_hdf_files.get(hdf_filenames.get("att_weight_penalty_scores"))
  if att_weight_penalty_scores_ground_truth_hdf:
    label_sync_scores_ground_truth_hdf["att_weight_penalty"] = att_weight_penalty_scores_ground_truth_hdf

  frame_sync_scores_ground_truth_hdf = {}
  length_model_scores_ground_truth_hdf = forward_jobs["ground_truth"].out_hdf_files.get(hdf_filenames.get("length_model_log_scores"))
  if length_model_scores_ground_truth_hdf:
    frame_sync_scores_ground_truth_hdf["length_model"] = length_model_scores_ground_truth_hdf

  label_sync_scores_search_hdf = {
    "label_model": forward_jobs["search"].out_hdf_files["label_model_log_scores.hdf"]
  }
  att_weight_penalty_scores_search_hdf = forward_jobs["search"].out_hdf_files.get(
    hdf_filenames.get("att_weight_penalty_scores"))
  if att_weight_penalty_scores_search_hdf:
    label_sync_scores_search_hdf["att_weight_penalty"] = att_weight_penalty_scores_search_hdf

  frame_sync_scores_search_hdf = {}
  length_model_scores_search_hdf = forward_jobs["search"].out_hdf_files.get(
    hdf_filenames.get("length_model_log_scores"))
  if length_model_scores_search_hdf:
    frame_sync_scores_search_hdf["length_model"] = length_model_scores_search_hdf

  calc_search_error_job = CalcSearchErrorJobV2(
    label_sync_scores_ground_truth_hdf=label_sync_scores_ground_truth_hdf,
    frame_sync_scores_ground_truth_hdf=frame_sync_scores_ground_truth_hdf,
    targets_ground_truth_hdf=forward_jobs["ground_truth"].out_hdf_files["targets.hdf"],
    label_sync_scores_search_hdf=label_sync_scores_search_hdf,
    frame_sync_scores_search_hdf=frame_sync_scores_search_hdf,
    targets_search_hdf=forward_jobs["search"].out_hdf_files["targets.hdf"],
    blank_idx=blank_idx,
    json_vocab_path=config_builder.dependencies.vocab_path
  )
  calc_search_error_job.add_alias("%s/analysis/search_errors/%s/calc_search_errors" % (alias, corpus_key))
  tk.register_output("%s/analysis/search_errors/%s/calc_search_errors" % (alias, corpus_key), calc_search_error_job.out_search_errors)

