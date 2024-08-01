from sisyphus import tk, Path
from typing import Dict, List, Union

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.training import PtCheckpoint
from i6_core.text.processing import WriteToTextFileJob

from i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJobV2
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import ConfigBuilderRF, SegmentalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf import dump_att_weights as dump_att_weights_forward_funcs


def dump_att_weights(
        config_builder: ConfigBuilderRF,
        seq_tags: List[str],
        corpus_key: str,
        checkpoint: Union[PtCheckpoint, Dict],
        returnn_root: Path,
        returnn_python_exe: Path,
        alias: str,
):
  segment_file = WriteToTextFileJob(
    content=seq_tags
  )

  config_opts = {
    "corpus_key": corpus_key,
    "dataset_opts": {
      "segment_paths": {corpus_key: segment_file.out_file},
    },
    "forward_step_func": dump_att_weights_forward_funcs._returnn_v2_forward_step,
    "forward_callback": dump_att_weights_forward_funcs._returnn_v2_get_forward_callback,
    "dump_att_weight_def": dump_att_weights_forward_funcs.dump_att_weights,
  }
  if isinstance(config_builder, SegmentalAttConfigBuilderRF):
    config_opts["dataset_opts"]["hdf_targets"] = LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths

  target_blank_idx = None

  output_files = [
    "att_weights.hdf",
    "targets.hdf",
  ]
  if isinstance(config_builder, SegmentalAttConfigBuilderRF) and config_builder.center_window_size is not None:
    output_files += ["seg_lens.hdf", "seg_starts.hdf", "center_positions.hdf", ]
    target_blank_idx = config_builder.variant_params["dependencies"].model_hyperparameters.blank_idx

  dump_att_weight_config = config_builder.get_dump_att_weight_config(opts=config_opts)
  dump_att_weights_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=dump_att_weight_config,
    returnn_root=returnn_root,
    returnn_python_exe=returnn_python_exe,
    output_files=output_files,
    mem_rqmt=6,
    time_rqmt=1,
  )
  dump_att_weights_job.add_alias(f"{alias}/analysis/dump_att_weights")
  tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_files["att_weights.hdf"])

  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=dump_att_weights_job.out_files["att_weights.hdf"],
    targets_hdf=dump_att_weights_job.out_files["targets.hdf"],
    seg_lens_hdf=dump_att_weights_job.out_files.get("seg_lens.hdf"),
    seg_starts_hdf=dump_att_weights_job.out_files.get("seg_starts.hdf"),
    center_positions_hdf=dump_att_weights_job.out_files.get("center_positions.hdf"),
    target_blank_idx=target_blank_idx,
    ref_alignment_blank_idx=LibrispeechBPE10025_CTC_ALIGNMENT.model_hyperparameters.blank_idx,
    ref_alignment_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths[corpus_key],
    json_vocab_path=config_builder.variant_params["dependencies"].vocab_path,
    ctc_alignment_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths[corpus_key],
    segment_whitelist=seq_tags,
  )
  plot_att_weights_job.add_alias(f"{alias}/analysis/plot_att_weights")
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)
