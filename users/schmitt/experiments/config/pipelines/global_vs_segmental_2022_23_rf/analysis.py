from sisyphus import tk, Path
from typing import Dict, List, Union, Optional
import copy

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.training import PtCheckpoint
from i6_core.text.processing import WriteToTextFileJob
from i6_core.returnn import ReturnnDumpHDFJob

from i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJobV2
from i6_experiments.users.schmitt.recognition.search_errors import CalcSearchErrorJobRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import ConfigBuilderRF, SegmentalAttConfigBuilderRF, GlobalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf import dump_att_weights as dump_att_weights_forward_funcs
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf import analyze_gradients as analyze_gradients_forward_funcs
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import realignment as realignment_forward_funcs
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_ import forward as global_att_forward_funcs


def dump_att_weights(
        config_builder: ConfigBuilderRF,
        seq_tags: List[str],
        corpus_key: str,
        checkpoint: Union[PtCheckpoint, Dict],
        returnn_root: Path,
        returnn_python_exe: Path,
        alias: str,
        hdf_targets: Optional[Path],
        ref_alignment_hdf: Path,
        ref_alignment_blank_idx: int,
        ref_alignment_vocab_path: Path,
):
  if "concat" in corpus_key:
    corpus_key_ = corpus_key
    corpus_key = corpus_key_.split("_")[0]
    concat_num = int(corpus_key_.split("_")[1].split("-")[1])
  else:
    concat_num = None
    corpus_key_ = corpus_key

  segment_file = WriteToTextFileJob(
    content=seq_tags
  )

  config_opts = {
    "corpus_key": corpus_key,
    "dataset_opts": {
      "segment_paths": {corpus_key: segment_file.out_file},
      "concat_num": concat_num,
    },
    "forward_step_func": dump_att_weights_forward_funcs._returnn_v2_forward_step,
    "forward_callback": dump_att_weights_forward_funcs._returnn_v2_get_forward_callback,
    "dump_att_weight_def": dump_att_weights_forward_funcs.dump_att_weights,
  }
  if hdf_targets is not None:
    config_opts["dataset_opts"]["hdf_targets"] = {corpus_key: hdf_targets}
  else:
    assert isinstance(config_builder, GlobalAttConfigBuilderRF), "need hdf_targets (alignment) for segmental att model"

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
    ref_alignment_blank_idx=ref_alignment_blank_idx,
    ref_alignment_hdf=ref_alignment_hdf,
    ref_alignment_json_vocab_path=ref_alignment_vocab_path,
    json_vocab_path=config_builder.variant_params["dependencies"].vocab_path,
    ctc_alignment_hdf=ref_alignment_hdf,
    segment_whitelist=seq_tags,
  )
  plot_att_weights_job.add_alias(f"{alias}/analysis/plot_att_weights")
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)


def analyze_gradients(
        config_builder: ConfigBuilderRF,
        seq_tags: List[str],
        corpus_key: str,
        checkpoint: Union[PtCheckpoint, Dict],
        returnn_root: Path,
        returnn_python_exe: Path,
        alias: str,
        hdf_targets: Optional[Path],
        ref_alignment_hdf: Path,
        ref_alignment_blank_idx: int,
        ref_alignment_vocab_path: Path,
        seq_alias: str,
):
  assert seq_alias in ("ground-truth", "search")

  segment_file = WriteToTextFileJob(
    content=seq_tags
  )

  if "concat" in corpus_key:
    corpus_key_ = corpus_key
    corpus_key = corpus_key_.split("_")[0]
    concat_num = int(corpus_key_.split("_")[1].split("-")[1])

    segment_paths_dict = {"concat_segment_paths": {corpus_key: segment_file.out_file}}

    concat_alignment_dataset_dict = {
      "class": "ConcatSeqsDataset",
      "dataset": {
        "class": "HDFDataset",
        "files": [ref_alignment_hdf],
        "partition_epoch": 1,
        "use_cache_manager": True,
      },
      "seq_len_file": config_builder.variant_params["dataset"]["corpus"].seq_len_files[corpus_key],
      "seq_list_file": segment_file.out_file,
    }
    ref_alignment_hdf = ReturnnDumpHDFJob(
      concat_alignment_dataset_dict, returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
    ).out_hdf
  else:
    concat_num = None

    segment_paths_dict = {"segment_paths": {corpus_key: segment_file.out_file}}

  config_opts = {
    "corpus_key": corpus_key,
    "dataset_opts": {
      **segment_paths_dict,
      "concat_num": concat_num,
      "target_is_alignment": isinstance(config_builder, SegmentalAttConfigBuilderRF),
    },
    "forward_step_func": analyze_gradients_forward_funcs._returnn_v2_forward_step,
    "forward_callback": analyze_gradients_forward_funcs._returnn_v2_get_forward_callback,
    "analyze_gradients_def": analyze_gradients_forward_funcs.analyze_gradients,
    "json_vocab_path": config_builder.variant_params["dependencies"].vocab_path,
  }

  if isinstance(config_builder, SegmentalAttConfigBuilderRF):
    if hdf_targets is None:
      realign_config = config_builder.get_realign_config(
        opts={
          "corpus_key": corpus_key,
          "realign_def": realignment_forward_funcs.model_realign_,
          "forward_step_func": realignment_forward_funcs._returnn_v2_forward_step,
          "forward_callback": realignment_forward_funcs._returnn_v2_get_forward_callback,
          "dataset_opts": {"target_is_alignment": False, "concat_num": concat_num},
          "batch_size": 15_000,
        })
      realign_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=realign_config,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        output_files=["scores.py.gz", "realignment.hdf"],
        mem_rqmt=6,
        time_rqmt=1,
      )
      hdf_targets = realign_job.out_files["realignment.hdf"]

      if ref_alignment_hdf is None:
        ref_alignment_hdf = hdf_targets
        ref_alignment_vocab_path = config_opts["json_vocab_path"]
        ref_alignment_blank_idx = config_builder.variant_params["dependencies"].model_hyperparameters.blank_idx

  config_opts.update({
    "ref_alignment_hdf": ref_alignment_hdf,
    "ref_alignment_blank_idx": ref_alignment_blank_idx,
    "ref_alignment_vocab_path": ref_alignment_vocab_path,
  })

  if hdf_targets is not None:
    config_opts["dataset_opts"]["hdf_targets"] = {corpus_key: hdf_targets}

  if concat_num is not None:
    config_opts["dataset_opts"]["repeat_in_between_last_frame_up_to_multiple_of"] = {
      "data": config_builder.red_factor,
    }
    # if bpe_codes_path is None, this means we use sentencepiece and we do not need to remove the in-between postfix
    if isinstance(config_builder, GlobalAttConfigBuilderRF) and config_builder.variant_params["dependencies"].bpe_codes_path is not None:
      config_opts["dataset_opts"]["remove_in_between_postfix"] = {
        "targets": config_builder.variant_params["dependencies"].model_hyperparameters.sos_idx,
      }

  analyze_gradients_config = config_builder.get_analyze_gradients_config(opts=config_opts)

  if concat_num is not None and seq_alias == "search":
    forward_dataset_dict = copy.deepcopy(analyze_gradients_config.config["forward_data"])
    del forward_dataset_dict["dataset"]["data_map"]["targets"]
    forward_dataset_dict["dataset"]["datasets"]["zip_dataset"]["targets"] = None
    concat_samples_hdf = ReturnnDumpHDFJob(
      forward_dataset_dict, returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
    ).out_hdf
    analyze_gradients_config.config["forward_data"]["dataset"]["datasets"]["zip_dataset"] = copy.deepcopy(analyze_gradients_config.config["forward_data"]["dataset"]["datasets"]["align"])
    analyze_gradients_config.config["forward_data"]["dataset"]["datasets"]["zip_dataset"]["files"] = [concat_samples_hdf]
    analyze_gradients_config.config["forward_data"] = analyze_gradients_config.config["forward_data"]["dataset"]

  analyze_gradients_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=analyze_gradients_config,
    returnn_root=returnn_root,
    returnn_python_exe=returnn_python_exe,
    output_files=["targets.hdf"],
    mem_rqmt=6,
    time_rqmt=1,
  )
  analyze_gradients_job.add_alias(f"{alias}/analysis/analyze_gradients_{seq_alias}")
  tk.register_output(analyze_gradients_job.get_one_alias(), analyze_gradients_job.out_files["targets.hdf"])


def calculate_search_errors(
        config_builder: ConfigBuilderRF,
        checkpoint: Union[PtCheckpoint, Dict],
        returnn_root: Path,
        returnn_python_exe: Path,
        search_hyps_file: Path,
        best_search_hyp_hdf: Path,
        alias: str,
        corpus_key: str,
):
  if "concat" in corpus_key:
    corpus_key_ = corpus_key
    corpus_key = corpus_key_.split("_")[0]
    concat_num = int(corpus_key_.split("_")[1].split("-")[1])
  else:
    concat_num = None
    corpus_key_ = corpus_key

  if isinstance(config_builder, SegmentalAttConfigBuilderRF):
    realign_config = config_builder.get_realign_config(
      opts={
        "corpus_key": corpus_key,
        "realign_def": realignment_forward_funcs.model_realign_,
        "forward_step_func": realignment_forward_funcs._returnn_v2_forward_step,
        "forward_callback": realignment_forward_funcs._returnn_v2_get_forward_callback,
        "dataset_opts": {"target_is_alignment": False, "concat_num": concat_num},
        "batch_size": 15_000,
      })
    realign_job = ReturnnForwardJobV2(
      model_checkpoint=checkpoint,
      returnn_config=realign_config,
      returnn_root=returnn_root,
      returnn_python_exe=returnn_python_exe,
      output_files=["scores.py.gz", "realignment.hdf"],
      mem_rqmt=6,
      time_rqmt=1,
    )
    ground_truth_scores_file = realign_job.out_files["scores.py.gz"]
    ground_truth_seqs_hdf = realign_job.out_files["realignment.hdf"]

    target_blank_idx = config_builder.variant_params["dependencies"].model_hyperparameters.blank_idx
  else:
    assert isinstance(config_builder, GlobalAttConfigBuilderRF)
    forward_config = config_builder.get_forward_config(
      opts={
        "corpus_key": corpus_key,
        "forward_def": global_att_forward_funcs.model_forward,
        "forward_step_func": global_att_forward_funcs._returnn_v2_forward_step,
        "forward_callback": global_att_forward_funcs._returnn_v2_get_forward_callback,
        "dataset_opts": {"target_is_alignment": False, "concat_num": concat_num},
        "batch_size": 15_000,
      }
    )
    forward_job = ReturnnForwardJobV2(
      model_checkpoint=checkpoint,
      returnn_config=forward_config,
      returnn_root=returnn_root,
      returnn_python_exe=returnn_python_exe,
      output_files=["scores.py.gz"],
      mem_rqmt=6,
      time_rqmt=1,
    )
    ground_truth_scores_file = forward_job.out_files["scores.py.gz"]

    forward_dataset = forward_config.config["forward_data"]
    forward_dataset["data_map"]["data"] = ("zip_dataset", "classes")
    del forward_dataset["data_map"]["targets"]
    forward_dataset["datasets"]["zip_dataset"]["targets"]["seq_postfix"] = None

    ground_truth_seqs_hdf = ReturnnDumpHDFJob(
      forward_dataset, returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
    ).out_hdf

    target_blank_idx = None

  calc_search_errors_job = CalcSearchErrorJobRF(
    ground_truth_scores_file=ground_truth_scores_file,
    search_hyps_file=search_hyps_file,
    search_seqs_hdf=best_search_hyp_hdf,
    ground_truth_hdf=ground_truth_seqs_hdf,
    target_blank_idx=target_blank_idx,
  )
  calc_search_errors_job.add_alias(f"{alias}/analysis/calc_search_errors")
  tk.register_output(calc_search_errors_job.get_one_alias(), calc_search_errors_job.out_search_errors)
