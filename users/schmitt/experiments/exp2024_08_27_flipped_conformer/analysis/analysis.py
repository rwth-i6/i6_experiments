from sisyphus import tk, Path
from typing import Dict, List, Union, Optional
import copy

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.training import PtCheckpoint
from i6_core.text.processing import WriteToTextFileJob
from i6_core.returnn import ReturnnDumpHDFJob
from i6_core.corpus.segments import SegmentCorpusJob

from ..config_builder import AEDConfigBuilder
from .analyze_gradients import _returnn_v2_forward_step as analyze_gradients_forward_step
from .analyze_gradients import _returnn_v2_get_forward_callback as analyze_gradients_get_forward_callback
from .analyze_gradients import analyze_gradients as analyze_gradients_def
from .dump_gradients import _returnn_v2_forward_step as dump_gradients_forward_step
from .dump_gradients import _returnn_v2_get_forward_callback as dump_gradients_get_forward_callback
from .dump_gradients import dump_gradients as dump_gradients_def
from .dump_self_att import _returnn_v2_forward_step as dump_self_att_forward_step
from .dump_self_att import _returnn_v2_get_forward_callback as dump_self_att_get_forward_callback
from .dump_self_att import dump_self_att as dump_self_att_def
from .gmm_alignments import LIBRISPEECH_GMM_ALIGNMENT

from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.alignment.alignment import ForcedAlignOnScoreMatrixJob, CalculateSilenceStatistics


def analyze_gradients(
        config_builder: AEDConfigBuilder,
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
        do_forced_align_on_gradients: bool = False,
        plot_encoder_gradient_graph: bool = False,
        plot_encoder_layers: bool = False,
        plot_log_gradients: bool = False,
):
  assert seq_alias in ("ground-truth", "search")

  segment_file = WriteToTextFileJob(
    content=seq_tags
  )

  config_opts = {
    "dataset_opts": {
      "segment_paths": {corpus_key: segment_file.out_file},
      "corpus_key": corpus_key,
    },
    "forward_step_func": analyze_gradients_forward_step,
    "forward_callback": analyze_gradients_get_forward_callback,
    "analyze_gradients_def": analyze_gradients_def,
    "json_vocab_path": config_builder.vocab_opts["vocab_path"],
    "plot_encoder_gradient_graph": plot_encoder_gradient_graph,
    "plot_encoder_layers": plot_encoder_layers,
    "plot_log_gradients": plot_log_gradients
  }

  config_opts.update({
    "ref_alignment_hdf": ref_alignment_hdf,
    "ref_alignment_blank_idx": ref_alignment_blank_idx,
    "ref_alignment_vocab_path": ref_alignment_vocab_path,
  })

  if hdf_targets is not None:
    config_opts["dataset_opts"]["hdf_targets"] = {corpus_key: hdf_targets}

  analyze_gradients_config = config_builder.get_analyze_gradients_config(opts=config_opts)

  output_files = ["targets.hdf"]
  if plot_encoder_layers or plot_log_gradients:
    num_conformer_layers = config_builder.config_dict["encoder_opts"]["num_layers"]
    output_files += [f"enc-{n}" for n in range(num_conformer_layers)]

  analyze_gradients_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=analyze_gradients_config,
    returnn_root=returnn_root,
    returnn_python_exe=returnn_python_exe,
    output_files=output_files,
    mem_rqmt=12,
    time_rqmt=2,
    device="cpu",
    cpu_rqmt=3,
  )
  analyze_gradients_job.rqmt["sbatch_args"] = ["--exclude", "cn-260"]
  analyze_gradients_job.add_alias(f"{alias}/analysis/analyze_gradients_{seq_alias}/{'_'.join([tag.split('/')[-1] for tag in seq_tags])}")
  tk.register_output(analyze_gradients_job.get_one_alias(), analyze_gradients_job.out_files["targets.hdf"])

  # if do_forced_align_on_gradients:
  #   forced_align_job = ForcedAlignOnScoreMatrixJob(
  #     score_matrix_hdf=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/forward/ReturnnForwardJobV2.utMFNvPkCZvC/work/x_linear/log-prob-grads_wrt_x_linear_log-space/att_weights.hdf"),
  #   )
  #   forced_align_job.add_alias(f"{alias}/analysis/gradient_forced_align_{seq_alias}")
  #   tk.register_output(forced_align_job.get_one_alias(), forced_align_job.out_align)

  return analyze_gradients_job


def dump_gradients(
        config_builder: AEDConfigBuilder,
        seq_tags: Optional[List[str]],
        corpus_key: str,
        checkpoint: Union[PtCheckpoint, Dict],
        returnn_root: Path,
        returnn_python_exe: Path,
        alias: str,
        hdf_targets: Optional[Path],
        seq_alias: str,
        input_layer_name: str = "encoder_input",
):
  assert seq_alias in ("ground-truth", "search")

  if seq_tags is None:
    segment_corpus_job = SegmentCorpusJob(
      bliss_corpus=config_builder.dataset.corpus_paths[{"train": "train-other-960"}[corpus_key]],
      num_segments=100,
    )
    segment_file = list(segment_corpus_job.out_single_segment_files.values())[0]
  else:
    segment_file = WriteToTextFileJob(
      content=seq_tags
    ).out_file

  config_opts = {
    "dataset_opts": {
      "segment_paths": {corpus_key: segment_file},
      "corpus_key": corpus_key,
    },
    "forward_step_func": dump_gradients_forward_step,
    "forward_callback": dump_gradients_get_forward_callback,
    "dump_gradients_def": dump_gradients_def,
    "input_layer_name": input_layer_name,
  }

  if hdf_targets is not None:
    config_opts["dataset_opts"]["hdf_targets"] = {corpus_key: hdf_targets}

  dump_gradients_config = config_builder.get_dump_gradients_config(opts=config_opts)

  dump_gradients_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=dump_gradients_config,
    returnn_root=returnn_root,
    returnn_python_exe=returnn_python_exe,
    output_files=["gradients.hdf"],
    mem_rqmt=6,
    time_rqmt=5,
  )
  dump_gradients_job.add_alias(f"{alias}/analysis/dump_gradients_wrt_{input_layer_name}/{seq_alias}")
  tk.register_output(dump_gradients_job.get_one_alias(), dump_gradients_job.out_files["gradients.hdf"])

  return dump_gradients_job


def dump_self_att(
        config_builder: AEDConfigBuilder,
        seq_tags: Optional[List[str]],
        corpus_key: str,
        checkpoint: Union[PtCheckpoint, Dict],
        returnn_root: Path,
        returnn_python_exe: Path,
        alias: str,
        hdf_targets: Optional[Path],
        seq_alias: str,
):
  assert seq_alias in ("ground-truth", "search")

  if seq_tags is None:
    segment_corpus_job = SegmentCorpusJob(
      bliss_corpus=config_builder.dataset.corpus_paths[{"train": "train-other-960"}[corpus_key]],
      num_segments=100,
    )
    segment_file = list(segment_corpus_job.out_single_segment_files.values())[0]
  else:
    segment_file = WriteToTextFileJob(
      content=seq_tags
    ).out_file

  config_opts = {
    "dataset_opts": {
      "segment_paths": {corpus_key: segment_file},
      "corpus_key": corpus_key,
      "partition_epoch": 100,
    },
    "forward_step_func": dump_self_att_forward_step,
    "forward_callback": dump_self_att_get_forward_callback,
    "dump_self_att_def": dump_self_att_def,
  }

  if hdf_targets is not None:
    config_opts["dataset_opts"]["hdf_targets"] = {corpus_key: hdf_targets}

  dump_self_att_config = config_builder.get_dump_self_att_config(opts=config_opts)

  dump_self_att_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=dump_self_att_config,
    returnn_root=returnn_root,
    returnn_python_exe=returnn_python_exe,
    output_files=[f"self-att-energies_head-{i}.hdf" for i in range(9)],
    mem_rqmt=6,
    time_rqmt=1,
    device="cpu",
  )
  dump_self_att_job.add_alias(f"{alias}/analysis/dump_self_att/{seq_alias}")
  tk.register_output(dump_self_att_job.get_one_alias(), dump_self_att_job.out_files["self-att-energies_head-0.hdf"])

  return dump_self_att_job
