from typing import Tuple, Optional, List, Union, Dict, Callable
import copy

from i6_core.returnn.training import PtCheckpoint
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.corpus.segments import SegmentCorpusJob

from sisyphus import Path, tk

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechCtcAttConformerConfigBuilderRF, LibrispeechSegmentalAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LIBRISPEECH_CORPUS


def generic_returnn_realignment(
        alias: str,
        config_builder: Union[LibrispeechCtcAttConformerConfigBuilderRF, LibrispeechSegmentalAttConformerConfigBuilderRF],
        realign_def: Callable,
        forward_step_func: Callable,
        forward_callback: Callable,
        checkpoint: Union[PtCheckpoint, Dict],
        checkpoint_alias: str,
        plot: bool = False,
        batch_size: int = 15_000,
        time_rqmt: int = 1,
        corpus_key: str = "dev-other",
        use_multi_proc_dataset: bool = False,
        load_ignore_missing: bool = False,
        concurrent: int = 1,
        dump_scores: bool = True,
):
  alias += (
    f"/returnn_realignment/{checkpoint_alias}-checkpoint"
  )

  if isinstance(checkpoint, PtCheckpoint):
    checkpoint = checkpoint
  else:
    assert isinstance(checkpoint, dict)
    checkpoint = config_builder.get_recog_checkpoints(**checkpoint)[checkpoint_alias]


  opts = {
    "realign_def": realign_def,
    "forward_step_func": forward_step_func,
    "forward_callback": forward_callback,
    "dataset_opts": {
      "target_is_alignment": False, "corpus_key": corpus_key, "use_multi_proc": use_multi_proc_dataset},
    "batch_size": batch_size,
  }
  if load_ignore_missing:
    opts["preload_from_files"] = {
      "trained_model": {
        "filename": checkpoint,
        "ignore_missing": True
      }
    }

  if concurrent > 1:
    segment_corpus_job = SegmentCorpusJob(
      bliss_corpus=LIBRISPEECH_CORPUS.corpus_paths[{"train": "train-other-960"}[corpus_key]],
      num_segments=concurrent,
    )
    segment_paths = segment_corpus_job.out_single_segment_files.values()
  else:
    segment_paths = [None]

  alignments = []
  for i, segment_path in enumerate(segment_paths):
    opts["dataset_opts"]["segment_paths"] = {corpus_key: segment_path}

    realign_config = config_builder.get_realign_config(opts=opts)

    output_files = ["realignment.hdf"]
    if dump_scores:
      output_files.append("scores.py.gz")

    realign_job = ReturnnForwardJobV2(
      model_checkpoint=None if load_ignore_missing else checkpoint,
      returnn_config=realign_config,
      returnn_root=RETURNN_CURRENT_ROOT,
      returnn_python_exe=RETURNN_EXE_NEW,
      output_files=output_files,
      mem_rqmt=6,
      time_rqmt=time_rqmt,
    )
    realign_job.add_alias(f"{alias}/realignment_{corpus_key}{f'_{i}' if segment_path else ''}")
    tk.register_output(realign_job.get_one_alias(), realign_job.out_files["realignment.hdf"])

    alignments.append(realign_job.out_files["realignment.hdf"])

  # if plot:
  #   plot_alignment_job = PlotAlignmentJob(
  #     alignment_hdf=realign_job.out_files["realignment.hdf"],
  #     # ref_alignment_hdf=Path(
  #     #   "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline/no-finetuning/ctc_alignments/dev-other/output/alignments.hdf"),
  #     ref_alignment_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["dev-other"],
  #     json_vocab_path=config_builder.variant_params["dependencies"].vocab_path,
  #     ref_alignment_json_vocab_path=LibrispeechBPE10025_CTC_ALIGNMENT.vocab_path,
  #     target_blank_idx=config_builder.variant_params["dependencies"].model_hyperparameters.blank_idx,
  #     segment_list=[
  #       "dev-other/3660-6517-0005/3660-6517-0005",
  #       "dev-other/6467-62797-0001/6467-62797-0001",
  #       "dev-other/6467-62797-0002/6467-62797-0002",
  #       "dev-other/7697-105815-0015/7697-105815-0015",
  #       "dev-other/7697-105815-0051/7697-105815-0051",
  #       # high ctc-cog error
  #       "dev-other/6123-59150-0027/6123-59150-0027",
  #       # non-monotonic att weights
  #       "dev-other/1255-138279-0000/1255-138279-0000",
  #       "dev-other/7601-291468-0006/7601-291468-0006",
  #       "dev-other/7601-101619-0003/7601-101619-0003"
  #     ],
  #     ref_alignment_blank_idx=10025,
  #   )
  #   plot_alignment_job.add_alias(f"{alias}/plot_realignment")
  #   tk.register_output(plot_alignment_job.get_one_alias(), plot_alignment_job.out_plot_dir)

  return alignments
