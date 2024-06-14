from typing import Tuple, Optional, List, Union, Dict
import copy

from i6_core.returnn.training import PtCheckpoint
from i6_core.returnn.forward import ReturnnForwardJob, ReturnnForwardJobV2
from sisyphus import Path, tk

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import SegmentalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingPipeline, RasrSegmentalAttDecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.realignment_new import RasrRealignmentExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.recog import _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.realignment import model_realign_, _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import _returnn_v2_get_joint_model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT

def center_window_returnn_realignment(
        alias: str,
        config_builder: SegmentalAttConfigBuilderRF,
        checkpoint: Union[PtCheckpoint, Dict],
        checkpoint_alias: str,
        plot: bool = False,
):
  alias += (
    f"/returnn_realignment/{checkpoint_alias}-checkpoint"
  )

  if isinstance(checkpoint, PtCheckpoint):
    checkpoint = checkpoint
  else:
    assert isinstance(checkpoint, dict)
    checkpoint = config_builder.get_recog_checkpoints(**checkpoint)[checkpoint_alias]

  realign_config = config_builder.get_realign_config(
    opts={
      "corpus_key": "dev-other",
      "realign_def": model_realign_,
      "forward_step_func": _returnn_v2_forward_step,
      "forward_callback": _returnn_v2_get_forward_callback,
    })

  realign_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=realign_config,
    returnn_root=RETURNN_CURRENT_ROOT,
    returnn_python_exe=RETURNN_EXE_NEW,
    output_files=["scores.py.gz", "realignment.hdf"],
    mem_rqmt=6,
    time_rqmt=1,
  )
  realign_job.add_alias(f"{alias}/realignment")
  tk.register_output(realign_job.get_one_alias(), realign_job.out_files["realignment.hdf"])

  if plot:
    plot_alignment_job = PlotAlignmentJob(
      alignment_hdf=realign_job.out_files["realignment.hdf"],
      # ref_alignment_hdf=Path(
      #   "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline/no-finetuning/ctc_alignments/dev-other/output/alignments.hdf"),
      ref_alignment_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["dev-other"],
      json_vocab_path=Path(
        "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"),
      target_blank_idx=0,
      segment_list=[
        "dev-other/3660-6517-0005/3660-6517-0005",
        "dev-other/6467-62797-0001/6467-62797-0001",
        "dev-other/6467-62797-0002/6467-62797-0002",
        "dev-other/7697-105815-0015/7697-105815-0015",
        "dev-other/7697-105815-0051/7697-105815-0051",
        # high ctc-cog error
        "dev-other/6123-59150-0027/6123-59150-0027",
        # non-monotonic att weights
        "dev-other/1255-138279-0000/1255-138279-0000",
        "dev-other/7601-291468-0006/7601-291468-0006",
        "dev-other/7601-101619-0003/7601-101619-0003"
      ],
      ref_alignment_blank_idx=10025,
    )
    plot_alignment_job.add_alias(f"{alias}/plot_realignment")
    tk.register_output(plot_alignment_job.get_one_alias(), plot_alignment_job.out_plot_dir)
