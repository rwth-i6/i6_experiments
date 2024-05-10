import copy
from typing import Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_tedlium2.global_att.zeineldeen import (
  zeineldeen
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_tedlium2.global_att import (
  recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_tedlium2.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_CURRENT_ROOT, RETURNN_EXE, RETURNN_EXE_NEW, RETURNN_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.tedlium2.label_singletons import TEDLIUM2BPE1057_CTC_ALIGNMENT
from i6_experiments.users.schmitt.alignment.alignment import AlignmentStatisticsJob
from i6_experiments.users.schmitt.alignment.att_weights import AttentionWeightStatisticsJob, ScatterAttentionWeightMonotonicityAgainstWERJob
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob, PlotAttentionWeightsJobV2

from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.text.processing import WriteToTextFileJob
from i6_core.returnn.training import Checkpoint

from sisyphus import tk
from sisyphus import Path


def run_exps():
  for model_alias, config_builder in zeineldeen.global_att_baseline():
    for train_alias, checkpoint, use_ctc, checkpoint_alias in (
            (f"{model_alias}/w-ctc", external_checkpoints["aed_w_ctc"], True, "best-4-avg"),
            (f"{model_alias}/w-ctc-mon", external_checkpoints["aed_w_ctc_mon"], True, "best-4-avg"),
            (f"{model_alias}/wo-ctc", external_checkpoints["aed_wo_ctc"], False, "best"),
            (f"{model_alias}/wo-ctc-mon", external_checkpoints["aed_wo_ctc_mon"], False, "best-4-avg"),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        corpus_key="dev",
        checkpoint_aliases=(checkpoint_alias,),
        run_analysis=True,
        analysis_opts={
          "att_weight_seq_tags": [
            "TED-LIUM-realease2/ElizabethGilbert_2009/62",
            "TED-LIUM-realease2/WadeDavis_2003/9",
            "TED-LIUM-realease2/ElizabethGilbert_2009/37",
            "TED-LIUM-realease2/WadeDavis_2003/21",
            "TED-LIUM-realease2/WadeDavis_2003/23",
            "TED-LIUM-realease2/BlaiseAguerayArcas_2007/14",
            "TED-LIUM-realease2/ElizabethGilbert_2009/27",
            "TED-LIUM-realease2/WadeDavis_2003/28",
            "TED-LIUM-realease2/AlGore_2009/22",
            "TED-LIUM-realease2/ElizabethGilbert_2009/72",
            "TED-LIUM-realease2/CraigVenter_2008/61",
            "TED-LIUM-realease2/WadeDavis_2003/42",
            "TED-LIUM-realease2/WadeDavis_2003/62",
            "TED-LIUM-realease2/WadeDavis_2003/25",
            "TED-LIUM-realease2/ElizabethGilbert_2009/79",
            "TED-LIUM-realease2/CraigVenter_2008/30",
            "TED-LIUM-realease2/WadeDavis_2003/38",
            "TED-LIUM-realease2/ElizabethGilbert_2009/31",
            "TED-LIUM-realease2/DavidMerrill_2009/12",
            "TED-LIUM-realease2/WadeDavis_2003/66",
            "TED-LIUM-realease2/ElizabethGilbert_2009/76",
            "TED-LIUM-realease2/ElizabethGilbert_2009/46",
            "TED-LIUM-realease2/WadeDavis_2003/90",
            "TED-LIUM-realease2/WadeDavis_2003/75",
            "TED-LIUM-realease2/ElizabethGilbert_2009/23",
            "TED-LIUM-realease2/WadeDavis_2003/45",
            "TED-LIUM-realease2/CraigVenter_2008/54",
            "TED-LIUM-realease2/WadeDavis_2003/35",
            "TED-LIUM-realease2/WadeDavis_2003/24",
            "TED-LIUM-realease2/BlaiseAguerayArcas_2007/4",
            "TED-LIUM-realease2/ElizabethGilbert_2009/51",
            "TED-LIUM-realease2/CraigVenter_2008/1",
            "TED-LIUM-realease2/ElizabethGilbert_2009/55",
            "TED-LIUM-realease2/ElizabethGilbert_2009/11",
            "TED-LIUM-realease2/WadeDavis_2003/58",
            "TED-LIUM-realease2/WadeDavis_2003/13",
            "TED-LIUM-realease2/ElizabethGilbert_2009/42",
            "TED-LIUM-realease2/ElizabethGilbert_2009/56",
            "TED-LIUM-realease2/BrianCox_2009U/15",
            "TED-LIUM-realease2/WadeDavis_2003/89",
            "TED-LIUM-realease2/BrianCox_2009U/8",
            "TED-LIUM-realease2/WadeDavis_2003/19",
            "TED-LIUM-realease2/BrianCox_2009U/1",
            "TED-LIUM-realease2/WadeDavis_2003/11",
            "TED-LIUM-realease2/BrianCox_2009U/2",
            "TED-LIUM-realease2/ElizabethGilbert_2009/4",
            "TED-LIUM-realease2/CraigVenter_2008/73",
            "TED-LIUM-realease2/BarrySchwartz_2005G/32",
            "TED-LIUM-realease2/BlaiseAguerayArcas_2007/19",
            "TED-LIUM-realease2/BarrySchwartz_2005G/66"
          ],
          "plot_energies": True,
          "dump_ctc": use_ctc,
          "calc_att_weight_stats": True,
          "dump_ctc_probs": use_ctc and train_alias.endswith("w-ctc"),
        },
      )

      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        corpus_key="test",
        checkpoint_aliases=(checkpoint_alias,),
      )


def register_ctc_alignments():
  for model_alias, config_builder in zeineldeen.global_att_baseline():
    # this is Mohammad's 5.6 WER model
    for train_alias, checkpoint, checkpoint_alias in (
            (f"{model_alias}/w-ctc", external_checkpoints["aed_w_ctc"], "best-4-avg"),
    ):
      ctc_alignments = {}

      for corpus_key in ("train", "dev", "test"):
        eval_config = config_builder.get_ctc_align_config(
          corpus_key=corpus_key,
          opts={
            "align_target": "data:targets",
            "hdf_filename": "alignments.hdf",
            "dataset_opts": {"seq_postfix": None}
          }
        )

        forward_job = ReturnnForwardJob(
          model_checkpoint=checkpoint,
          returnn_config=eval_config,
          returnn_root=RETURNN_CURRENT_ROOT,
          returnn_python_exe=RETURNN_EXE_NEW,
          hdf_outputs=["alignments.hdf"],
          eval_mode=True
        )
        forward_job.add_alias("%s/ctc_alignments/%s" % (train_alias, corpus_key))
        tk.register_output(forward_job.get_one_alias(), forward_job.out_hdf_files["alignments.hdf"])

        ctc_alignments[corpus_key] = forward_job.out_hdf_files["alignments.hdf"]

      ctc_alignments["devtrain"] = ctc_alignments["train"]
      TEDLIUM2BPE1057_CTC_ALIGNMENT.alignment_paths = copy.deepcopy(ctc_alignments)

      analysis_alias = f"datasets/Ted-Lium-2/ctc-alignments/{TEDLIUM2BPE1057_CTC_ALIGNMENT.alias}"
      for corpus_key in ctc_alignments:
        if corpus_key == "devtrain":
          continue
        statistics_job = AlignmentStatisticsJob(
          alignment=ctc_alignments[corpus_key],
          json_vocab=config_builder.dependencies.vocab_path,
          blank_idx=1057,
          silence_idx=None,
          returnn_root=RETURNN_ROOT,
          returnn_python_exe=RETURNN_EXE_NEW
        )
        statistics_job.add_alias(f"{analysis_alias}/statistics/{corpus_key}")
        tk.register_output(statistics_job.get_one_alias(), statistics_job.out_statistics)
