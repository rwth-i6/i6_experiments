import copy
from typing import Dict

from sisyphus import tk
from sisyphus import Path

from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.returnn.training import ReturnnTrainingJob

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_CURRENT_ROOT, RETURNN_EXE, RETURNN_EXE_NEW, RETURNN_ROOT
# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder import LibrispeechConformerGlobalAttentionConfigBuilder, LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import LibrispeechConformerGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob

from i6_experiments.users.schmitt.alignment.alignment import AlignmentStatisticsJob, AlignmentAddEosJob

directory_name = "models/ls_conformer/ctc_aligns"


class GlobalAttCtcAlignment:
  def __init__(self):
    model_type = "librispeech_conformer_glob_att"
    variant_name = "glob.conformer.mohammad.5.6"
    variant_params = copy.deepcopy(models[model_type][variant_name])
    self.base_alias = "%s/%s/%s" % (directory_name, model_type, variant_name)
    self.config_builder = config_builder = LibrispeechConformerGlobalAttentionConfigBuilder(
      dependencies=variant_params["dependencies"],
      variant_params=variant_params,
    )

    self._ctc_alignments = {}
    self._statistics_jobs = {}

    self.get_global_attention_ctc_align()
    self.get_statistics_jobs()

  @property
  def statistics_jobs(self) -> Dict[str, AlignmentStatisticsJob]:
    if self._statistics_jobs == {}:
      raise ValueError("You first need to run get_statistics_jobs()!")
    else:
      return self._statistics_jobs

  @statistics_jobs.setter
  def statistics_jobs(self, value):
    assert isinstance(value, dict)
    assert "train" in value and "cv" in value and "devtrain" in value
    self._statistics_jobs = value

  @property
  def ctc_alignments(self):
    if self._ctc_alignments == {}:
      raise ValueError("You first need to run get_global_attention_ctc_align()!")
    else:
      return self._ctc_alignments

  @ctc_alignments.setter
  def ctc_alignments(self, value):
    assert isinstance(value, dict)
    assert "train" in value and "cv" in value and "devtrain" in value
    self._ctc_alignments = value
    # plot_align_job = PlotAlignmentJob(
    #   alignment_hdf=value["train"],
    #   json_vocab_path=self.config_builder.dependencies.vocab_path,
    #   target_blank_idx=10025,
    #   segment_list=[
    #     "train-other-960/6157-40556-0085/6157-40556-0085",
    #     "train-other-960/421-124401-0044/421-124401-0044",
    #     "train-other-960/4211-3819-0000/4211-3819-0000",
    #     "train-other-960/5911-52164-0075/5911-52164-0075"
    #   ]
    # )
    # tk.register_output("plot_alignment", plot_align_job.out_plot_dir)

  def ctc_alignments_with_eos(self, segment_paths: Dict[str, Path], blank_idx: int, eos_idx: int):
    return {
      corpus_key: AlignmentAddEosJob(
        hdf_align_path=alignment_path,
        segment_file=segment_paths.get(corpus_key, None),
        blank_idx=blank_idx,
        eos_idx=eos_idx,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
      ).out_align for corpus_key, alignment_path in self.ctc_alignments.items()
    }

  def get_statistics_jobs(self):
    statistics_jobs = {}
    for corpus_key in self.ctc_alignments:
      statistics_job = AlignmentStatisticsJob(
        alignment=self.ctc_alignments[corpus_key],
        json_vocab=self.config_builder.dependencies.vocab_path,
        blank_idx=10025,
        silence_idx=20000,  # dummy idx which is larger than the vocab size
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE_NEW
      )
      statistics_job.add_alias("datasets/LibriSpeech/statistics/%s" % corpus_key)
      tk.register_output(statistics_job.get_one_alias(), statistics_job.out_statistics)
      statistics_jobs[corpus_key] = statistics_job
    self.statistics_jobs = statistics_jobs

  def get_global_attention_ctc_align(self):
    """

    :return:
    """
    ctc_aligns_global_att = {}

    for corpus_key in ("cv", "train", "dev-other"):
      eval_config = self.config_builder.get_ctc_align_config(
        corpus_key=corpus_key,
        opts={
          "align_target": "data:targets",
          "hdf_filename": "alignments.hdf",
          "dataset_opts": {"seq_postfix": None}
        }
      )

      forward_job = ReturnnForwardJob(
        model_checkpoint=external_checkpoints["glob.conformer.mohammad.5.6"],
        returnn_config=eval_config,
        returnn_root=RETURNN_CURRENT_ROOT,
        returnn_python_exe=RETURNN_EXE_NEW if corpus_key == "dev-other" else RETURNN_EXE,
        hdf_outputs=["alignments.hdf"],
        eval_mode=True
      )
      forward_job.add_alias("%s/dump_ctc_%s" % (self.base_alias, corpus_key))
      tk.register_output(forward_job.get_one_alias(), forward_job.out_hdf_files["alignments.hdf"])

      ctc_aligns_global_att[corpus_key] = forward_job.out_hdf_files["alignments.hdf"]

      # statistics_job = AlignmentStatisticsJob(
      #   alignment=ctc_aligns_global_att[corpus_key],
      #   json_vocab=self.config_builder.dependencies.vocab_path,
      #   blank_idx=10025,
      #   silence_idx=20000,  # dummy idx which is larger than the vocab size
      #   returnn_root=RETURNN_ROOT,
      #   returnn_python_exe=RETURNN_EXE_NEW
      # )
      # statistics_job.add_alias("datasets/LibriSpeech/statistics/%s" % corpus_key)
      # tk.register_output(statistics_job.get_one_alias(), statistics_job.out_statistics)

    ctc_aligns_global_att["devtrain"] = ctc_aligns_global_att["train"]

    self.ctc_alignments = ctc_aligns_global_att


# global_att_ctc_align = GlobalAttCtcAlignment()
