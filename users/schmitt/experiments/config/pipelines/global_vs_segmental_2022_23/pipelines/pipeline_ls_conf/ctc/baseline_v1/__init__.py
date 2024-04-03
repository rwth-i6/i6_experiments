from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.ctc.baseline_v1 import \
  baseline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.ctc import (
  train
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_LABELS_WITH_SILENCE
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.gmm_alignments import LIBRISPEECH_GMM_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_CURRENT_ROOT, RETURNN_EXE_NEW, RETURNN_ROOT
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob
from i6_experiments.users.schmitt.alignment.alignment import AlignmentStatisticsJob, CompareBpeAndGmmAlignments

from i6_core.returnn.forward import ReturnnForwardJob

from sisyphus import tk, Path


def run_exps():
  for model_alias, config_builder in baseline.ctc_baseline():
    for train_alias, checkpoint in train.train_ctc_import_global(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(40,),
    ):
      checkpoint_dict = config_builder.get_recog_checkpoints(**checkpoint)

      # ctc alignments look much worse than in other experiment
      # get_ctc_alignments(
      #   checkpoint=checkpoint_dict["last"],
      #   config_builder=config_builder,
      #   train_alias=train_alias
      # )

  for model_alias, config_builder in baseline.ctc_baseline():
    for train_alias, checkpoint in train.train_ctc_import_global_only_train_ctc(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(20,),
    ):
      checkpoint_dict = config_builder.get_recog_checkpoints(**checkpoint)

      get_ctc_alignments(
        checkpoint=checkpoint_dict["last"],
        config_builder=config_builder,
        train_alias=train_alias
      )


def get_ctc_alignments(checkpoint, config_builder, train_alias):
  eval_config = config_builder.get_ctc_align_config(
    corpus_key="train",
    opts={
      "align_target": "data:targets",
      "hdf_filename": "alignments.hdf",
      "dataset_opts": {
        "hdf_targets": {"train": LibrispeechBPE10025_LABELS_WITH_SILENCE.label_paths["train"]},
      }
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
  forward_job.add_alias("%s/dump_ctc_%s" % (train_alias, "train"))
  tk.register_output(forward_job.get_one_alias(), forward_job.out_hdf_files["alignments.hdf"])
  alignment = forward_job.out_hdf_files["alignments.hdf"]

  plot_align_job = PlotAlignmentJob(
    alignment_hdf=alignment,
    json_vocab_path=config_builder.dependencies.vocab_path,
    target_blank_idx=10026,
    segment_list=[
      "train-other-960/6157-40556-0085/6157-40556-0085",
      "train-other-960/421-124401-0044/421-124401-0044",
      "train-other-960/4211-3819-0000/4211-3819-0000",
      "train-other-960/5911-52164-0075/5911-52164-0075"
    ],
    silence_idx=10025
  )
  plot_align_job.add_alias("%s/plot_ctc_%s" % (train_alias, "train"))
  tk.register_output(plot_align_job.get_one_alias(), plot_align_job.out_plot_dir)

  statistics_job = AlignmentStatisticsJob(
    alignment=alignment,
    json_vocab=config_builder.dependencies.vocab_path,
    blank_idx=10026,
    silence_idx=10025,
    returnn_root=RETURNN_ROOT,
    returnn_python_exe=RETURNN_EXE_NEW
  )
  statistics_job.add_alias("%s/ctc_statistics_%s" % (train_alias, "train"))
  tk.register_output(statistics_job.get_one_alias(), statistics_job.out_statistics)

  compare_alignments_job = CompareBpeAndGmmAlignments(
    bpe_align_hdf=alignment,
    bpe_vocab=config_builder.dependencies._vocab_path,
    bpe_blank_idx=10026,
    bpe_downsampling_factor=6,
    phoneme_alignment_cache=LIBRISPEECH_GMM_ALIGNMENT.alignment_caches["train"],
    phoneme_downsampling_factor=1,
    allophone_file=LIBRISPEECH_GMM_ALIGNMENT.allophone_file,
    silence_phone="[SILENCE]",
    returnn_python_exe=RETURNN_EXE_NEW,
    returnn_root=RETURNN_CURRENT_ROOT,
    time_rqmt=2
  )
  compare_alignments_job.add_alias("%s/compare_statistics_%s" % (train_alias, "train"))
  tk.register_output(compare_alignments_job.get_one_alias(), compare_alignments_job.out_statistics)
