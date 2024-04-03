import copy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att import (
  recog, train
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_CURRENT_ROOT, RETURNN_EXE, RETURNN_EXE_NEW, RETURNN_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT
from i6_experiments.users.schmitt.alignment.alignment import AlignmentStatisticsJob
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob

from i6_core.returnn.forward import ReturnnForwardJob

from sisyphus import tk


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline():
    for train_alias, checkpoint in ((f"{model_alias}/no-finetuning", external_checkpoints[default_import_model_name]),):
      # recognition without continued training (this is then Mohammad's 5.6 WER model)
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
      )
      register_ctc_alignments(config_builder, train_alias, checkpoint)

    for train_alias, checkpoint in train.train_global_att_import_global(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(100,),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last", "best", "best-4-avg"),
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best",),
        lm_type="trafo",
        lm_scale_list=(0.6,),
        ilm_scale_list=(0.4,),
        ilm_type="mini_att",
        beam_size_list=(12, 50, 84)
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best",),
        lm_type="trafo",
        lm_scale_list=(0.54,),
        ilm_scale_list=(0.4,),
        ilm_type="mini_att",
        beam_size_list=(50, 84)
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best",),
        run_analysis=True,
        att_weight_seq_tags=[
          "dev-other/3660-6517-0005/3660-6517-0005",
          "dev-other/6467-62797-0001/6467-62797-0001",
          "dev-other/6467-62797-0002/6467-62797-0002",
          "dev-other/7697-105815-0015/7697-105815-0015",
          "dev-other/7697-105815-0051/7697-105815-0051",
          "dev-other/1650-167613-0018/1650-167613-0018",  # small window good
          "dev-other/8254-115543-0026/8254-115543-0026",
          "dev-other/6455-66379-0014/6455-66379-0014",  # small window bad
        ]
      )
      recog.global_att_returnn_label_sync_beam_search_concat_recog(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_alias="last",
        concat_nums=(2, 4)
      )

    for train_alias, checkpoint in train.train_global_att_import_global(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(40,),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best",),
        lm_type="trafo",
        lm_scale_list=(0.56, 0.58, 0.6),
        ilm_scale_list=(0.4, 0.5),
        ilm_type="mini_att",
      )

    for train_alias, checkpoint in train.train_global_att_import_global_freeze_encoder(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(300,),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
        lm_type="trafo",
        lm_scale_list=(0.54, 0.56, 0.58, 0.6),
        ilm_scale_list=(0.4, 0.5),
        ilm_type="mini_att",
      )


def register_ctc_alignments(config_builder, train_alias, checkpoint):
  ctc_alignments = {}

  for corpus_key in ("cv", "train", "dev-other"):
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
      returnn_python_exe=RETURNN_EXE_NEW if corpus_key == "dev-other" else RETURNN_EXE,
      hdf_outputs=["alignments.hdf"],
      eval_mode=True
    )
    forward_job.add_alias("%s/ctc_alignments/%s" % (train_alias, corpus_key))
    tk.register_output(forward_job.get_one_alias(), forward_job.out_hdf_files["alignments.hdf"])

    ctc_alignments[corpus_key] = forward_job.out_hdf_files["alignments.hdf"]

  ctc_alignments["devtrain"] = ctc_alignments["train"]
  LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths = copy.deepcopy(ctc_alignments)

  analysis_alias = f"datasets/LibriSpeech/ctc-alignments/{LibrispeechBPE10025_CTC_ALIGNMENT.alias}"
  for corpus_key in ctc_alignments:
    if corpus_key == "devtrain":
      continue
    statistics_job = AlignmentStatisticsJob(
      alignment=ctc_alignments[corpus_key],
      json_vocab=config_builder.dependencies.vocab_path,
      blank_idx=10025,
      silence_idx=None,
      returnn_root=RETURNN_ROOT,
      returnn_python_exe=RETURNN_EXE_NEW
    )
    statistics_job.add_alias(f"{analysis_alias}/statistics/{corpus_key}")
    tk.register_output(statistics_job.get_one_alias(), statistics_job.out_statistics)

    if corpus_key == "train":
      plot_align_job = PlotAlignmentJob(
        alignment_hdf=ctc_alignments[corpus_key],
        json_vocab_path=config_builder.dependencies.vocab_path,
        target_blank_idx=10025,
        segment_list=[
          "train-other-960/6157-40556-0085/6157-40556-0085",
          "train-other-960/421-124401-0044/421-124401-0044",
          "train-other-960/4211-3819-0000/4211-3819-0000",
          "train-other-960/5911-52164-0075/5911-52164-0075"
        ],
      )
      plot_align_job.add_alias(f"{analysis_alias}/plot_alignment/{corpus_key}")
      tk.register_output(plot_align_job.get_one_alias(), plot_align_job.out_plot_dir)
