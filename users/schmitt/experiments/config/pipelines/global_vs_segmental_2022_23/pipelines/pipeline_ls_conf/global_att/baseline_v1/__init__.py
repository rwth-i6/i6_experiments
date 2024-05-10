import copy
from typing import Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att import (
  recog, train
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_CURRENT_ROOT, RETURNN_EXE, RETURNN_EXE_NEW, RETURNN_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT
from i6_experiments.users.schmitt.alignment.alignment import AlignmentStatisticsJob
from i6_experiments.users.schmitt.alignment.att_weights import AttentionWeightStatisticsJob
from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob, PlotAttentionWeightsJobV2

from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.text.processing import WriteToTextFileJob
from i6_core.returnn.training import Checkpoint

from sisyphus import tk


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline():
    # this is Mohammad's 5.6 WER model
    for train_alias, checkpoint in ((f"{model_alias}/no-finetuning", external_checkpoints[default_import_model_name]),):
      compare_center_of_gravity_and_ctc(config_builder, train_alias, checkpoint)
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
        run_analysis=True,
        analysis_opts={
          "att_weight_seq_tags": [
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
            # 10 non-monotonic att weights
            "dev-other/1630-141772-0015/1630-141772-0015",
            "dev-other/8173-294714-0041/8173-294714-0041"
            # 20 non-monotonic att weights
            "dev-other/7601-101619-0004/7601-101619-0004",
            "dev-other/4572-112375-0009/4572-112375-0009"
            # non-monotonic argmax
            "dev-other/7601-175351-0014/7601-175351-0014",
            "dev-other/6123-59150-0027/6123-59150-0027",
            "dev-other/1255-138279-0000/1255-138279-0000",
            "dev-other/3663-172528-0038/3663-172528-0038"
          ],
          "plot_energies": True,
          "dump_ctc_probs": True,
        }
      )

    # continue training for 1 epoch
    for train_alias, checkpoint in train.train_global_att_import_global(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(20,),
      use_ctc_loss=False,
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
      )

    # continue training for 5 epochs
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
        analysis_opts={
          "att_weight_seq_tags": [
            "dev-other/3660-6517-0005/3660-6517-0005",
            "dev-other/6467-62797-0001/6467-62797-0001",
            "dev-other/6467-62797-0002/6467-62797-0002",
            "dev-other/7697-105815-0015/7697-105815-0015",
            "dev-other/7697-105815-0051/7697-105815-0051",
            "dev-other/1650-167613-0018/1650-167613-0018",  # small window good
            "dev-other/8254-115543-0026/8254-115543-0026",
            "dev-other/6455-66379-0014/6455-66379-0014",  # small window bad
          ]
        }
      )
      recog.global_att_returnn_label_sync_beam_search_concat_recog(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_alias="last",
        concat_nums=(2, 4)
      )
    # train from scratch
    for use_ctc_loss in (True, False):
      for train_alias, checkpoints in train.train_from_scratch(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs_list=(100,),
              use_ctc_loss=use_ctc_loss
      ):
        plot_att_weights_during_training(
          train_alias=train_alias,
          checkpoints=checkpoints,
          config_builder=config_builder,
          use_ctc_loss=use_ctc_loss
        )


def register_ctc_alignments():
  for model_alias, config_builder in baseline.global_att_baseline():
    # this is Mohammad's 5.6 WER model
    for train_alias, checkpoint in ((f"{model_alias}/no-finetuning", external_checkpoints[default_import_model_name]),):
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


def compare_center_of_gravity_and_ctc(config_builder, train_alias, checkpoint):
  for corpus_key in ("dev-other", "dev-clean", "test-other", "test-clean"):
    eval_config = config_builder.get_dump_att_weight_config(
      corpus_key=corpus_key,
      opts={
        "hdf_filenames": {
          "att_weights": "att_weights.hdf",
          "targets": "targets.hdf",
          "ctc_alignment": "ctc_alignment.hdf"
        },
        "dataset_opts": {"seq_postfix": None}
      }
    )

    forward_job = ReturnnForwardJob(
      model_checkpoint=checkpoint,
      returnn_config=eval_config,
      returnn_root=RETURNN_CURRENT_ROOT,
      returnn_python_exe=RETURNN_EXE_NEW,
      hdf_outputs=["att_weights.hdf", "ctc_alignment.hdf", "targets.hdf"],
      eval_mode=True
    )

    compare_cog_to_ctc_job = AttentionWeightStatisticsJob(
      att_weights_hdf=forward_job.out_hdf_files["att_weights.hdf"],
      bpe_vocab=config_builder.dependencies.vocab_path,
      ctc_alignment_hdf=forward_job.out_hdf_files["ctc_alignment.hdf"],
      segment_file=None,
      ctc_blank_idx=10025,
    )
    compare_cog_to_ctc_job.add_alias("%s/compare_cog_and_ctc/%s" % (train_alias, corpus_key))
    tk.register_output(compare_cog_to_ctc_job.get_one_alias(), compare_cog_to_ctc_job.out_statistics)


def plot_att_weights_during_training(
        train_alias: str,
        checkpoints: Dict[int, Checkpoint],
        config_builder,
        use_ctc_loss: bool,
):
  att_weight_seq_tags = [
    "dev-other/3660-6517-0005/3660-6517-0005",
    "dev-other/6467-62797-0001/6467-62797-0001",
    "dev-other/6467-62797-0002/6467-62797-0002",
    "dev-other/7697-105815-0015/7697-105815-0015",
    "dev-other/7697-105815-0051/7697-105815-0051",
  ]
  write_cv_segments_to_file_job = WriteToTextFileJob(
    content=att_weight_seq_tags
  )
  # dump and plot att weights every 3 epochs
  for epoch in checkpoints:
    if epoch % 3 != 0:
      continue

    hdf_filenames = {
      "att_weights": "att_weights.hdf",
      "targets": "targets.hdf"
    }
    if use_ctc_loss:
      hdf_filenames["ctc_alignment"] = "ctc_alignment.hdf"

    eval_config = config_builder.get_dump_att_weight_config(
      corpus_key="dev-other",
      opts={
        "hdf_filenames": hdf_filenames,
        "dataset_opts": {
          "seq_postfix": None,
          "segment_paths": {"dev-other": write_cv_segments_to_file_job.out_file}
        },
        "network_epoch": epoch
      }
    )

    # if not use_ctc_loss:
    #   # layer cannot be loaded since not used in training
    #   del eval_config.config["network"]["ctc"]

    forward_job = ReturnnForwardJob(
      model_checkpoint=checkpoints[epoch],
      returnn_config=eval_config,
      returnn_root=RETURNN_CURRENT_ROOT,
      returnn_python_exe=RETURNN_EXE_NEW,
      hdf_outputs=list(hdf_filenames.values()),
      eval_mode=True
    )
    forward_job.add_alias("%s/att_weights_during_training/%s/epoch-%s/dump" % (train_alias, "dev-other", epoch))
    tk.register_output(forward_job.get_one_alias(), forward_job.out_hdf_files["att_weights.hdf"])

    plot_att_weights_job = PlotAttentionWeightsJobV2(
      att_weight_hdf=forward_job.out_hdf_files["att_weights.hdf"],
      targets_hdf=forward_job.out_hdf_files["targets.hdf"],
      seg_lens_hdf=None,
      seg_starts_hdf=None,
      center_positions_hdf=None,
      target_blank_idx=None,
      ref_alignment_blank_idx=10025,
      ref_alignment_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["dev-other"],
      json_vocab_path=config_builder.dependencies.vocab_path,
      ctc_alignment_hdf=forward_job.out_hdf_files.get("ctc_alignment.hdf"),
    )
    plot_att_weights_job.add_alias(
      "%s/att_weights_during_training/%s/epoch-%s/plot" % (train_alias, "dev-other", epoch))
    tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)
