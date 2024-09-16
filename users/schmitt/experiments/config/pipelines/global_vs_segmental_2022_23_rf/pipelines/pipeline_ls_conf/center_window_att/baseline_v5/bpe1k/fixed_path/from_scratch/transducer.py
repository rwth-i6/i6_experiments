from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE1056_CTC_ALIGNMENT

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ------------------------------ Transducer ------------------------------

  # only current frame in readout (Running)
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          bpe_vocab_size=1056,
          use_correct_dim_tags=True,
  ):
    keep_epochs = list(range(200, 2_000, 200)) + [20, 100]
    for train_alias, checkpoint in train.train_center_window_att(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs=500,
            lr_scheduling_type="dyn_lr_piecewise_linear_epoch-wise_v2",
            batch_size=15_000,
            use_mgpu=True,
            use_speed_pert=False,
            training_type="fixed-path",
            accum_grad_multiple_step=4,
            hdf_targets=LibrispeechBPE1056_CTC_ALIGNMENT.alignment_paths,
            keep_epochs=keep_epochs,
            filter_data_len=19.5 * 16_000,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch in keep_epochs:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
            only_do_analysis=True,
          )

      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=PtCheckpoint(Path("/work/smt4/thulke/schmitt/sisyphus_work_dirs/03-09-24_aed_flipped_encoder/i6_core/returnn/training/ReturnnTrainingJob.l0Lvf5SbkMF6/output/models/epoch.067_.pt")),
        checkpoint_aliases=(f"epoch-{67}",),
        run_analysis=True,
        analysis_dump_gradients=True,
        only_do_analysis=True,
        corpus_keys=("train",),
        att_weight_seq_tags=None,
        analysis_ground_truth_hdf=LibrispeechBPE1056_CTC_ALIGNMENT.alignment_paths["train"],
      )
