from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ------------------- use current frame additionally to att in readout ---------------------
  for use_current_frame_in_readout in (True, False):
    for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
            win_size_list=(129,),
            blank_decoder_version=4,
            use_att_ctx_in_state=False,
            use_weight_feedback=False,
            use_current_frame_in_readout=use_current_frame_in_readout,
    ):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs_list=(300,),
              const_lr_list=(1e-4,),
              # batch_size=15_000,
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )

  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(499,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          use_current_frame_in_readout=True,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
            batch_size=10_000,
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

  # -------------------------- full-sum training --------------------------------

  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(None,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          bpe_vocab_size=5048,
          use_correct_dim_tags=True,
          use_current_frame_in_readout=True,
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            use_speed_pert=True,
            batch_size=3_000,
            time_rqmt=80,
            checkpoint_alias="luca-aed-bpe5k",
            lr_scheduling_type="const_then_linear",
            use_mgpu=False
    ):
      pass
      # for epoch, chckpt in checkpoint["checkpoints"].items():
      #   realign.center_window_returnn_realignment(
      #     alias=train_alias,
      #     config_builder=config_builder,
      #     checkpoint=chckpt,
      #     checkpoint_alias=f"epoch-{epoch}",
      #     plot=True,
      #   )
