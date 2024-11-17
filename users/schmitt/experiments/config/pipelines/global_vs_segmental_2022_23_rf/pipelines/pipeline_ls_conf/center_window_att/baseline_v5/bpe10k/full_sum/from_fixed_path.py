from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ----------------- first fixed-path, then full-sum (Running) -----------------
  fixed_path_checkpoint_ep300 = None
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
  ):
    for fixed_path_train_alias, fixed_path_checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
    ):
      fixed_path_checkpoint_ep300 = fixed_path_checkpoint["checkpoints"][300]
      recog.center_window_returnn_frame_wise_beam_search(
        alias=fixed_path_train_alias,
        config_builder=config_builder,
        checkpoint=fixed_path_checkpoint,
      )

  for win_size, use_current_frame_in_readout in (
          (1, False),
          (None, True),
  ):
    for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
            win_size_list=(win_size,),
            blank_decoder_version=4,
            use_att_ctx_in_state=False,
            use_weight_feedback=False,
            use_correct_dim_tags=True,
            use_current_frame_in_readout=use_current_frame_in_readout,
    ):
      for full_sum_train_alias, full_sum_checkpoint in train.train_center_window_att(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs=300,
              use_speed_pert=True,
              batch_size=5_000,
              time_rqmt=80,
              gpu_mem_rqmt=24,
              checkpoint_alias="ctc-fixed-path-300ep",
              lr_scheduling_opts={"type": "const"},
              use_mgpu=False,
              checkpoint_path=fixed_path_checkpoint_ep300,
              training_type="full-sum",
      ):
        for epoch, chckpt in full_sum_checkpoint["checkpoints"].items():
          if epoch % 50 == 0 and epoch not in (50, 250):
            recog.center_window_returnn_frame_wise_beam_search(
              alias=full_sum_train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_aliases=(f"epoch-{epoch}",),
              run_analysis=True,
            )
          if epoch == 109:
            realign.center_window_returnn_realignment(
              alias=full_sum_train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_alias=f"epoch-{epoch}",
              plot=True,
            )
