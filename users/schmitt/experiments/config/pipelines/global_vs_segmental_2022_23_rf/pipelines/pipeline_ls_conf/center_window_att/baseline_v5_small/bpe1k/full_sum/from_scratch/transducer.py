from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5_small import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

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
    for train_alias, checkpoint in train.train_center_window_att(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs=500,
            use_speed_pert=True,
            batch_size=10_000,
            use_mgpu=True,
            keep_epochs=[100, 200, 300, 400, 500],
            lr_scheduling_type="dyn_lr_piecewise_linear_epoch-wise_v2",
            filter_data_len=19.5 * 16_000,  # 19.5s with 16kHz,
            training_type="full-sum",
            accum_grad_multiple_step=2,  # to be comparable to batch_size=5_000
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch % 20 == 0:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
          )
