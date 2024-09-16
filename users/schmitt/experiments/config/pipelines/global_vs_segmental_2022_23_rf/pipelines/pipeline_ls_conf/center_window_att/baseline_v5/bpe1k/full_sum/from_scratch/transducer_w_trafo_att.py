from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ------------------------------ Transducer + global trafo att in Readout ------------------------------

  keep_epochs = [80, 160, 240, 320, 400]

  # no prev att in state
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(None,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          bpe_vocab_size=1056,
          use_correct_dim_tags=True,
          use_current_frame_in_readout=True,
          use_trafo_att=True,
  ):
    for train_alias, checkpoint in train.train_center_window_att(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs=500,
            use_speed_pert=True,
            batch_size=5_000,
            time_rqmt=80,
            use_mgpu=True,
            training_type="full-sum",
            keep_epochs=keep_epochs,
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
          )
