from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v7 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(None,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          bpe_vocab_size=10240,
          use_correct_dim_tags=True,
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            checkpoint_alias="albert-aed-trafo-decoder-bpe10k",
            lr_scheduling_type="const_then_linear",
            use_mgpu=False,
            batch_size=6_000,
            time_rqmt=80,
            use_speed_pert=True,
            gpu_mem_rqmt=24,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch == 3:
          realign.center_window_returnn_realignment(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_alias=f"epoch-{epoch}",
            plot=True,
            batch_size=4_000,
            time_rqmt=3,
          )
