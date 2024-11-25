from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  pass
  # # -------------------------- full-sum training --------------------------------
  #
  # for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
  #         win_size_list=(1,),
  #         blank_decoder_version=4,
  #         use_att_ctx_in_state=False,
  #         use_weight_feedback=False,
  #         bpe_vocab_size=5048,
  #         use_correct_dim_tags=True,
  # ):
  #   # -------------------------- from-global-att (bpe 5k) (Running) --------------------------------
  #   for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
  #           alias=model_alias,
  #           config_builder=config_builder,
  #           n_epochs_list=(300,),
  #           use_speed_pert=True,
  #           batch_size=3_000,
  #           time_rqmt=80,
  #           checkpoint_alias="luca-aed-bpe5k",
  #           lr_scheduling_type="const_then_linear",
  #           use_mgpu=False
  #   ):
  #     recog.center_window_returnn_frame_wise_beam_search(
  #       alias=train_alias,
  #       config_builder=config_builder,
  #       checkpoint=checkpoint,
  #     )
