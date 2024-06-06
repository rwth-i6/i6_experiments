from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v4 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog
)


def run_exps():
  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,), label_decoder_state="nb-lstm", use_att_ctx_in_state=False, use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(200, 300),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,), label_decoder_state="joint-lstm", use_att_ctx_in_state=False, use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(200, 300),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
          label_decoder_state="nb-lstm",
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          bpe_vocab_size=1056,
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(125,),
      use_speed_pert=True,
      batch_size=3_000,
      time_rqmt=80,
      use_mgpu=False,
    ):
      pass
      # recog.center_window_returnn_frame_wise_beam_search(
      #   alias=train_alias,
      #   config_builder=config_builder,
      #   checkpoint=checkpoint,
      # )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
          label_decoder_state="nb-lstm",
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          bpe_vocab_size=1056,
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(125,),
      use_speed_pert=True,
      batch_size=3_000,
      time_rqmt=1,
      use_mgpu=True,
    ):
      pass
      # recog.center_window_returnn_frame_wise_beam_search(
      #   alias=train_alias,
      #   config_builder=config_builder,
      #   checkpoint=checkpoint,
      # )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(1,),
          label_decoder_state="nb-lstm",
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(125,),
      use_speed_pert=True,
      batch_size=3_000,
      time_rqmt=80,
      use_mgpu=False,
      beam_size=100
    ):
      pass
      # recog.center_window_returnn_frame_wise_beam_search(
      #   alias=train_alias,
      #   config_builder=config_builder,
      #   checkpoint=checkpoint,
      # )
