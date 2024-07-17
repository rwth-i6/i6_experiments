from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v4 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)


def run_exps():
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(5,),
          label_decoder_state="joint-lstm",
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        use_recombination=None,
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        run_analysis=True,
        att_weight_seq_tags=[
          "dev-other/116-288045-0017/116-288045-0017",
          "dev-other/116-288045-0014/116-288045-0014",
        ]
      )
