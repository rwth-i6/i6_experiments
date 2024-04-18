from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2 import (
  att_weight_interpolation,
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att import (
  recog, train
)


def run_exps():
  for model_alias, config_builder in baseline.center_window_att_baseline(
    win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_import_global_global_ctc_align(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(300,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        run_analysis=True,
        att_weight_seq_tags=[
          "dev-other/3660-6517-0005/3660-6517-0005",
          "dev-other/6467-62797-0001/6467-62797-0001",
          "dev-other/6467-62797-0002/6467-62797-0002",
          "dev-other/7697-105815-0015/7697-105815-0015",
          "dev-other/7697-105815-0051/7697-105815-0051",
          "dev-other/1650-167613-0018/1650-167613-0018",  # small window good
          "dev-other/8254-115543-0026/8254-115543-0026",
          "dev-other/6455-66379-0014/6455-66379-0014",  # small window bad
        ]
      )

      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        lm_scale_list=(0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64),
        lm_type="trafo",
        ilm_scale_list=(0.4, 0.5),
        ilm_type="mini_att",
        checkpoint_aliases=("last",)
      )

      recog.center_window_returnn_frame_wise_beam_search_use_global_att_ilm(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        lm_scale_list=(0.54, 0.56, 0.58),
        lm_type="trafo",
        ilm_scale_list=(0.4, 0.5),
        checkpoint_aliases=("last",),
        ilm_correct_eos=False
      )

      recog.center_window_returnn_frame_wise_beam_search_use_global_att_ilm(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        lm_scale_list=(0.54, 0.56, 0.58),
        lm_type="trafo",
        ilm_scale_list=(0.4, 0.5),
        checkpoint_aliases=("last",),
        ilm_correct_eos=True
      )

      recog.center_window_rasr_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        max_segment_len_list=(-1, 20, 30),
      )

  for model_alias, config_builder in baseline.center_window_att_baseline(
          win_size_list=(129,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_import_global_global_ctc_align(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(200,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        run_analysis=False,
        att_weight_seq_tags=[
          "dev-other/3660-6517-0005/3660-6517-0005",
          "dev-other/6467-62797-0001/6467-62797-0001",
          "dev-other/6467-62797-0002/6467-62797-0002",
          "dev-other/7697-105815-0015/7697-105815-0015",
          "dev-other/7697-105815-0051/7697-105815-0051",
          "dev-other/1650-167613-0018/1650-167613-0018",  # small window good
          "dev-other/8254-115543-0026/8254-115543-0026",
          "dev-other/6455-66379-0014/6455-66379-0014",  # small window bad
        ]
      )

      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        lm_scale_list=(0.54, 0.56, 0.58, 0.6, 0.62, 0.64),
        lm_type="trafo",
        ilm_scale_list=(0.5, 0.6),
        ilm_type="mini_att",
        checkpoint_aliases=("last",)
      )

  for model_alias, config_builder in baseline.center_window_att_baseline(
          win_size_list=(5, 129),
  ):
    for train_alias, checkpoint in train.train_center_window_att_import_global_global_ctc_align(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(40,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        lm_type="trafo",
        ilm_type="mini_att",
        lm_scale_list=(0.54, 0.56, 0.58),
        ilm_scale_list=(0.4, 0.5),
        checkpoint_aliases=("last",)
      )

  for model_alias, config_builder in baseline.center_window_att_baseline(
          win_size_list=(129, 5),
  ):
    for train_alias, checkpoint in train.train_center_window_att_from_scratch(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(2035,),
            use_ctc_loss=True
    ):
      pass  # no recog for now

    for train_alias, checkpoint in train.train_center_window_att_from_scratch(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(2035,),
            use_ctc_loss=False
    ):
      pass  # no recog for now

  for model_alias, config_builder in baseline.center_window_att_baseline(
          win_size_list=(5, 129),
  ):
    for train_alias, checkpoint in train.train_center_window_att_import_global_global_ctc_align_freeze_encoder(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best",),
        lm_type="trafo",
        ilm_type="mini_att",
        lm_scale_list=(0.6, 0.62, 0.64),
        ilm_scale_list=(0.4, 0.5, 0.6)
      )

  for model_alias, config_builder in att_weight_interpolation.center_window_att_gaussian_att_weight_interpolation(
    win_size_list=(129,),
    std_list=(1., 2.),
    gauss_scale_list=(1.,),
    dist_type_list=("gauss",),
  ):
    for train_alias, checkpoint in train.train_center_window_att_import_global_global_ctc_align(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(300,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
        lm_type="trafo",
        ilm_type="mini_att",
        lm_scale_list=(0.52, 0.54, 0.56, 0.58, 0.6),
        ilm_scale_list=(0.3, 0.4, 0.5)
      )
