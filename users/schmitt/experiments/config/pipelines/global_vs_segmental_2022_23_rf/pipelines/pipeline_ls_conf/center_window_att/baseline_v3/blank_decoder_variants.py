from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ------------------------- Blank Decoder 7 (Done) -----------------------
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(5,),
          blank_decoder_version=7,
          blank_decoder_opts={"version": 7, "distance_embed_dim": 64, "max_distance": 10}
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

  # ------------------------- Blank Decoder 9 -----------------------
  # Running
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(None,),
          blank_decoder_version=9,
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
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        run_analysis=True,
        analyze_gradients=True,
      )


  # ------------------- blank decoder v4 ---------------------
  # Done
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1, 5,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
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
        use_recombination="sum",
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        lm_type="trafo",
        lm_scale_list=(0.54,),
        ilm_type="mini_att",
        ilm_scale_list=(0.44,),
        use_recombination="sum",
        corpus_keys=("dev-other", "test-other"),
        beam_size_list=(84,),
        subtract_ilm_eos_score=True,
      )

  # Done
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(5,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
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
        checkpoint_aliases=("last",),
        run_analysis=True,
        analyze_gradients=True,
      )

  # ------------------- blank decoder v1 ---------------------
  for blank_decoder_version in (1,):
    for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
            win_size_list=(129,),
            blank_decoder_version=blank_decoder_version,
            use_att_ctx_in_state=True,
            use_weight_feedback=True,
    ):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs_list=(100, 300),
              const_lr_list=(1e-4,),
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          use_recombination="sum",
        )

        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=(f"last",),
          run_analysis=True,
          only_do_analysis=True,
          analyze_gradients=True,
          analsis_analyze_gradients_plot_log_gradients=False,
          analysis_analyze_gradients_plot_encoder_layers=False,
          analysis_ground_truth_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["train"],
          att_weight_seq_tags=[
            "train-other-960/1246-124548-0042/1246-124548-0042",
            "train-other-960/40-222-0033/40-222-0033",
            "train-other-960/103-1240-0038/103-1240-0038",
          ],
          corpus_keys=("train",),
        )

        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=(f"last",),
          run_analysis=True,
          analyze_gradients_search=True,
          analsis_analyze_gradients_plot_log_gradients=False,
          analysis_analyze_gradients_plot_encoder_layers=False,
          analysis_ground_truth_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["dev-other"],
          att_weight_seq_tags=[
            "dev-other/7697-105815-0015/7697-105815-0015",
          ],
          corpus_keys=("dev-other",),
        )
