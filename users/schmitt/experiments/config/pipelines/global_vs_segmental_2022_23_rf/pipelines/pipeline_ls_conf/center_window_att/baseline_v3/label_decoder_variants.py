from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3 import (
  get_config_builder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT


def run_exps():
  # ------------------- use current frame additionally to att in readout ---------------------
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(499,),
          use_current_frame_in_readout=True,
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
        checkpoint_aliases=("last",),
        run_analysis=True,
        only_do_analysis=True,
        analyze_gradients=True,
        analsis_analyze_gradients_plot_log_gradients=True,
        analysis_analyze_gradients_plot_encoder_layers=True,
        analysis_ground_truth_hdf=LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths["train"],
        att_weight_seq_tags=[
          "train-other-960/1246-124548-0042/1246-124548-0042",
          "train-other-960/40-222-0033/40-222-0033",
          "train-other-960/103-1240-0038/103-1240-0038",
        ],
        corpus_keys=("train",),
      )

