from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ------------------- Transducer ---------------------

  # window size 1 (like transducer) (Done)
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          use_weight_feedback=False,
          use_att_ctx_in_state=False,
          blank_decoder_version=4,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(537,),
            nb_loss_scale=6.0,
            use_mgpu=True,
            ctc_aux_loss_layers=(6, 12),
            ctc_aux_loss_focal_loss_factors=(1.0, 1.0),
            ctc_aux_loss_scales=(0.3, 1.0)
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
        lm_type="trafo",
        lm_scale_list=(0.7,),
        ilm_type="mini_att",
        ilm_scale_list=(0.4, 0.5),
        subtract_ilm_eos_score=True,
        use_recombination="sum",
        corpus_keys=("dev-other",),
        beam_size_list=(12,),
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        lm_type="trafo",
        lm_scale_list=(0.54, 0.6),
        ilm_type="mini_att",
        ilm_scale_list=(0.5, 0.6),
        subtract_ilm_eos_score=True,
        use_recombination="sum",
        corpus_keys=("dev-other",),
        beam_size_list=(12,),
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        analyze_gradients=True,
        run_analysis=True,
      )
