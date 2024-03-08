from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2 import (
  att_weight_interpolation,
  baseline,
  freeze_encoder
)


def run_exps():
  baseline.center_window_att_import_global_global_ctc_align_baseline(
    n_epochs_list=(200, 300),
    win_size_list=(5,),
  )
  baseline.center_window_att_import_global_global_ctc_align_baseline(
    n_epochs_list=(200,),
    win_size_list=(129,),
    checkpoint_aliases=("last",),
    run_analysis=True
  )
  baseline.center_window_att_import_global_global_ctc_align_baseline(
    n_epochs_list=(200,),
    win_size_list=(129,),
    lm_scale_list=(0.5, 0.54, 0.56, 0.58, 0.6, 0.62),
    lm_type="trafo",
    ilm_scale_list=(0.3, 0.4, 0.5),
    ilm_type="mini_att",
    checkpoint_aliases=("last",)
  )
  baseline.center_window_att_import_global_global_ctc_align_baseline_from_scratch(
    win_size_list=(129,), n_epochs_list=(2035,)
  )
  baseline.center_window_att_import_global_global_ctc_align_baseline_rasr_recog(
    n_epochs_list=(200,),
    win_size_list=(5,),
  )

  att_weight_interpolation.center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation(
    std_list=(1., 2.),
    gauss_scale_list=(1.,),
    n_epochs_list=(200, 300),
    win_size_list=(129,),
    dist_type_list=("gauss",)
  )

  freeze_encoder.center_window_att_import_global_global_ctc_align_baseline_freeze_encoder(
    n_epochs_list=(10,),
    win_size_list=(5, 129),
  )
