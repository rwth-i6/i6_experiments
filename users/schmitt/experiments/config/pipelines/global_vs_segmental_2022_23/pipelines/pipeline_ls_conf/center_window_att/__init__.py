from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att import (
  simple_ablations,
  att_weight_penalty,
  att_weight_interpolation,
  concat_recognition,
  no_finetuning,
  couple_length_label_model,
  length_model_variants
)


def run_exps():
  simple_ablations.center_window_att_import_global_global_ctc_align_baseline(
    n_epochs_list=(100, 10), win_size_list=(1, 4, 5, 9, 17, 33, 65, 128, 129)
  )
  simple_ablations.center_window_att_import_global_global_ctc_align_no_weight_feedback(
    n_epochs_list=(10,), win_size_list=(128,)
  )
  simple_ablations.center_window_att_import_global_global_ctc_align_no_weight_feedback(
    n_epochs_list=(10,), win_size_list=(4,), use_old_global_att_to_seg_att_maker=False
  )
  simple_ablations.center_window_att_import_global_global_ctc_align_no_weight_feedback_rasr_recog(
    n_epochs_list=(10,), win_size_list=(128,), max_segment_len_list=(-1,)
  )

  att_weight_penalty.center_window_att_import_global_global_ctc_align_att_weight_penalty_recog()
  att_weight_penalty.center_window_att_import_global_global_ctc_align_att_weight_penalty_train(
    loss_scale_list=(1.,), mult_weight_list=(0.005,), exp_weight_list=(2.0,)
  )

  att_weight_interpolation.center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation(
    std_list=(1., 2., 4., 8.),
    gauss_scale_list=(1.,),
    n_epochs_list=(10,),
    win_size_list=(5, 9, 17),
    dist_type_list=("gauss", "laplace")
  )

  couple_length_label_model.center_window_att_import_global_global_ctc_align_pos_pred_att_weight_interpolation(
    pos_pred_scale_list=(0.5, 1.0),
    n_epochs_list=(10,),
    win_size_list=(5, 129,)
  )
  couple_length_label_model.center_window_att_import_global_global_ctc_align_pos_pred_att_weight_interpolation(
    pos_pred_scale_list=(0.5, 1.0),
    n_epochs_list=(100,),
    win_size_list=(5, 129,)
  )
  couple_length_label_model.center_window_att_import_global_global_ctc_align_expected_position_aux_loss(
    loss_scale_list=(1.,), win_size_list=(5, 129),
  )
  couple_length_label_model.center_window_att_import_global_global_ctc_align_expected_position_aux_loss(
    loss_scale_list=(1.,), win_size_list=(5, 129), n_epochs_list=(100,)
  )

  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_linear_layer()
  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_no_label_feedback()
  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_only_non_blank_ctx()
  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_diff_emb_size(
    emb_size_list=(8, 16, 32)
  )

