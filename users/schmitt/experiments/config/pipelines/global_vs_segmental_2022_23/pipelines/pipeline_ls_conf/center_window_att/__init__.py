from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att import (
  simple_ablations,
  att_weight_penalty,
  att_weight_interpolation,
  concat_recognition,
  no_finetuning,
  couple_length_label_model,
  length_model_variants,
  chunking
)


def run_exps():
  simple_ablations.center_window_att_import_global_global_ctc_align_baseline(
    n_epochs_list=(100, 10), win_size_list=(1, 5, 9, 129)
  )
  simple_ablations.center_window_att_import_global_global_ctc_align_only_train_length_model(
    n_epochs_list=(10,), win_size_list=(5, 129)
  )
  simple_ablations.center_window_att_import_global_global_ctc_align_no_weight_feedback(
    n_epochs_list=(10,), win_size_list=(128,)
  )
  simple_ablations.center_window_att_import_global_global_ctc_align_no_weight_feedback_rasr_recog(
    n_epochs_list=(10,), win_size_list=(128,), max_segment_len_list=(-1,)
  )

  chunking.center_window_att_import_global_global_ctc_align_chunking(
    win_size_list=(5, 129), n_epochs_list=(10,), chunk_params_data_list=((170000, 85000), (200000, 100000))
  )

  att_weight_penalty.center_window_att_import_global_global_ctc_align_att_weight_penalty_recog()
  att_weight_penalty.center_window_att_import_global_global_ctc_align_att_weight_penalty_train(
    loss_scale_list=(1.,), mult_weight_list=(0.005,), exp_weight_list=(2.0,), win_size_list=(5, 129)
  )

  att_weight_interpolation.center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation(
    std_list=(1.,),
    gauss_scale_list=(1.,),
    n_epochs_list=(10,),
    win_size_list=(5,),
    dist_type_list=("gauss", "laplace")
  )
  att_weight_interpolation.center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation(
    std_list=(1.,),
    gauss_scale_list=(1.,),
    n_epochs_list=(10,),
    win_size_list=(129,),
    dist_type_list=("gauss",)
  )

  couple_length_label_model.center_window_att_import_global_global_ctc_align_pos_pred_att_weight_interpolation(
    pos_pred_scale_list=(0.5, 1.0),
    n_epochs_list=(10,),
    win_size_list=(5, 129,),
    use_normalization_list=(True, False)
  )
  couple_length_label_model.center_window_att_import_global_global_ctc_align_expected_position_aux_loss(
    loss_scale_list=(1.,), win_size_list=(5, 129), use_normalization_list=(True, False)
  )

  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_linear_layer()
  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_linear_layer_use_label_model_state()
  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_linear_layer_only_non_blank_ctx()
  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_use_label_model_state(
    n_epochs_list=(10, 20, 60, 100)
  )
  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_only_non_blank_ctx()
  length_model_variants.center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_only_non_blank_ctx_w_eos()


