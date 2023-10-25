from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns, seg_att, center_window_att, global_att


def run_exps():
  ctc_aligns.get_global_attention_ctc_align()

  center_window_att.center_window_att_import_global_global_ctc_align_weight_feedback(
    n_epochs_list=(100, 10),
  )
  center_window_att.center_window_att_import_global_global_ctc_align(
    n_epochs_list=(100, 10),
  )
  center_window_att.center_window_att_import_global_global_ctc_align_large_window_problems()
  center_window_att.center_window_att_import_global_global_ctc_align_att_weight_penalty_recog()
  # center_window_att.center_window_att_import_global_global_ctc_align_att_weight_penalty_train(
  #   loss_scale_list=(1.,), mult_weight_list=(0.005, 0.0025, 0.01), exp_weight_list=(2.0, 1.0, 0.5)
  # )
  # center_window_att.center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation(
  #   std_list=(0.5,), gauss_scale_list=(0.5, 0.75), n_epochs_list=(10, 100), win_size_list=(4, 128)
  # )
  center_window_att.center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation_plus_att_weight_recog_penalty(
    std_list=(0.5,), gauss_scale_list=(0.5,), n_epochs_list=(10,), win_size_list=(128,)
  )
  center_window_att.center_window_att_import_global_global_ctc_align_expected_position_aux_loss(
    loss_scale_list=(1.,)
  )
  center_window_att.center_window_att_import_global_global_ctc_align_length_model_no_label_feedback()
  center_window_att.center_window_att_import_global_global_ctc_align_length_model_diff_emb_size()

  seg_att.seg_att_import_global_global_ctc_align(n_epochs_list=(100,))
  seg_att.seg_att_import_global_global_ctc_align_align_augment(n_epochs_list=(10,))

  global_att.glob_att_import_global_no_finetuning()
  global_att.glob_att_import_global_diff_epochs_diff_lrs()
  global_att.glob_att_import_global_concat_recog()
  global_att.center_window_att_import_global_do_label_sync_search()
  global_att.center_window_att_import_global_do_label_sync_search(
    win_size_list=(128,), n_epochs_list=(10, 100), weight_feedback_list=(True,), center_window_use_eos=True
  )
