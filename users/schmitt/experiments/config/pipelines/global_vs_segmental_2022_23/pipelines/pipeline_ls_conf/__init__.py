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
  center_window_att.center_window_att_import_global_global_ctc_align_no_finetuning()
  center_window_att.center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model()
  center_window_att.center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model_blank_penalty()

  seg_att.seg_att_import_global_global_ctc_align(n_epochs_list=(100,))
  seg_att.seg_att_import_global_global_ctc_align_align_augment(n_epochs_list=(10,))

  global_att.glob_att_import_global_no_finetuning()
  global_att.glob_att_import_global_diff_epochs_diff_lrs()
  global_att.glob_att_import_global_concat_recog()
  global_att.center_window_att_import_global_do_label_sync_search()
