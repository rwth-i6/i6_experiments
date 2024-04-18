from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1 import (
  simple_ablations,
  att_weight_penalty,
  att_weight_interpolation,
  concat_recognition,
  no_finetuning,
  couple_length_label_model,
  length_model_variants,
  chunking,
  bpe_lm,
  word_lm,
  rasr_recog
)


def run_exps():
  # rasr_recog.center_window_att_import_global_global_ctc_align(
  #   n_epochs_list=(10,), win_size_list=(5,), max_segment_len_list=(-1,)
  # )


  # no_finetuning.center_window_att_import_global_global_ctc_align_only_train_length_model_chunking(
  #   n_epochs_list=(10,), win_size_list=(129, 5)
  # )
  # no_finetuning.center_window_att_import_global_global_ctc_align_only_train_length_model_use_label_model_state_only_non_blank_ctx(
  #   n_epochs_list=(10, 20,), win_size_list=(129, 5)
  # )
  # no_finetuning.center_window_att_import_global_global_ctc_align_only_train_length_model_use_label_model_state_only_non_blank_ctx_eos()

  # chunking.center_window_att_import_global_global_ctc_align_chunking(
  #   win_size_list=(5, 129), n_epochs_list=(10,), chunk_params_data_list=((100000, 50000), (200000, 100000), (300000, 150000))
  # )

  # att_weight_penalty.center_window_att_import_global_global_ctc_align_att_weight_penalty_recog()
  # att_weight_penalty.center_window_att_import_global_global_ctc_align_att_weight_penalty_train(
  #   loss_scale_list=(1.,), mult_weight_list=(0.005,), exp_weight_list=(2.0,), win_size_list=(5, 129)
  # )
  pass


