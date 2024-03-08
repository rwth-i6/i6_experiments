from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att import global_att
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.segmental_att import seg_att
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att import baseline_v1 as center_window_baseline_v1
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att import baseline_v2 as center_window_baseline_v2


def run_exps():
  # ctc_aligns.get_global_attention_ctc_align()

  center_window_baseline_v1.run_exps()
  center_window_baseline_v2.run_exps()

  # seg_att.seg_att_import_global_global_ctc_align(n_epochs_list=(100,))
  seg_att.seg_att_import_global_global_ctc_align_align_augment(n_epochs_list=(10,))

  global_att.glob_att_import_global_no_finetuning()
  global_att.glob_att_import_global_diff_epochs_diff_lrs(
    n_epochs_list=(200, 300)
  )
  global_att.glob_att_import_global_diff_epochs_diff_lrs(
    n_epochs_list=(100,), analysis_checkpoint_alias="best"
  )
  global_att.glob_att_import_global_concat_recog()
  # global_att.glob_att_import_global_bpe_lm(
  #   n_epochs_list=(100,),
  #   lm_type="trafo",
  #   lm_scale_list=(0.2, 0.3, 0.4, 0.5, 0.6),
  #   lm_recog_checkpoint_alias="best")
  # global_att.glob_att_import_global_bpe_lm(
  #   n_epochs_list=(100,),
  #   lm_type="lstm",
  #   lm_scale_list=(0.2, 0.3, 0.4, 0.5, 0.6),
  #   lm_recog_checkpoint_alias="best")
  global_att.glob_att_import_global_bpe_lm(
    n_epochs_list=(100,),
    lm_type="trafo",
    lm_scale_list=(0.6,),
    ilm_scale_list=(0.4,),
    ilm_type="mini_att",
    lm_recog_checkpoint_alias="best",
    beam_size_list=(12, 50, 84)
  )
  global_att.glob_att_import_global_bpe_lm(
    n_epochs_list=(100,),
    lm_type="trafo",
    lm_scale_list=(0.54,),
    ilm_scale_list=(0.4,),
    ilm_type="mini_att",
    lm_recog_checkpoint_alias="best",
    beam_size_list=(50, 84)
  )
  # global_att.glob_att_import_global_bpe_lm(
  #   n_epochs_list=(100,),
  #   lm_type="lstm",
  #   lm_scale_list=(0.5, 0.6, 0.7),
  #   ilm_scale_list=(0.4, 0.5, 0.6, 0.7),
  #   ilm_type="mini_att",
  #   lm_recog_checkpoint_alias="best")
  # global_att.glob_att_import_global_bpe_lm(
  #   n_epochs_list=(100,),
  #   lm_type="trafo",
  #   lm_scale_list=(0.2, 0.3, 0.4, 0.5),
  #   ilm_scale_list=(0.1, 0.2, 0.3, 0.4),
  #   ilm_type="zero_att",
  #   lm_recog_checkpoint_alias="best")
  # global_att.glob_att_import_global_bpe_lm(
  #   n_epochs_list=(100,),
  #   lm_type="lstm",
  #   lm_scale_list=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
  #   ilm_scale_list=(0.3, 0.4, 0.5, 0.6),
  #   ilm_type="zero_att",
  #   lm_recog_checkpoint_alias="best")
