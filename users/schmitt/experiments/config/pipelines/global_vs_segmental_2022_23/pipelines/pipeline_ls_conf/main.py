from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf_mod import ctc_aligns, seg_att, center_window_att


def run_exps():
  center_window_att.train_center_window_att_import_global_global_ctc_align()
