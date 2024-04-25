from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.segmental_att import seg_att
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att import baseline_v2 as center_window_baseline_v2
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att import baseline_v1 as center_window_baseline_v1
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.ctc import baseline_v1 as ctc_baseline_v1
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att import baseline_v1 as global_att_baseline_v1


def run_exps():
  ctc_baseline_v1.run_exps()
  # we first need to register the ctc alignments from the global attention model
  # since this is needed for training the segmental model and also for jobs like PlotAttentionWeightsJob
  global_att_baseline_v1.register_ctc_alignments()

  global_att_baseline_v1.run_exps()

  center_window_baseline_v2.run_exps()
  center_window_baseline_v1.run_exps()

  # seg_att.seg_att_import_global_global_ctc_align_align_augment(n_epochs_list=(10,))
