from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v1 as center_window_baseline_v1
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v3 as center_window_baseline_v3
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v4 as center_window_baseline_v4
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import baseline_v1 as global_att_baseline_v1

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att import baseline_v1 as global_att_baseline_v1_no_rf


def run_exps():
  # this is needed in order to train the segmental attention model
  global_att_baseline_v1_no_rf.register_ctc_alignments()

  global_att_baseline_v1.run_exps()
  center_window_baseline_v3.run_exps()
  center_window_baseline_v4.run_exps()
