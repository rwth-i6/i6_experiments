from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v1 as center_window_baseline_v1
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v3 as center_window_baseline_v3
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v4 as center_window_baseline_v4
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v5 as center_window_baseline_v5
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v5_small as center_window_baseline_v5_small
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v6 as center_window_baseline_v6
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v7 as center_window_baseline_v7
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import baseline_v1 as global_att_baseline_v1
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import baseline_v2 as global_att_baseline_v2

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att import baseline_v1 as global_att_baseline_v1_no_rf


def run_exps():
  # this is needed in order to train the segmental attention model
  global_att_baseline_v1_no_rf.register_ctc_alignments()

  global_att_baseline_v1.run_exps()
  global_att_baseline_v2.run_exps()
  # center_window_baseline_v1.rune_exps()
  center_window_baseline_v3.run_exps()
  center_window_baseline_v4.run_exps()
  center_window_baseline_v5.run_exps()
  # center_window_baseline_v5_small.run_exps()
  center_window_baseline_v6.run_exps()
  center_window_baseline_v7.run_exps()
