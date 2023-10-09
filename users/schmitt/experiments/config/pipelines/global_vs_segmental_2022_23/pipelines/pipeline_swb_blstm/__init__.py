from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_swb_blstm import global_att, seg_att


def run_exps():
  global_att.glob_att()

  seg_att.seg_att()
