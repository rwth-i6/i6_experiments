from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_tedlium2.global_att import zeineldeen as global_att_zeineldeen
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_tedlium2 import ctc_aligns


def run_exps():
  # ctc_aligns.calc_align_stats()
  global_att_zeineldeen.register_ctc_alignments()
  global_att_zeineldeen.run_exps()
