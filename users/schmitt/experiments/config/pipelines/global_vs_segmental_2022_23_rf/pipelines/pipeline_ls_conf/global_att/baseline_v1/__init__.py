from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from . import bpe1k, bpe5k, bpe10k


def run_exps():
  bpe10k.run_exps()
  bpe5k.run_exps()
  bpe1k.run_exps()  # still need checkpoint from Luca




