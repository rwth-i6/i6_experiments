import os

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm import (
  train
)


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline_rf(
          label_type="bpe10025",
  ):
    for train_alias, checkpoint in train.train_lm(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=40,
      use_mgpu=True,
      batch_size=5_000,
    ):
      pass
