import os

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm import (
  train
)


def run_exps():
  model_kwargs_list = [
    {"num_layers": 24, "model_dim": 512},
    {"num_layers": 32, "model_dim": 1024},
  ]
  for model_kwargs in model_kwargs_list:
    for model_alias, config_builder in baseline.global_att_baseline_rf(
            label_type="bpe10025",
            **model_kwargs,
    ):
      for train_alias, checkpoint in train.train_lm(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs=40,
        use_mgpu=True,
        batch_size=5_000,
      ):
        pass
