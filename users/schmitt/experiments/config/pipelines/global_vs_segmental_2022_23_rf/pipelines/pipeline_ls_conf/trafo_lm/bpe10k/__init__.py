import os

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm import (
  train
)

import returnn.frontend as rf


def run_exps():
  model_kwargs_list = [
    {"num_layers": 24, "model_dim": 512},
    {
      "num_layers": 24,
      "model_dim": 1024,
      "pos_enc": None,
      "norm": "rf.RMSNorm",
      "ff": "rf.decoder.transformer.FeedForwardGated",
      "decoder_layer_opts": dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False))
    },
    {
      "num_layers": 32,
      "model_dim": 1024,
      "pos_enc": None,
      "norm": "rf.RMSNorm",
      "ff": "rf.decoder.transformer.FeedForwardGated",
      "decoder_layer_opts": dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False))
    }
  ]

  for model_kwargs in model_kwargs_list:
    if model_kwargs["model_dim"] == 512:
      batch_size = 5_000
      accum_grad_multiple_step = 2
      max_seqs = 100
    else:
      batch_size = 2_000
      accum_grad_multiple_step = 1
      max_seqs = 32

    for model_alias, config_builder in baseline.global_att_baseline_rf(
            label_type="bpe10025",
            **model_kwargs,
    ):
      for train_alias, checkpoint in train.train_lm(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs=40,
        use_mgpu=True,
        batch_size=batch_size,
        accum_grad_multiple_step=accum_grad_multiple_step,
        max_seqs=max_seqs,
      ):
        pass
