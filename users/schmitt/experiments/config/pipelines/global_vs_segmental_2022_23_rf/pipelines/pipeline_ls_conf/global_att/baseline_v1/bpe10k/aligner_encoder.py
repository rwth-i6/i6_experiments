from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import \
  external_checkpoints, default_import_model_name


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=False,
          use_att_ctx_in_state=False,
          replace_att_by_h_s=True,
  ):
    use_mgpu = False
    accum_grad_multiple_step = 2
    batch_size = 35_000
    n_epochs = 2_000

    for train_alias, checkpoint in train.train_global_att(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs=n_epochs,
            batch_size=batch_size,
            gpu_mem_rqmt=24,
            accum_grad_multiple_step=accum_grad_multiple_step,
            use_mgpu=use_mgpu,
            use_torch_amp=False,
            filter_data_len=19.5 * 16_000,
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        corpus_keys=("dev-other",),
      )
