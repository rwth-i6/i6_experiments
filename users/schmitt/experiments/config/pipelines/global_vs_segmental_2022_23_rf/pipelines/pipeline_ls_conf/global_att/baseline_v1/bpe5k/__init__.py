from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=True,
          label_type="bpe5048",
  ):
    for train_alias, checkpoint in (
            (f"{model_alias}/import_glob.conformer.luca.bpe5k", external_checkpoints["luca-aed-bpe5k"]),
            (f"{model_alias}/import_glob.conformer.luca.bpe5k.w-ctc", external_checkpoints["luca-aed-bpe5k-w-ctc"]),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-luca",),
        # run_analysis=True,
        # analyze_gradients=True,
        # plot_att_weights=False,
      )

      for corpus_key in [
        "dev-other_0.1-5.1",
        "dev-other_5.1-10.1",
        "dev-other_10.1-15.1",
        "dev-other_15.1-20.1",
        "dev-other_20.1-25.1",
        "dev-other_25.1-30.1",
        "dev-other_30.1-35.1",
      ]:
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("best-luca",),
          corpus_keys=(corpus_key,),
        )

    for train_alias, checkpoint in train.train_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(300,),
      const_lr_list=(1e-4,),
      import_model_name="luca-aed-bpe5k",
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )




