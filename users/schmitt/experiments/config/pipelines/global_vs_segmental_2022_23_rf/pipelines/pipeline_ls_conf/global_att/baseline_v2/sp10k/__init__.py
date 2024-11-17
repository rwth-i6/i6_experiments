from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v2 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline_rf(use_weight_feedback=True):
    for train_alias, checkpoint in (
            (f"{model_alias}/import_albert-aed-trafo-decoder-bpe10k", external_checkpoints["albert-aed-trafo-decoder-bpe10k"]),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("epoch-498",),
        plot_att_weights=False,
        analyze_gradients=True,
        run_analysis=True,
      )
      for corpus_key in [
        # "dev-other_0.1-5.1",
        # "dev-other_5.1-10.1",
        # "dev-other_10.1-15.1",
        # "dev-other_15.1-20.1",
        # "dev-other_20.1-25.1",
        # "dev-other_25.1-30.1",
        "dev-other_30.1-35.1",
      ]:
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("epoch-498",),
          corpus_keys=(corpus_key,),
        )

      for concat_num in (8,):
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("epoch-498",),
          batch_size=30_000,
          run_analysis=True,
          analyze_gradients=True,
          plot_att_weights=False,
          concat_num=concat_num,
          att_weight_seq_tags=[
            "dev-other/116-288045-0008/116-288045-0008;dev-other/116-288045-0009/116-288045-0009;dev-other/116-288045-0010/116-288045-0010;dev-other/116-288045-0011/116-288045-0011;dev-other/116-288045-0012/116-288045-0012;dev-other/116-288045-0013/116-288045-0013;dev-other/116-288045-0014/116-288045-0014;dev-other/116-288045-0015/116-288045-0015",
            "dev-other/116-288048-0000/116-288048-0000;dev-other/116-288048-0001/116-288048-0001;dev-other/116-288048-0002/116-288048-0002;dev-other/116-288048-0003/116-288048-0003;dev-other/116-288048-0004/116-288048-0004;dev-other/116-288048-0005/116-288048-0005;dev-other/116-288048-0006/116-288048-0006;dev-other/116-288048-0007/116-288048-0007"
          ]
        )




