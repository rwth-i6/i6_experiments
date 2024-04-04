from typing import Tuple, Optional


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2.alias import alias as base_alias

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingPipeline


def center_window_att_import_global_global_ctc_align_baseline_freeze_encoder(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/baseline/train_from_global_att_checkpoint/freeze_encoder/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
          train_opts={"freeze_encoder": True}
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        ilm_opts = {"type": ilm_type}
        if ilm_type == "mini_att":
          ilm_opts.update({
            "use_se_loss": False,
            "correct_eos": False,
          })
        ReturnnSegmentalAttDecodingPipeline(
          alias=alias,
          config_builder=config_builder,
          checkpoint={
            "model_dir": model_dir,
            "learning_rates": learning_rates,
            "key": "dev_score_label_model/output_prob",
            "checkpoints": checkpoints,
            "n_epochs": n_epochs
          },
          checkpoint_aliases=checkpoint_aliases,
          beam_sizes=beam_size_list,
          lm_scales=lm_scale_list,
          lm_opts={"type": lm_type, "add_lm_eos_last_frame": True},
          ilm_scales=ilm_scale_list,
          ilm_opts=ilm_opts,
          run_analysis=run_analysis
        ).run()
