from typing import Tuple, Optional


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2.alias import alias as base_alias

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingExperiment, ReturnnSegmentalAttDecodingPipeline, RasrSegmentalAttDecodingExperiment


def center_window_att_import_global_global_ctc_align_baseline_from_scratch(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/baseline/train_from_scratch/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
          train_rqmt={"time": 168},
          train_opts={
            "dataset_opts": {"use_speed_pert": True, "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}},
            "import_model_train_epoch1": None,
            "lr_opts": {
              "type": "dynamic_lr",
              "dynamic_lr_opts": {
                "initial_lr": 0.0009 / 10,
                "peak_lr": 0.0009,
                "final_lr": 1e-6,
                "cycle_ep": 915,
                "total_ep": n_epochs,
                "n_step": 1350
              }
            }
          }
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()


def center_window_att_import_global_global_ctc_align_baseline(
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
        alias = f"{base_alias}/baseline/train_from_global_att_checkpoint/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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


def center_window_att_import_global_global_ctc_align_baseline_rasr_recog(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/baseline/train_from_global_att_checkpoint/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        for max_segment_len in (-1, 20):
          if max_segment_len == -1:
            search_rqmt = {"mem": 4, "time": 1, "gpu": 0}
            concurrent = 100
          else:
            search_rqmt = {"mem": 8, "time": 12, "gpu": 0}
            concurrent = 200
          for pruning_preset in ("simple-beam-search",):
            for open_vocab in (True,):
              RasrSegmentalAttDecodingExperiment(
                alias=alias,
                search_rqmt=search_rqmt,
                reduction_factor=960,
                reduction_subtrahend=399,
                max_segment_len=max_segment_len,
                concurrent=concurrent,
                open_vocab=open_vocab,
                checkpoint={
                    "model_dir": model_dir,
                    "learning_rates": learning_rates,
                    "key": "dev_score_label_model/output_prob",
                    "checkpoints": checkpoints,
                    "n_epochs": n_epochs
                },
                pruning_preset=pruning_preset,
                checkpoint_alias="last",
                config_builder=config_builder,
              ).run_eval()
