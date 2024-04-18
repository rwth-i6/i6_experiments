from typing import Tuple, Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingExperiment


def center_window_att_import_global_global_ctc_align_baseline(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/simple_ablations/diff_win_sizes/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        for checkpoint_alias in ("last", "best", "best-4-avg"):
          recog_exp = ReturnnSegmentalAttDecodingExperiment(
            alias=alias,
            config_builder=config_builder,
            checkpoint={
              "model_dir": model_dir,
              "learning_rates": learning_rates,
              "key": "dev_score_label_model/output_prob",
              "checkpoints": checkpoints,
              "n_epochs": n_epochs
            },
            checkpoint_alias=checkpoint_alias,
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_no_weight_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/simple_ablations/no_weight_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=False,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        for checkpoint_alias in ("last", "best", "best-4-avg"):
          recog_exp = ReturnnSegmentalAttDecodingExperiment(
            alias=alias,
            config_builder=config_builder,
            checkpoint={
              "model_dir": model_dir,
              "learning_rates": learning_rates,
              "key": "dev_score_label_model/output_prob",
              "checkpoints": checkpoints,
              "n_epochs": n_epochs
            },
            checkpoint_alias=checkpoint_alias,
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_no_ctc_loss(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/simple_ablations/no_ctc_loss/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
          train_opts={"no_ctc_loss": True}
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        for checkpoint_alias in ("last", "best", "best-4-avg"):
          recog_exp = ReturnnSegmentalAttDecodingExperiment(
            alias=alias,
            config_builder=config_builder,
            checkpoint={
              "model_dir": model_dir,
              "learning_rates": learning_rates,
              "key": "dev_score_label_model/output_prob",
              "checkpoints": checkpoints,
              "n_epochs": n_epochs
            },
            checkpoint_alias=checkpoint_alias,
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()
