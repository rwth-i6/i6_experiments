from typing import Tuple, Optional


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingExperiment


def center_window_att_import_global_global_ctc_align_att_weight_penalty_recog(
        win_size_list: Tuple[int, ...] = (129,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for mult_weight in (
                0.005, #0.01, 0.02, 0.03, 0.04, 0.05
        ):
          for exp_weight in (2.0,):
            alias = f"{base_alias}/att_weight_penalty/penalty_only_in_recog/win-size-%d_%d-epochs_%f-const-lr/mult-weight-%f_exp-weight-%f" % (
              win_size, n_epochs, const_lr, mult_weight, exp_weight
            )

            train_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              use_old_global_att_to_seg_att_maker=False
            )
            train_exp = SegmentalTrainExperiment(
              config_builder=train_config_builder,
              alias=alias,
              num_epochs=n_epochs,
            )
            checkpoints, model_dir, learning_rates = train_exp.run_train()

            w_penalty_alias = alias + "/w_penalty"
            recog_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              att_weight_recog_penalty_opts={
                "mult_weight": mult_weight,
                "exp_weight": exp_weight
              },
              use_old_global_att_to_seg_att_maker=False
            )

            for checkpoint_alias in ("last", "best", "best-4-avg"):
              recog_exp = ReturnnSegmentalAttDecodingExperiment(
                alias=w_penalty_alias,
                config_builder=recog_config_builder,
                checkpoint={
                  "model_dir": model_dir,
                  "learning_rates": learning_rates,
                  "key": "dev_score_label_model/output_prob",
                  "checkpoints": checkpoints,
                  "n_epochs": n_epochs
                },
                checkpoint_alias=checkpoint_alias,
                recog_opts={
                  "search_corpus_key": "dev-other"
                },
              )
              recog_exp.run_eval()
              if checkpoint_alias == analysis_checkpoint_alias:
                recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_att_weight_penalty_train(
        win_size_list: Tuple[int, ...] = (129,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        mult_weight_list: Tuple[float, ...] = (0.005,),
        exp_weight_list: Tuple[float, ...] = (2.0,),
        loss_scale_list: Tuple[float, ...] = (1.0,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for mult_weight in mult_weight_list:
          for exp_weight in exp_weight_list:
            for loss_scale in loss_scale_list:
              alias = f"{base_alias}/att_weight_penalty/penalty_in_train/win-size-%d_%d-epochs_%f-const-lr/mult-weight-%f_exp-weight-%f/loss-scale-%f" % (
                win_size, n_epochs, const_lr, mult_weight, exp_weight, loss_scale
              )

              train_config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                att_weight_recog_penalty_opts={
                  "mult_weight": mult_weight,
                  "exp_weight": exp_weight,
                  "use_as_loss": True,
                  "loss_scale": loss_scale,
                },
                use_old_global_att_to_seg_att_maker=False
              )
              train_exp = SegmentalTrainExperiment(
                config_builder=train_config_builder,
                alias=alias,
                num_epochs=n_epochs,
              )
              checkpoints, model_dir, learning_rates = train_exp.run_train()

              w_penalty_alias = alias + "/w_penalty"
              recog_config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                att_weight_recog_penalty_opts={
                  "mult_weight": mult_weight,
                  "exp_weight": exp_weight
                },
                use_old_global_att_to_seg_att_maker=False
              )

              for checkpoint_alias in ("last", "best", "best-4-avg"):
                recog_exp = ReturnnSegmentalAttDecodingExperiment(
                  alias=w_penalty_alias,
                  config_builder=recog_config_builder,
                  checkpoint={
                    "model_dir": model_dir,
                    "learning_rates": learning_rates,
                    "key": "dev_score_label_model/output_prob",
                    "checkpoints": checkpoints,
                    "n_epochs": n_epochs
                  },
                  checkpoint_alias=checkpoint_alias,
                  recog_opts={
                    "search_corpus_key": "dev-other"
                  },
                )
                recog_exp.run_eval()
                if checkpoint_alias == analysis_checkpoint_alias:
                  recog_exp.run_analysis()
