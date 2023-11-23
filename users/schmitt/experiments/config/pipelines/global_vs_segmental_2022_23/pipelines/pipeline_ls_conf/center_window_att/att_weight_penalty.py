from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import default_import_model_name, get_center_window_att_config_builder, standard_train_recog_center_window_att_import_global, recog_center_window_att_import_global

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment


def center_window_att_import_global_global_ctc_align_att_weight_penalty_recog(
        win_size_list: Tuple[int, ...] = (129,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for mult_weight in (
                0.005, #0.01, 0.02, 0.03, 0.04, 0.05
        ):
          for exp_weight in (2.0,):
            alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_att_weight_penalty_recog/win-size-%d_%d-epochs_%f-const-lr/mult-weight-%f_exp-weight-%f" % (
              default_import_model_name, win_size, n_epochs, const_lr, mult_weight, exp_weight
            )

            train_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
            )
            train_exp = SegmentalTrainExperiment(
              config_builder=train_config_builder,
              alias=alias,
              n_epochs=n_epochs,
              import_model_train_epoch1=external_checkpoints[default_import_model_name],
              align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
              lr_opts={
                "type": "const_then_linear",
                "const_lr": const_lr,
                "const_frac": 1 / 3,
                "final_lr": 1e-6,
                "num_epochs": n_epochs
              },
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
            )
            recog_center_window_att_import_global(
              alias=w_penalty_alias,
              config_builder=recog_config_builder,
              checkpoint=checkpoints[n_epochs],
              analyse=True,
              search_corpus_key="dev-other",
            )

            wo_penalty_alias = alias + "/wo_penalty"
            recog_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              att_weight_recog_penalty_opts=None,
            )
            recog_center_window_att_import_global(
              alias=wo_penalty_alias,
              config_builder=recog_config_builder,
              checkpoint=checkpoints[n_epochs],
              analyse=True,
              search_corpus_key="dev-other",
            )


def center_window_att_import_global_global_ctc_align_att_weight_penalty_train(
        win_size_list: Tuple[int, ...] = (129,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        mult_weight_list: Tuple[float, ...] = (0.005,),
        exp_weight_list: Tuple[float, ...] = (2.0,),
        loss_scale_list: Tuple[float, ...] = (1.0,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for mult_weight in mult_weight_list:
          for exp_weight in exp_weight_list:
            for loss_scale in loss_scale_list:
              alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_att_weight_penalty_train/win-size-%d_%d-epochs_%f-const-lr/mult-weight-%f_exp-weight-%f/loss-scale-%f" % (
                default_import_model_name, win_size, n_epochs, const_lr, mult_weight, exp_weight, loss_scale
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
              )
              train_exp = SegmentalTrainExperiment(
                config_builder=train_config_builder,
                alias=alias,
                n_epochs=n_epochs,
                import_model_train_epoch1=external_checkpoints[default_import_model_name],
                align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
                lr_opts={
                  "type": "const_then_linear",
                  "const_lr": const_lr,
                  "const_frac": 1 / 3,
                  "final_lr": 1e-6,
                  "num_epochs": n_epochs
                },
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
              )
              recog_center_window_att_import_global(
                alias=w_penalty_alias,
                config_builder=recog_config_builder,
                checkpoint=checkpoints[n_epochs],
                analyse=True,
                search_corpus_key="dev-other",
              )

              wo_penalty_alias = alias + "/wo_penalty"
              recog_config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                att_weight_recog_penalty_opts=None,
              )
              recog_center_window_att_import_global(
                alias=wo_penalty_alias,
                config_builder=recog_config_builder,
                checkpoint=checkpoints[n_epochs],
                analyse=True,
                search_corpus_key="dev-other",
              )
