from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import default_import_model_name, get_center_window_att_config_builder, standard_train_recog_center_window_att_import_global, recog_center_window_att_import_global

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment


def center_window_att_import_global_global_ctc_align_pos_pred_att_weight_interpolation(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        pos_pred_scale_list: Tuple[float, ...] = (.5,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for gauss_scale in pos_pred_scale_list:
          alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_pos_pred_att_weight_interpolation/win-size-%d_%d-epochs_%f-const-lr/pos_pred_scale-%f" % (
            default_import_model_name, win_size, n_epochs, const_lr, gauss_scale
          )

          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            pos_pred_att_weight_interpolation_opts={"pos_pred_scale": gauss_scale},
            length_model_opts={"use_embedding": False, "layer_class": "lstm"},
          )

          standard_train_recog_center_window_att_import_global(
            config_builder=config_builder,
            alias=alias,
            n_epochs=n_epochs,
            const_lr=const_lr
          )


def center_window_att_import_global_global_ctc_align_expected_position_aux_loss(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        loss_scale_list: Tuple[float, ...] = (1.,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for loss_scale in loss_scale_list:
          alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_expected_position_aux_loss/win-size-%d_%d-epochs_%f-const-lr/loss_scale-%f" % (
            default_import_model_name, win_size, n_epochs, const_lr, loss_scale
          )

          train_config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            expected_position_aux_loss_opts={"loss_scale": loss_scale},
            length_model_opts={"use_embedding": False, "layer_class": "lstm"},
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

          recog_config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            length_model_opts={"use_embedding": False, "layer_class": "lstm"},
          )
          recog_center_window_att_import_global(
            alias=alias,
            config_builder=recog_config_builder,
            checkpoint=checkpoints[n_epochs],
            analyse=True,
            search_corpus_key="dev-other"
          )
