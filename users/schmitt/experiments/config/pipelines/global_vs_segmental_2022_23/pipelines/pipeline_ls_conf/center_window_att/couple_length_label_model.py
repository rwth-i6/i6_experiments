from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  default_import_model_name,
  get_center_window_att_config_builder,
  standard_train_recog_center_window_att_import_global,
  returnn_recog_center_window_att_import_global,
  train_center_window_att_import_global,
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment


def center_window_att_import_global_global_ctc_align_pos_pred_att_weight_interpolation(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        pos_pred_scale_list: Tuple[float, ...] = (.5,),
        use_normalization_list: Tuple[bool, ...] = (True,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for pos_pred_scale in pos_pred_scale_list:
          for use_normalization in use_normalization_list:
            alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/couple_length_and_label_model/pos_pred_att_weight_interpolation/win-size-%d_%d-epochs_%f-const-lr/%s/pos_pred_scale-%f" % (
              default_import_model_name, win_size, n_epochs, const_lr, "w-normalization" if use_normalization else "wo-normalization", pos_pred_scale
            )

            config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              pos_pred_att_weight_interpolation_opts={
                "pos_pred_scale": pos_pred_scale, "use_normalization": use_normalization
              },
              length_model_opts={"use_embedding": False, "layer_class": "lstm"},
              use_old_global_att_to_seg_att_maker=False,
            )

            standard_train_recog_center_window_att_import_global(
              config_builder=config_builder,
              alias=alias,
              train_opts={"num_epochs": n_epochs, "const_lr": const_lr},
            )


def center_window_att_import_global_global_ctc_align_expected_position_aux_loss(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        loss_scale_list: Tuple[float, ...] = (1.,),
        use_normalization_list: Tuple[bool, ...] = (True,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for loss_scale in loss_scale_list:
          for use_normalization in use_normalization_list:
            alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/couple_length_and_label_model/expected_position_aux_loss/win-size-%d_%d-epochs_%f-const-lr/%s/loss_scale-%f" % (
              default_import_model_name, win_size, n_epochs, const_lr, "w-normalization" if use_normalization else "wo-normalization", loss_scale
            )

            train_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              expected_position_aux_loss_opts={"loss_scale": loss_scale, "use_normalization": use_normalization},
              length_model_opts={"use_embedding": False, "layer_class": "lstm"},
              use_old_global_att_to_seg_att_maker=False,
            )
            checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
              alias=alias,
              config_builder=train_config_builder,
              train_opts={"num_epochs": n_epochs, "const_lr": const_lr},
            )

            recog_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              length_model_opts={"use_embedding": False, "layer_class": "lstm"},
              use_old_global_att_to_seg_att_maker=False,
            )
            returnn_recog_center_window_att_import_global(
              alias=alias,
              config_builder=recog_config_builder,
              checkpoint=checkpoints[n_epochs],
              recog_opts={"analyse": True, "search_corpus_key": "dev-other"},
            )
