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


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
        dist_type_list: Tuple[str, ...] = ("gauss_double_exp_clipped",),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for std in std_list:
          for gauss_scale in gauss_scale_list:
            for dist_type in dist_type_list:
              alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/att_weight_interpolation/win-size-%d_%d-epochs_%f-const-lr/%s/std-%f_scale-%f" % (
                default_import_model_name, win_size, n_epochs, const_lr, dist_type, std, gauss_scale
              )

              config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
                use_old_global_att_to_seg_att_maker=False
              )

              standard_train_recog_center_window_att_import_global(
                config_builder=config_builder,
                alias=alias,
                train_opts={"num_epochs": n_epochs, "const_lr": const_lr}
              )


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation_plus_att_weight_recog_penalty(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for std in std_list:
          for gauss_scale in gauss_scale_list:
            dist_type = "gauss_double_exp_clipped"
            alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/att_weight_interpolation_plus_att_weight_recog_penalty/win-size-%d_%d-epochs_%f-const-lr/%s/std-%f_gauss_scale-%f" % (
              default_import_model_name, win_size, n_epochs, const_lr, dist_type, std, gauss_scale
            )

            train_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
              use_old_global_att_to_seg_att_maker=False
            )
            checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
              alias=alias,
              config_builder=train_config_builder,
              train_opts={"num_epochs": n_epochs, "const_lr": const_lr},
            )

            recog_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": "gauss_double_exp_clipped"},
              att_weight_recog_penalty_opts={
                "mult_weight": 0.005,
                "exp_weight": 2.0
              },
              use_old_global_att_to_seg_att_maker=False
            )
            returnn_recog_center_window_att_import_global(
              alias=alias,
              config_builder=recog_config_builder,
              checkpoint=checkpoints[n_epochs],
              recog_opts={"analyse": True, "search_corpus_key": "dev-other"},
            )
