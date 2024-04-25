from typing import Tuple, Optional, List


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2.baseline import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2.alias import alias as base_alias


def center_window_att_gaussian_att_weight_interpolation(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
        dist_type_list: Tuple[str, ...] = ("gauss",),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for std in std_list:
          for gauss_scale in gauss_scale_list:
            for dist_type in dist_type_list:
              alias = f"{base_alias}/att_weight_interpolation/win-size-%d/%s/std-%f_scale-%f" % (
                win_size, dist_type, std, gauss_scale
              )

              yield alias, get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
                length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
              )
