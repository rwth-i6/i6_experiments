from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  default_import_model_name,
  get_center_window_att_config_builder,
  standard_train_recog_center_window_att_import_global,
  returnn_recog_center_window_att_import_global,
  rasr_recog_center_window_att_import_global
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment


def center_window_att_import_global_global_ctc_align_chunking(
        win_size_list: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        chunk_params_data_list: Tuple[Tuple[int, int], ...] = ((170000, 85000),),
):
  chunking_opts = {
    "chunk_size_targets": 0,
    "chunk_step_targets": 0,
    "red_factor": 960,
    "red_subtrahend": 399,
  }

  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for chunk_params_data in chunk_params_data_list:
          chunk_size, chunk_step = chunk_params_data

          alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/chunking/win-size-%d_%d-epochs_%f-const-lr/chunk-size-%d_chunk-step-%d" % (
            default_import_model_name, win_size, n_epochs, const_lr, chunk_size, chunk_step
          )

          chunking_opts.update({
            "chunk_size_data": chunk_size,
            "chunk_step_data": chunk_step,
          })

          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
          )

          standard_train_recog_center_window_att_import_global(
            config_builder=config_builder,
            alias=alias,
            train_opts={"num_epochs": n_epochs, "const_lr": const_lr, "chunking_opts": chunking_opts},
          )
