from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import default_import_model_name, get_center_window_att_config_builder, standard_train_recog_center_window_att_import_global, recog_center_window_att_import_global

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation_plus_att_weight_recog_penalty(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        concat_num_list: Tuple[int, ...] = (2, 4),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for concat_num in concat_num_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/concat_recognition/win-size-%d_%d-epochs" % (
          default_import_model_name, win_size, n_epochs
        )

        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
        )
        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          import_model_train_epoch1=external_checkpoints[default_import_model_name],
          align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
          lr_opts={
            "type": "const_then_linear",
            "const_lr": 1e-4,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          concat_num=concat_num,
          checkpoint=checkpoints[n_epochs],
          analyse=True,
          search_corpus_key="dev-other",
        )
