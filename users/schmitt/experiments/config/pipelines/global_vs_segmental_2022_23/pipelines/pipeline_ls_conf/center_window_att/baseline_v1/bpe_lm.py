from typing import Tuple

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
  standard_train_recog_center_window_att_import_global,
  returnn_recog_center_window_att_import_global,
  train_center_window_att_import_global,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias


def center_window_att_import_global_global_ctc_align_bpe_lm(
        win_size_list: Tuple[int, ...] = (5,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        lm_scale_list: Tuple[float, ...] = (0.4,),
        lm_recog_checkpoint_alias: str = "last",
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/bpe_lm/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )

        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
        )

        checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          train_opts={"num_epochs": n_epochs, "const_lr": const_lr},
        )

        for lm_scale in lm_scale_list:
          returnn_recog_center_window_att_import_global(
            config_builder=config_builder,
            alias=alias,
            checkpoint=checkpoints[n_epochs],
            recog_opts={
              "lm_opts": {
                "scale": lm_scale,
                "add_lm_eos_last_frame": True,
                "type": "trafo",
                "checkpoint_alias": lm_recog_checkpoint_alias
              },
              # "load_ignore_missing_vars": True,
              "batch_size": 1200000  # default batch size (2400000) leads to OOM
            },
            analyse=False
          )


def center_window_att_import_global_global_ctc_align_bpe_lm_ilm_correction(
        win_size_list: Tuple[int, ...] = (5,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        lm_scale_list: Tuple[float, ...] = (0.54,),
        ilm_scale_list: Tuple[float, ...] = (0.4,),
        mini_lstm_use_eos_list: Tuple[bool, ...] = (False,),
        add_lm_eos_last_frame_list: Tuple[bool, ...] = (True,),
        use_se_loss_list: Tuple[bool, ...] = (True,),
        mini_att_train_num_epochs: int = 10,
        lm_recog_checkpoint_alias: str = "last",
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/bpe_lm/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )

        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
        )

        for mini_lstm_use_eos in mini_lstm_use_eos_list:
          for lm_scale in lm_scale_list:
            for ilm_scale in ilm_scale_list:
              for add_lm_eos_last_frame in add_lm_eos_last_frame_list:
                for use_se_loss in use_se_loss_list:
                  standard_train_recog_center_window_att_import_global(
                    config_builder=config_builder,
                    alias=alias,
                    train_opts={
                      "num_epochs": n_epochs,
                      "const_lr": const_lr,
                      "train_mini_lstm_opts": {
                        "use_eos": mini_lstm_use_eos,
                        "num_epochs": mini_att_train_num_epochs,
                        "use_se_loss": use_se_loss
                      }
                    },
                    recog_opts={
                      "returnn_recog": True,
                      "lm_opts": {
                        "scale": lm_scale,
                        "add_lm_eos_last_frame": add_lm_eos_last_frame,
                        "type": "trafo",
                        "checkpoint_alias": lm_recog_checkpoint_alias
                      },
                      "ilm_correction_opts": {
                        "scale": ilm_scale,
                        "correct_eos": mini_lstm_use_eos,
                        "mini_att_train_num_epochs": mini_att_train_num_epochs,
                        "use_se_loss": use_se_loss
                      },
                      # "load_ignore_missing_vars": True,
                      "batch_size": 1200000  # default batch size (2400000) leads to OOM
                    },
                    analyse=False
                  )
