from typing import Tuple, Optional


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingExperiment

def center_window_att_import_global_global_ctc_align_pos_pred_att_weight_interpolation(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        pos_pred_scale_list: Tuple[float, ...] = (.5,),
        use_normalization_list: Tuple[bool, ...] = (True,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for pos_pred_scale in pos_pred_scale_list:
          for use_normalization in use_normalization_list:
            alias = f"{base_alias}/couple_length_and_label_model/pos_pred_att_weight_interpolation/win-size-%d_%d-epochs_%f-const-lr/%s/pos_pred_scale-%f" % (
              win_size, n_epochs, const_lr, "w-normalization" if use_normalization else "wo-normalization", pos_pred_scale
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
                recog_opts={
                  "search_corpus_key": "dev-other"
                },
              )
              recog_exp.run_eval()
              if checkpoint_alias == analysis_checkpoint_alias:
                recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_expected_position_aux_loss(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        loss_scale_list: Tuple[float, ...] = (1.,),
        use_normalization_list: Tuple[bool, ...] = (True,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for loss_scale in loss_scale_list:
          for use_normalization in use_normalization_list:
            alias = f"{base_alias}/couple_length_and_label_model/expected_position_aux_loss/win-size-%d_%d-epochs_%f-const-lr/%s/loss_scale-%f" % (
              win_size, n_epochs, const_lr, "w-normalization" if use_normalization else "wo-normalization", loss_scale
            )

            train_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              expected_position_aux_loss_opts={"loss_scale": loss_scale, "use_normalization": use_normalization},
              length_model_opts={"use_embedding": False, "layer_class": "lstm"},
              use_old_global_att_to_seg_att_maker=False,
            )
            train_exp = SegmentalTrainExperiment(
              config_builder=train_config_builder,
              alias=alias,
              num_epochs=n_epochs,
            )
            checkpoints, model_dir, learning_rates = train_exp.run_train()

            recog_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              length_model_opts={
                "use_embedding": False,
                "layer_class": "lstm"
              },
              use_old_global_att_to_seg_att_maker=False,
            )

            for checkpoint_alias in ("last", "best", "best-4-avg"):
              recog_exp = ReturnnSegmentalAttDecodingExperiment(
                alias=alias,
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
