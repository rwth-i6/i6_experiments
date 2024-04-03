from typing import Tuple, Optional


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
  returnn_recog_center_window_att_import_global,
  train_center_window_att_import_global,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
        dist_type_list: Tuple[str, ...] = ("gauss_double_exp_clipped",),
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for std in std_list:
          for gauss_scale in gauss_scale_list:
            for dist_type in dist_type_list:
              alias = f"{base_alias}/att_weight_interpolation/win-size-%d_%d-epochs_%f-const-lr/%s/std-%f_scale-%f" % (
                win_size, n_epochs, const_lr, dist_type, std, gauss_scale
              )

              config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
                use_old_global_att_to_seg_att_maker=False
              )

              train_exp = SegmentalTrainExperiment(
                config_builder=config_builder,
                alias=alias,
                num_epochs=n_epochs,
              )
              checkpoints, model_dir, learning_rates = train_exp.run_train()

              ilm_opts = {"type": ilm_type}
              if ilm_type == "mini_att":
                ilm_opts.update({
                  "use_se_loss": False,
                  "correct_eos": False,
                })
              ReturnnSegmentalAttDecodingPipeline(
                alias=alias,
                config_builder=config_builder,
                checkpoint={
                  "model_dir": model_dir,
                  "learning_rates": learning_rates,
                  "key": "dev_score_label_model/output_prob",
                  "checkpoints": checkpoints,
                  "n_epochs": n_epochs
                },
                checkpoint_aliases=checkpoint_aliases,
                beam_sizes=beam_size_list,
                lm_scales=lm_scale_list,
                lm_opts={"type": lm_type, "add_lm_eos_last_frame": True},
                ilm_scales=ilm_scale_list,
                ilm_opts=ilm_opts,
                run_analysis=run_analysis
              ).run()


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation_no_length_model_static_blank_penalty(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
        dist_type_list: Tuple[str, ...] = ("gauss_double_exp_clipped",),
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for std in std_list:
          for gauss_scale in gauss_scale_list:
            for dist_type in dist_type_list:
              alias = f"{base_alias}/att_weight_interpolation_no_length_model_static_blank_penalty/win-size-%d_%d-epochs_%f-const-lr/%s/std-%f_scale-%f" % (
                win_size, n_epochs, const_lr, dist_type, std, gauss_scale
              )

              train_config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
                use_old_global_att_to_seg_att_maker=False
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
                gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
                length_scale=0.0,
                blank_penalty=0.2,
              )

              ilm_opts = {"type": ilm_type}
              if ilm_type == "mini_att":
                ilm_opts.update({
                  "use_se_loss": False,
                  "correct_eos": False,
                })
              ReturnnSegmentalAttDecodingPipeline(
                alias=alias,
                config_builder=recog_config_builder,
                checkpoint={
                  "model_dir": model_dir,
                  "learning_rates": learning_rates,
                  "key": "dev_score_label_model/output_prob",
                  "checkpoints": checkpoints,
                  "n_epochs": n_epochs
                },
                checkpoint_aliases=checkpoint_aliases,
                beam_sizes=beam_size_list,
                lm_scales=lm_scale_list,
                lm_opts={"type": lm_type, "add_lm_eos_last_frame": True},
                ilm_scales=ilm_scale_list,
                ilm_opts=ilm_opts,
                run_analysis=run_analysis
              ).run()


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation_no_length_model_dynamic_blank_penalty(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
        dist_type_list: Tuple[str, ...] = ("gauss_double_exp_clipped",),
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for std in std_list:
          for gauss_scale in gauss_scale_list:
            for dist_type in dist_type_list:
              alias = f"{base_alias}/att_weight_interpolation_no_length_model_dynamic_blank_penalty/win-size-%d_%d-epochs_%f-const-lr/%s/std-%f_scale-%f" % (
                win_size, n_epochs, const_lr, dist_type, std, gauss_scale
              )

              train_config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
                use_old_global_att_to_seg_att_maker=False
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
                gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
                length_scale=0.0,
                blank_penalty="dynamic",
              )

              ilm_opts = {"type": ilm_type}
              if ilm_type == "mini_att":
                ilm_opts.update({
                  "use_se_loss": False,
                  "correct_eos": False,
                })
              ReturnnSegmentalAttDecodingPipeline(
                alias=alias,
                config_builder=recog_config_builder,
                checkpoint={
                  "model_dir": model_dir,
                  "learning_rates": learning_rates,
                  "key": "dev_score_label_model/output_prob",
                  "checkpoints": checkpoints,
                  "n_epochs": n_epochs
                },
                checkpoint_aliases=checkpoint_aliases,
                beam_sizes=beam_size_list,
                lm_scales=lm_scale_list,
                lm_opts={"type": lm_type, "add_lm_eos_last_frame": True},
                ilm_scales=ilm_scale_list,
                ilm_opts=ilm_opts,
                run_analysis=run_analysis
              ).run()


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation_static_seg_length_model(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
        dist_type_list: Tuple[str, ...] = ("gauss_double_exp_clipped",),
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for std in std_list:
          for gauss_scale in gauss_scale_list:
            for dist_type in dist_type_list:
              alias = f"{base_alias}/att_weight_interpolation_static_seg_length_model/win-size-%d_%d-epochs_%f-const-lr/%s/std-%f_scale-%f" % (
                win_size, n_epochs, const_lr, dist_type, std, gauss_scale
              )

              train_config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
                use_old_global_att_to_seg_att_maker=False
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
                gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale, "dist_type": dist_type},
                length_model_opts={
                  "type": "static-segmental",
                  "max_segment_len": 20,
                  "label_dependent_means": ctc_aligns.global_att_ctc_align.statistics_jobs["train"].out_label_dep_stats_var,
                },
              )

              ilm_opts = {"type": ilm_type}
              if ilm_type == "mini_att":
                ilm_opts.update({
                  "use_se_loss": False,
                  "correct_eos": False,
                })
              ReturnnSegmentalAttDecodingPipeline(
                alias=alias,
                config_builder=recog_config_builder,
                checkpoint={
                  "model_dir": model_dir,
                  "learning_rates": learning_rates,
                  "key": "dev_score_label_model/output_prob",
                  "checkpoints": checkpoints,
                  "n_epochs": n_epochs
                },
                checkpoint_aliases=checkpoint_aliases,
                beam_sizes=beam_size_list,
                lm_scales=lm_scale_list,
                lm_opts={"type": lm_type, "add_lm_eos_last_frame": True},
                ilm_scales=ilm_scale_list,
                ilm_opts=ilm_opts,
                run_analysis=run_analysis
              ).run()


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
            alias = f"{base_alias}/att_weight_interpolation_plus_att_weight_recog_penalty/win-size-%d_%d-epochs_%f-const-lr/%s/std-%f_gauss_scale-%f" % (
              win_size, n_epochs, const_lr, dist_type, std, gauss_scale
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
              recog_opts={"corpus_key": "dev-other"},
            )
