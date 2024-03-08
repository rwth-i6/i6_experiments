from typing import Tuple, Optional


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingExperiment


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
        dist_type_list: Tuple[str, ...] = ("gauss",),
        analysis_checkpoint_alias: Optional[str] = None,
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
                length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
                use_old_global_att_to_seg_att_maker=False
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
                    "search_corpus_key": "dev-other",
                  },
                )
                recog_exp.run_eval()

                if checkpoint_alias == analysis_checkpoint_alias:
                  recog_exp.run_analysis()
