from typing import Tuple, Optional


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import get_center_window_att_config_builder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingExperiment

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.alias import alias as base_alias


def center_window_att_import_global_global_ctc_align_concat_recognition(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        concat_num_list: Tuple[int, ...] = (2, 4),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for concat_num in concat_num_list:
        alias = f"{base_alias}/concat_recognition/win-size-%d_%d-epochs" % (
          win_size, n_epochs
        )

        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
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
            recog_opts={"search_corpus_key": "dev-other", "dataset_opts": {"concat_num": concat_num}},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()

        # train_exp = SegmentalTrainExperiment(
        #   config_builder=config_builder,
        #   alias=alias,
        #   n_epochs=n_epochs,
        #   import_model_train_epoch1=external_checkpoints[default_import_model_name],
        #   align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
        #   lr_opts={
        #     "type": "const_then_linear",
        #     "const_lr": 1e-4,
        #     "const_frac": 1 / 3,
        #     "final_lr": 1e-6,
        #     "num_epochs": n_epochs
        #   },
        # )
        # checkpoints, model_dir, learning_rates = train_exp.run_train()
        #
        # recog_center_window_att_import_global(
        #   alias=alias,
        #   config_builder=config_builder,
        #   concat_num=concat_num,
        #   checkpoint=checkpoints[n_epochs],
        #   analyse=True,
        #   search_corpus_key="dev-other",
        # )
