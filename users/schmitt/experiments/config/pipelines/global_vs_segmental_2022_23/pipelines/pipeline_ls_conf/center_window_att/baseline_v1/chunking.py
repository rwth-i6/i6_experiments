from typing import Tuple, Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingExperiment

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment


def center_window_att_import_global_global_ctc_align_chunking(
        win_size_list: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        chunk_params_data_list: Tuple[Tuple[int, int], ...] = ((170000, 85000),),
        analysis_checkpoint_alias: Optional[str] = None,
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

          alias = f"{base_alias}/chunking/win-size-%d_%d-epochs_%f-const-lr/chunk-size-%d_chunk-step-%d" % (
            win_size, n_epochs, const_lr, chunk_size, chunk_step
          )

          chunking_opts.update({
            "chunk_size_data": chunk_size,
            "chunk_step_data": chunk_step,
          })

          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
          )

          train_exp = SegmentalTrainExperiment(
            config_builder=config_builder,
            alias=alias,
            num_epochs=n_epochs,
            train_opts={"chunking_opts": chunking_opts},
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

          # standard_train_recog_center_window_att_import_global(
          #   config_builder=config_builder,
          #   alias=alias,
          #   train_opts={"num_epochs": n_epochs, "const_lr": const_lr, "chunking_opts": chunking_opts},
          # )
