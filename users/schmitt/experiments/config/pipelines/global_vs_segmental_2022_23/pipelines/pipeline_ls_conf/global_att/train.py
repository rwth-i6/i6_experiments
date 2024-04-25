from typing import Tuple, Optional, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import LibrispeechConformerGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import GlobalTrainExperiment


def train_global_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerGlobalAttentionConfigBuilder,
        n_epochs_list: Tuple[int, ...] = (10,),
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_global_att_checkpoint/standard-training/%d-epochs" % (
      n_epochs
    )

    train_exp = GlobalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_score_output/output_prob",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield alias, checkpoint


def train_global_att_import_global_freeze_encoder(
        alias: str,
        config_builder: LibrispeechConformerGlobalAttentionConfigBuilder,
        n_epochs_list: Tuple[int, ...] = (10,),
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_global_att_checkpoint/freeze-encoder/%d-epochs" % (
      n_epochs
    )

    train_exp = GlobalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_opts={"freeze_encoder": True}
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_score_output/output_prob",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield alias, checkpoint


def train_from_scratch(
        alias: str,
        config_builder: LibrispeechConformerGlobalAttentionConfigBuilder,
        n_epochs_list: Tuple[int, ...],
        use_ctc_loss: bool,
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_scratch/%d-epochs_%s" % (n_epochs, "w-ctc" if use_ctc_loss else "wo-ctc")

    train_exp = GlobalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt={
        "time": 168
      },
      train_opts={
        "no_ctc_loss": not use_ctc_loss,
        "dataset_opts": {
          # "use_speed_pert": True,
          "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}
        },
        "import_model_train_epoch1": None,
        "lr_opts": {
          "type": "dynamic_lr",
          "dynamic_lr_opts": {
            "initial_lr": 0.0009 / 10,
            "peak_lr": 0.0009,
            "final_lr": 1e-6,
            "cycle_ep": 915,
            "total_ep": 2035,
            "n_step": 1350
          }
        },
        "cleanup_old_models": {
          "keep_best_n": 4,
          "keep_last_n": 1,
          "keep": [n_epochs]
        },
        "batch_size": 2400000,
      }
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    yield alias, checkpoints
