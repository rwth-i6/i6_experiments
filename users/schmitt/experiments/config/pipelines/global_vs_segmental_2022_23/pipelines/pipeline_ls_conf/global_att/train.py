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
