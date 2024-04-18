from typing import Tuple, Optional, List
import copy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.ctc import LibrispeechConformerCtcConfigBuilder


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import CtcTrainExperiment


def train_ctc_import_global(
        alias: str,
        config_builder: LibrispeechConformerCtcConfigBuilder,
        n_epochs_list: Tuple[int, ...] = (10,),
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_global_att_checkpoint/standard-training/%d-epochs" % (
      n_epochs
    )

    train_exp = CtcTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      # comment out for now because of hash but should be added back
      # train_opts={"preload_ignore_ctc": True}
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_score_label_model/output_prob",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield alias, checkpoint


def train_ctc_import_global_only_train_ctc(
        alias: str,
        config_builder: LibrispeechConformerCtcConfigBuilder,
        n_epochs_list: Tuple[int, ...] = (10,),
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_global_att_checkpoint/only-train-ctc/%d-epochs" % (
      n_epochs
    )

    train_exp = CtcTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_opts={"freeze_encoder": True, "preload_ignore_ctc": True}
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_score_label_model/output_prob",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield alias, checkpoint
