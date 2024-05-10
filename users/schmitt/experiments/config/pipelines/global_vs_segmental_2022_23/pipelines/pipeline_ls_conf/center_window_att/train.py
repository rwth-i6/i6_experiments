from typing import Tuple, Optional, List, Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name


def train_center_window_att_import_global_global_ctc_align(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        n_epochs_list: Tuple[int, ...] = (10,),
        use_ctc_loss: bool = True,
):
  for n_epochs in n_epochs_list:
    alias += f"/train_from_global_att_checkpoint/standard-training/{n_epochs}-epochs_{'w-ctc' if use_ctc_loss else 'wo-ctc'}"

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_opts={
        "no_ctc_loss": not use_ctc_loss,
      }
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


def train_center_window_att_from_scratch(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        n_epochs_list: Tuple[int, ...],
        use_ctc_loss: bool
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_scratch/%d-epochs_%s" % (n_epochs, "w-ctc" if use_ctc_loss else "wo-ctc")

    train_exp = SegmentalTrainExperiment(
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
            "total_ep": n_epochs,
            "n_step": 1350
          }
        }
      }
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


def only_train_length_model_center_window_att_import_global_global_ctc_align(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        n_epochs_list: Tuple[int, ...] = (10,),
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_global_att_checkpoint/only-train-length-model/%d-epochs" % (n_epochs,)

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_opts={"only_train_length_model": True}
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


def train_center_window_att_import_global_global_ctc_align_freeze_encoder(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        n_epochs_list: Tuple[int, ...] = (10,),
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_global_att_checkpoint/freeze-encoder/%d-epochs" % (n_epochs,)

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_opts={"freeze_encoder": True}
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


def train_center_window_att_import_global_global_ctc_align_only_import_encoder(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        n_epochs_list: Tuple[int, ...] = (10,),
        time_rqmt: int = 30
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_global_att_checkpoint/only-import-encoder/%d-epochs" % (n_epochs,)

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt={"time": time_rqmt},
      train_opts={
        "preload_from_files": {
          "existing_model:": {
            "filename": external_checkpoints[default_import_model_name],
            "init_for_train": True,
            "ignore_missing": True,
            "ignore_params_prefixes": ["output/", "label_model/"],
          }
        }
      }
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
