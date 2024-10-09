from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.lm import LibrispeechTrafoLmConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import LmTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo.train import _returnn_v2_train_step, from_scratch_training


def train_lm(
        alias: str,
        config_builder: LibrispeechTrafoLmConfigBuilderRF,
        n_epochs: int,
        batch_size: int = 15_000,
        time_rqmt: int = 168,
        gpu_mem_rqmt: int = 11,
        use_mgpu: bool = True,
        max_seq_length: int = 75,
        accum_grad_multiple_step: int = 2,
        max_seqs: int = 100,
):
  alias += (
          f"/train_from_scratch/{n_epochs}-epochs"
          f"{'_mgpu-4' if use_mgpu else ''}_bs-{batch_size}"
          f"_max-seq-len-{max_seq_length}_accum-grad-{accum_grad_multiple_step}_max-seqs-{max_seqs}"
  )

  train_opts = {
    "train_step_func": _returnn_v2_train_step,
    "train_def": from_scratch_training,
    "lr_opts": {
      "type": "dyn_lr_piecewise_linear_epoch-wise_v2",
      "init_lr": 1e-5,
      "peak_lr": 1e-3,
      "num_epochs": n_epochs,
    },
    "max_seq_length": {"data": max_seq_length},
    "batch_size": batch_size,
    "accum_grad_multiple_step": accum_grad_multiple_step,
    "max_seqs": max_seqs,
  }

  train_rqmt = {
    "time": time_rqmt,
    "gpu_mem": gpu_mem_rqmt,
  }

  if use_mgpu:
    train_rqmt.update({
      "horovod_num_processes": 4,
      "distributed_launch_cmd": "torchrun"
    })
    train_opts["torch_distributed"] = {"reduce_type": "param", "param_sync_step": 100}

  train_exp = LmTrainExperiment(
    config_builder=config_builder,
    alias=alias,
    num_epochs=n_epochs,
    train_rqmt=train_rqmt,
    train_opts=train_opts,
  )
  checkpoints, model_dir, learning_rates = train_exp.run_train()
  checkpoint = {
    "model_dir": model_dir,
    "learning_rates": learning_rates,
    "key": "dev_loss_ce",
    "checkpoints": checkpoints,
    "n_epochs": n_epochs
  }

  yield alias, checkpoint
