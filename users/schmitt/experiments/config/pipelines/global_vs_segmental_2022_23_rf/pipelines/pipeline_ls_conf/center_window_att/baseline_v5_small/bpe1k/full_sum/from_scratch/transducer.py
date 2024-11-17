from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5_small import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE1056_CTC_ALIGNMENT

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ------------------------------ Transducer ------------------------------

  n_full_epochs = 45
  regularization_type = "v3"
  for (
    win_size,
    use_trafo_att,
  ) in [
    (1, False),  # standard transducer
    # (None, False),  # transducer with LSTM attention
    # (None, True),  # transducer with transformer attention
  ]:
    for gpu_mem_rqmt in (11,):
      if gpu_mem_rqmt == 24:
        use_mgpu = False
        accum_grad_multiple_step = 2
        batch_size = 24_000
        n_epochs = n_full_epochs * 20
      else:
        use_mgpu = True
        accum_grad_multiple_step = 1
        batch_size = 12_000
        n_epochs = n_full_epochs * 20 // 4

      keep_epochs_step = n_epochs // 10
      keep_epochs = list(range(keep_epochs_step, n_epochs, keep_epochs_step))
      for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
              win_size_list=(win_size,),
              blank_decoder_version=4,
              use_att_ctx_in_state=False,
              use_weight_feedback=False,
              bpe_vocab_size=1056,
              use_correct_dim_tags=True,
              behavior_version=21,
              use_trafo_att=use_trafo_att,
              use_current_frame_in_readout=win_size is None,
      ):
        for train_alias, checkpoint in train.train_center_window_att(
                alias=model_alias,
                config_builder=config_builder,
                n_epochs=n_epochs,
                batch_size=batch_size,
                use_mgpu=use_mgpu,
                use_speed_pert=False,
                training_type="full-sum",
                keep_epochs=keep_epochs,
                filter_data_len=19.5 * 16_000,
                ctc_aux_loss_layers=(4, 8),
                gpu_mem_rqmt=gpu_mem_rqmt,
                accum_grad_multiple_step=accum_grad_multiple_step,
                regularization_type=regularization_type,
        ):
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
          )

          for epoch, chckpt in checkpoint["checkpoints"].items():
            if epoch in keep_epochs:
              recog.center_window_returnn_frame_wise_beam_search(
                alias=train_alias,
                config_builder=config_builder,
                checkpoint=chckpt,
                checkpoint_aliases=(f"epoch-{epoch}",),
              )
