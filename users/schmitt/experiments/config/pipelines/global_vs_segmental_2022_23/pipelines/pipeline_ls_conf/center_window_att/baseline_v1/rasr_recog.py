from typing import Tuple

from sisyphus import Path

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import RasrSegmentalAttDecodingExperiment


pruning_opts = {
  "simple-beam-search": {
    "label_pruning": 12.0,
    "label_pruning_limit": 12,
    "word_end_pruning": 12.0,
    "word_end_pruning_limit": 12,
    "simple_beam_search": True,
    "full_sum_decoding": False,
    "allow_recombination": False,
  },
  "score-based": {
    "allow_recombination": True,
    "full_sum_decoding": True,
    "label_pruning": 8.0,
    "label_pruning_limit": 128,
    "word_end_pruning": 8.0,
    "word_end_pruning_limit": 128,
    "simple_beam_search": False,
  }
}


def center_window_att_import_global_global_ctc_align(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        max_segment_len_list: Tuple[int, ...] = (-1,),
        pruning_list: Tuple[str, ...] = ("simple-beam-search",),
        open_vocab_list: Tuple[bool, ...] = (True,),
        checkpoint_alias: str = "last",
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/rasr_recog/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        for max_segment_len in max_segment_len_list:
          for pruning in pruning_list:
            for open_vocab in open_vocab_list:
              RasrSegmentalAttDecodingExperiment(
                search_rqmt={"mem": 4, "time": 1, "gpu": 0},
                native_lstm2_so_path=Path("/work/asr3/zeyer/schmitt/dependencies/tf_native_libraries/lstm2/simon/CompileNativeOpJob.Q1JsD9yc8hfb/output/NativeLstm2.so"),
                reduction_factor=960,
                reduction_subtrahend=399,
                max_segment_len=max_segment_len,
                concurrent=100,
                open_vocab=open_vocab,
                checkpoint={
                    "model_dir": model_dir,
                    "learning_rates": learning_rates,
                    "key": "dev_score_label_model/output_prob",
                    "checkpoints": checkpoints,
                    "n_epochs": n_epochs
                },
                checkpoint_alias=checkpoint_alias,
                **pruning_opts[pruning],
              )
