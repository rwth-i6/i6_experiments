from typing import Tuple, Optional

from sisyphus import Path

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.rasr_recog import pruning_opts
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import RasrSegmentalAttDecodingExperiment


def center_window_att_import_global_global_ctc_align_arpa_lm(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        max_segment_len_list: Tuple[int, ...] = (-1,),
        pruning_list: Tuple[str, ...] = ("simple-beam-search",),
        lm_scale_list: Tuple[float, ...] = (0.1, 0.15, 0.2),
        lm_lookahead_scale_list: Tuple[float, ...] = (0.3, 0.4, 0.5),
        checkpoint_alias: str = "last",
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/word_lm/arpa_lm/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        for lm_scale in lm_scale_list:
          for lm_lookahead_scale in lm_lookahead_scale_list:
            for max_segment_len in max_segment_len_list:
              for pruning in pruning_list:
                RasrSegmentalAttDecodingExperiment(
                  search_rqmt={
                    "mem": 8,
                    "time": 2,
                    "gpu": 0
                  },
                  native_lstm2_so_path=Path(
                    "/work/asr3/zeyer/schmitt/dependencies/tf_native_libraries/lstm2/simon/CompileNativeOpJob.Q1JsD9yc8hfb/output/NativeLstm2.so"),
                  reduction_factor=960,
                  reduction_subtrahend=399,
                  max_segment_len=max_segment_len,
                  concurrent=100,
                  open_vocab=False,
                  checkpoint={
                    "model_dir": model_dir,
                    "learning_rates": learning_rates,
                    "key": "dev_score_label_model/output_prob",
                    "checkpoints": checkpoints,
                    "n_epochs": n_epochs
                  },
                  lm_opts={
                    "type": "ARPA",
                    "file": config_builder.dependencies.arpa_lm_paths["arpa"],
                    "scale": lm_scale,
                    "image": config_builder.dependencies.rasr_format_paths.arpa_lm_image_path,
                  },
                  lm_lookahead_opts={
                    "scale": lm_lookahead_scale,
                    "cache_size_high": 3000,
                    "cache_size_low": 2000,
                    "history_limit": 1,
                  },
                  checkpoint_alias=checkpoint_alias,
                  **pruning_opts[pruning],
                )


def center_window_att_import_global_global_ctc_align_lstm_lm(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        max_segment_len_list: Tuple[int, ...] = (-1,),
        pruning_list: Tuple[str, ...] = ("simple-beam-search",),
        lm_scale_list: Tuple[float, ...] = (0.1, 0.15, 0.2),
        lm_lookahead_scale_list: Tuple[Optional[float], ...] = (0.3, 0.4, 0.5),
        use_gpu: bool = False,
        concurrent: int = 100,
        checkpoint_alias: str = "last",
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/word_lm/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        for lm_scale in lm_scale_list:
          for lm_lookahead_scale in lm_lookahead_scale_list:
            for max_segment_len in max_segment_len_list:
              for pruning in pruning_list:
                RasrSegmentalAttDecodingExperiment(
                  search_rqmt={"mem": 12, "time": 6, "gpu": 1 if use_gpu else 0},
                  native_lstm2_so_path=Path(
                    "/work/asr3/zeyer/schmitt/dependencies/tf_native_libraries/lstm2/simon/CompileNativeOpJob.Q1JsD9yc8hfb/output/NativeLstm2.so"),
                  reduction_factor=960,
                  reduction_subtrahend=399,
                  max_segment_len=max_segment_len,
                  concurrent=concurrent,
                  open_vocab=False,
                  checkpoint={
                    "model_dir": model_dir,
                    "learning_rates": learning_rates,
                    "key": "dev_score_label_model/output_prob",
                    "checkpoints": checkpoints,
                    "n_epochs": n_epochs
                  },
                  checkpoint_alias=checkpoint_alias,
                  lm_opts={
                      "type": "tfrnn",
                      "allow_reduced_history": False,
                      "max_batch_size": 256,
                      "min_batch_size": 4,
                      "opt_batch_size": 64,
                      "scale": lm_scale,
                      "sort_batch_request": False,
                      "transform_output_negate": True,
                      "vocab_file": config_builder.dependencies.nn_lm_vocab_paths["kazuki-lstm"],
                      "meta_graph_file": config_builder.dependencies.nn_lm_meta_graph_paths["kazuki-lstm"],
                      "saved_model_file": config_builder.dependencies.nn_lm_checkpoint_paths["kazuki-lstm"],
                      "vocab_unknown_word": "<unk>",
                    },
                  lm_lookahead_opts=None if lm_lookahead_scale is None else {
                      "scale": lm_lookahead_scale,
                      "cache_size_high": 3000,
                      "cache_size_low": 2000,
                      "history_limit": 1,
                      "type": "ARPA",
                      "file": config_builder.dependencies.arpa_lm_paths["arpa"],
                      "image": config_builder.dependencies.rasr_format_paths.arpa_lm_image_path,
                      "separate_lookahead_lm": True
                    },
                  **pruning_opts[pruning],
                )
