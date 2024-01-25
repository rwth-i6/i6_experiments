from typing import Tuple

from sisyphus import Path

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  default_import_model_name,
  get_center_window_att_config_builder,
  standard_train_recog_center_window_att_import_global,
  returnn_recog_center_window_att_import_global,
  rasr_recog_center_window_att_import_global,
  train_center_window_att_import_global,
  rasr_realign_center_window_att_import_global,
)


def center_window_att_import_global_global_ctc_align_baseline(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/simple_ablations/diff_win_sizes/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          train_opts={"num_epochs": n_epochs, "const_lr": const_lr},
        )


def center_window_att_import_global_global_ctc_align_no_weight_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/simple_ablations/no_weight_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=False,
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          train_opts={"num_epochs": n_epochs, "const_lr": const_lr},
        )


def center_window_att_import_global_global_ctc_align_no_ctc_loss(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/simple_ablations/no_ctc_loss/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          train_opts={"num_epochs": n_epochs, "const_lr": const_lr, "no_ctc_loss": True},
          recog_opts={"load_ignore_missing_vars": True},
        )


def center_window_att_import_global_global_ctc_align_no_weight_feedback_rasr_recog(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        max_segment_len_list: Tuple[int, ...] = (-1,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/simple_ablations/rasr_recog/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          use_old_global_att_to_seg_att_maker=False
        )

        checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          train_opts={"num_epochs": n_epochs, "const_lr": const_lr},
        )

        test_segment_list = [
          "dev-other/1630-141772-0022/1630-141772-0022",
          "dev-other/4572-112381-0003/4572-112381-0003",
          "dev-other/3663-172005-0000/3663-172005-0000",
          "dev-other/700-122867-0033/700-122867-0033",
          "dev-other/4153-185072-0006/4153-185072-0006",
          "dev-other/116-288046-0011/116-288046-0011",
          "dev-other/1651-136854-0031/1651-136854-0031",
          "dev-other/1651-136854-0030/1651-136854-0030",
          "dev-other/7641-96252-0004/7641-96252-0004",
          "dev-other/4831-25894-0025/4831-25894-0025",
          "dev-other/6467-56885-0002/6467-56885-0002",
          "dev-other/6123-59186-0012/6123-59186-0012",
          "dev-other/1255-138279-0013/1255-138279-0013",
          "dev-other/1650-173552-0009/1650-173552-0009",
          "dev-other/1701-141760-0040/1701-141760-0040",
          "dev-other/3663-172528-0053/3663-172528-0053",
          "dev-other/3915-98647-0020/3915-98647-0020",
          "dev-other/3915-98647-0021/3915-98647-0021",
          "dev-other/1585-131718-0029/1585-131718-0029",
          "dev-other/116-288046-0010/116-288046-0010",
        ]

        for max_segment_len in max_segment_len_list:
          rasr_recog_center_window_att_import_global(
            alias=alias,
            config_builder=config_builder,
            checkpoint=checkpoints[n_epochs],
            recog_opts={
              "max_segment_len": max_segment_len,
              "search_corpus_key": "dev-other",
              "search_rqmt": {"mem": 4, "time": 1, "gpu": 0},
              "concurrent": 100,
              "open_vocab": True,
              # "segment_list": test_segment_list,
            },
          )

        for lm_scale in [0.1, 0.15, 0.2]:
          for lm_lookahead_scale in [0.3, 0.4, 0.5]:
            rasr_recog_center_window_att_import_global(
              alias=alias,
              config_builder=config_builder,
              checkpoint=checkpoints[n_epochs],
              recog_opts={
                "max_segment_len": -1,
                "search_corpus_key": "dev-other",
                "search_rqmt": {"mem": 8, "time": 2, "gpu": 0},
                "lm_opts": {
                  "type": "ARPA",
                  "file": config_builder.dependencies.arpa_lm_paths["arpa"],
                  "scale": lm_scale,
                  "image": config_builder.dependencies.rasr_format_paths.arpa_lm_image_path,
                },
                "lm_lookahead_opts": {
                  "scale": lm_lookahead_scale,
                  "cache_size_high": 3000,
                  "cache_size_low": 2000,
                  "history_limit": 1,
                },
                "open_vocab": False,
                "concurrent": 100,
              }, )

        for max_segment_len in (-1, 20):
          rasr_recog_center_window_att_import_global(
            alias=alias,
            config_builder=config_builder,
            checkpoint=checkpoints[n_epochs],
            recog_opts={
              "max_segment_len": max_segment_len,
              "search_corpus_key": "dev-other",
              "search_rqmt": {"mem": 4, "time": 1 if max_segment_len == -1 else 6, "gpu": 0},
              "lm_opts": None,
              "lm_lookahead_opts": None,
              "open_vocab": False,
              "concurrent": 100,
            }, )

          rasr_recog_center_window_att_import_global(
            alias=alias,
            config_builder=config_builder,
            checkpoint=checkpoints[n_epochs],
            recog_opts={
              "max_segment_len": max_segment_len,
              "search_corpus_key": "dev-other",
              "search_rqmt": {"mem": 8, "time": 4 if max_segment_len == -1 else 10, "gpu": 0},
              "lm_opts": None,
              "lm_lookahead_opts": None,
              "open_vocab": False,
              "concurrent": 100,
              "allow_recombination": True,
              "full_sum_decoding": True,
              "label_pruning": 8.0,
              "label_pruning_limit": 128,
              "word_end_pruning": 8.0,
              "word_end_pruning_limit": 128,
              "simple_beam_search": False,
            }, )

        # rasr_realign_center_window_att_import_global(
        #   alias=alias,
        #   config_builder=config_builder,
        #   checkpoint=checkpoints[n_epochs],
        #   realign_opts={
        #     "search_corpus_key": "dev-other",
        #     "search_rqmt": {"mem": 8, "time": 4, "gpu": 0},
        #     "concurrent": 100,
        #   },
        # )

        for lm_scale in [0.05, 0.2, 0.3]:
          for lm_lookahead_scale in [0.1, 0.2, 0.3, 0.4]:
            rasr_recog_center_window_att_import_global(
              alias=alias,
              config_builder=config_builder,
              checkpoint=checkpoints[n_epochs],
              recog_opts={
                "max_segment_len": -1,
                "search_corpus_key": "dev-other",
                "search_rqmt": {"mem": 12, "time": 6, "gpu": 0},
                "concurrent": 100,
                "lm_opts": {
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
                "lm_lookahead_opts": {
                  "scale": lm_lookahead_scale,
                  "cache_size_high": 3000,
                  "cache_size_low": 2000,
                  "history_limit": 1,
                  "type": "ARPA",
                  "file": config_builder.dependencies.arpa_lm_paths["arpa"],
                  "image": config_builder.dependencies.rasr_format_paths.arpa_lm_image_path,
                  "separate_lookahead_lm": True
                },
                "open_vocab": False,
              }, )
