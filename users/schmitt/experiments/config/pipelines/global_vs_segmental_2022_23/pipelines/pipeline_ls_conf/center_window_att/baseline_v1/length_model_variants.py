from typing import Tuple, Optional


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingExperiment, ReturnnSegmentalAttDecodingPipeline


def center_window_att_import_global_global_ctc_align_length_model_no_label_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/no_label_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_embedding": False}
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_diff_emb_size(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        emb_size_list: Tuple[int, ...] = (64,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for emb_size in emb_size_list:
          alias = f"{base_alias}/length_model_variants/diff_emb_size/win-size-%d_%d-epochs_%f-const-lr/emb-size-%d" % (
            win_size, n_epochs, const_lr, emb_size
          )
          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            length_model_opts={"embedding_size": emb_size},
            use_old_global_att_to_seg_att_maker=False
          )

          train_exp = SegmentalTrainExperiment(
            config_builder=config_builder,
            alias=alias,
            num_epochs=n_epochs,
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


def center_window_att_import_global_global_ctc_align_length_model_only_non_blank_ctx(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        emb_size_list: Tuple[int, ...] = (128,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for emb_size in emb_size_list:
          alias = f"{base_alias}/length_model_variants/only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr/emb-size-%d" % (
            win_size, n_epochs, const_lr, emb_size
          )
          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            length_model_opts={"embedding_size": emb_size, "use_alignment_ctx": False},
            use_old_global_att_to_seg_att_maker=False
          )

          train_exp = SegmentalTrainExperiment(
            config_builder=config_builder,
            alias=alias,
            num_epochs=n_epochs,
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


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/use_label_model_state/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_no_label_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/use_label_model_state_no_label_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_embedding": False},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_no_label_feedback_no_encoder_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/use_label_model_state_no_label_feedback_no_encoder_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_embedding": False, "use_current_frame": False},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_no_encoder_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/use_label_model_state_no_encoder_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_current_frame": False},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_only_non_blank_ctx(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        lm_scale_list: Tuple[float, ...] = (0.0,),
        lm_type: Optional[str] = None,
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        beam_size_list: Tuple[int, ...] = (12,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        run_analysis: bool = False,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/use_label_model_state_only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        ilm_opts = {"type": ilm_type}
        if ilm_type == "mini_att":
          ilm_opts.update({
            "use_se_loss": False,
            "correct_eos": False,
          })
        ReturnnSegmentalAttDecodingPipeline(
          alias=alias,
          config_builder=config_builder,
          checkpoint={
            "model_dir": model_dir,
            "learning_rates": learning_rates,
            "key": "dev_score_label_model/output_prob",
            "checkpoints": checkpoints,
            "n_epochs": n_epochs
          },
          checkpoint_aliases=checkpoint_aliases,
          beam_sizes=beam_size_list,
          lm_scales=lm_scale_list,
          lm_opts={"type": lm_type, "add_lm_eos_last_frame": True},
          ilm_scales=ilm_scale_list,
          ilm_opts=ilm_opts,
          run_analysis=run_analysis
        ).run()


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_only_non_blank_ctx_w_eos(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/use_label_model_state_only_non_blank_ctx_w_eos/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False,
          search_remove_eos=True,
        )

        align_targets = ctc_aligns.global_att_ctc_align.ctc_alignments_with_eos(
          segment_paths=config_builder.dependencies.segment_paths,
          blank_idx=config_builder.dependencies.model_hyperparameters.blank_idx,
          eos_idx=config_builder.dependencies.model_hyperparameters.sos_idx
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
          train_opts={"dataset_opts": {"hdf_targets": align_targets}}
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_linear_layer_use_label_model_state(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/linear_layer_use_label_model_state/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "layer_class": "linear"},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_linear_layer(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/linear_layer/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"layer_class": "linear"},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_linear_layer_no_label_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/linear_layer_no_label_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"layer_class": "linear", "use_embedding": False},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_linear_layer_only_non_blank_ctx(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/linear_layer_only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"layer_class": "linear", "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_length_model_explicit_lstm(
        win_size_list: Tuple[int, ...] = (5,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/length_model_variants/explicit_lstm/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"layer_class": "lstm_explicit"},
          use_old_global_att_to_seg_att_maker=False,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
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
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()
