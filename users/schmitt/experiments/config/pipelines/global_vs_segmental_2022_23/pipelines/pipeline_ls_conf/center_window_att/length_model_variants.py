from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import default_import_model_name, get_center_window_att_config_builder, standard_train_recog_center_window_att_import_global, recog_center_window_att_import_global
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_ROOT, RETURNN_EXE

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog import ReturnnDecodingExperimentV2

from i6_experiments.users.schmitt.alignment.alignment import AlignmentAddEosJob

def center_window_att_import_global_global_ctc_align_length_model_no_label_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/no_label_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_embedding": False}
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_diff_emb_size(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        emb_size_list: Tuple[int, ...] = (64,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for emb_size in emb_size_list:
          alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/diff_emb_size/win-size-%d_%d-epochs_%f-const-lr/emb-size-%d" % (
            default_import_model_name, win_size, n_epochs, const_lr, emb_size
          )
          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            length_model_opts={"embedding_size": emb_size}
          )

          standard_train_recog_center_window_att_import_global(
            config_builder=config_builder,
            alias=alias,
            n_epochs=n_epochs,
            const_lr=const_lr
          )


def center_window_att_import_global_global_ctc_align_length_model_only_non_blank_ctx(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        emb_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for emb_size in emb_size_list:
          alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr/emb-size-%d" % (
            default_import_model_name, win_size, n_epochs, const_lr, emb_size
          )
          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            length_model_opts={"embedding_size": emb_size, "use_alignment_ctx": False},
            use_old_global_att_to_seg_att_maker=False
          )

          standard_train_recog_center_window_att_import_global(
            config_builder=config_builder,
            alias=alias,
            n_epochs=n_epochs,
            const_lr=const_lr
          )


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/use_label_model_state/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_no_label_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/use_label_model_state_no_label_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_embedding": False},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_no_label_feedback_no_encoder_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/use_label_model_state_no_label_feedback_no_encoder_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_embedding": False, "use_current_frame": False},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_no_encoder_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/use_label_model_state_no_encoder_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_current_frame": False},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_only_non_blank_ctx(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/use_label_model_state_only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_only_non_blank_ctx_w_eos(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/use_label_model_state_only_non_blank_ctx_w_eos/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False,
          search_remove_eos=True,
        )

        align_targets = {
          corpus_key: AlignmentAddEosJob(
            hdf_align_path=alignment_path,
            segment_file=config_builder.dependencies.segment_paths.get(corpus_key, None),
            blank_idx=config_builder.dependencies.model_hyperparameters.blank_idx,
            eos_idx=config_builder.dependencies.model_hyperparameters.sos_idx,
            returnn_python_exe=RETURNN_EXE,
            returnn_root=RETURNN_ROOT,
          ).out_align for corpus_key, alignment_path in ctc_aligns.global_att_ctc_align.ctc_alignments.items()
        }

        center_window_train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          import_model_train_epoch1=external_checkpoints[default_import_model_name],
          lr_opts={
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
          align_targets=align_targets,
        )
        center_window_checkpoints, _, _ = center_window_train_exp.run_train()

        recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=center_window_checkpoints[n_epochs],
          analyse=True,
          search_corpus_key="dev-other"
        )


# def center_window_att_import_global_global_ctc_align_length_model_use_label_model_state_only_non_blank_ctx_worse_than_global(
#         win_size_list: Tuple[int, ...] = (5, 129),
#         n_epochs_list: Tuple[int, ...] = (10,),
#         const_lr_list: Tuple[float, ...] = (1e-4,),
# ):
#   for win_size in win_size_list:
#     for n_epochs in n_epochs_list:
#       for const_lr in const_lr_list:
#         alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/use_label_model_state_only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr" % (
#           default_import_model_name, win_size, n_epochs, const_lr
#         )
#         config_builder = get_center_window_att_config_builder(
#           win_size=win_size,
#           use_weight_feedback=True,
#           length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
#           use_old_global_att_to_seg_att_maker=False
#         )
#
#         standard_train_recog_center_window_att_import_global(
#           config_builder=config_builder,
#           alias=alias,
#           n_epochs=n_epochs,
#           const_lr=const_lr,
#           att_weight_seq_tags=[
#             "dev-other/116-288046-0010/116-288046-0010",
#             "dev-other/116-288047-0012/116-288047-0012",
#             "dev-other/1255-138279-0022/1255-138279-0022",
#             "dev-other/1255-74899-0017/1255-74899-0017",
#             "dev-other/1585-131718-0001/1585-131718-0001",
#             "dev-other/1585-131718-0020/1585-131718-0020",
#             "dev-other/1650-167613-0054/1650-167613-0054",
#             "dev-other/1686-142278-0010/1686-142278-0010",
#           ]
#         )


def center_window_att_import_global_global_ctc_align_length_model_linear_layer_use_label_model_state(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/linear_layer_use_label_model_state/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "layer_class": "linear"},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_linear_layer(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/linear_layer/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"layer_class": "linear"},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_linear_layer_no_label_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/linear_layer_no_label_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"layer_class": "linear", "use_embedding": False},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_linear_layer_only_non_blank_ctx(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/length_model_variants/linear_layer_only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"layer_class": "linear", "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )
