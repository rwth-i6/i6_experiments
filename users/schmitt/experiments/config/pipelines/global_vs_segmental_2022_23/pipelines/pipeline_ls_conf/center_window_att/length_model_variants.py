from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import default_import_model_name, get_center_window_att_config_builder, standard_train_recog_center_window_att_import_global, recog_center_window_att_import_global


def center_window_att_import_global_global_ctc_align_length_model_no_label_feedback(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_length_model_no_label_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_embedding": False, "layer_class": "lstm"}
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_diff_emb_size(
        win_size_list: Tuple[int, ...] = (129,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        emb_size_list: Tuple[int, ...] = (64,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for emb_size in emb_size_list:
          alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_length_model_diff_emb_size/win-size-%d_%d-epochs_%f-const-lr/emb-size-%d" % (
            default_import_model_name, win_size, n_epochs, const_lr, emb_size
          )
          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            length_model_opts={"use_embedding": True, "embedding_size": emb_size, "layer_class": "lstm"}
          )

          standard_train_recog_center_window_att_import_global(
            config_builder=config_builder,
            alias=alias,
            n_epochs=n_epochs,
            const_lr=const_lr
          )


def center_window_att_import_global_global_ctc_align_length_model_only_non_blank_ctx(
        win_size_list: Tuple[int, ...] = (129,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_length_model_only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_embedding": True, "embedding_size": 128, "use_alignment_ctx": False, "layer_class": "lstm"},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_length_model_linear_layer(
        win_size_list: Tuple[int, ...] = (129,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_length_model_linear_layer/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_embedding": True, "embedding_size": 128, "use_alignment_ctx": True, "layer_class": "linear"},
          use_old_global_att_to_seg_att_maker=False
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )
