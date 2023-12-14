from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import default_import_model_name, get_center_window_att_config_builder, standard_train_recog_center_window_att_import_global, recog_center_window_att_import_global

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints


def center_window_att_import_global_global_ctc_align_no_finetuning(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/random_init_length_model/win-size-%d" % (
      default_import_model_name, win_size
    )
    config_builder = get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
    )

    recog_center_window_att_import_global(
      alias=alias,
      config_builder=config_builder,
      checkpoint=external_checkpoints[default_import_model_name],
      analyse=False,
      search_corpus_key="dev-other",
      load_ignore_missing_vars=True,  # otherwise RETURNN will complain about missing length model params
    )


def center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/no_length_model/win-size-%d" % (
      default_import_model_name, win_size
    )
    config_builder = get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
      length_scale=0.0,
    )

    recog_center_window_att_import_global(
      alias=alias,
      config_builder=config_builder,
      checkpoint=external_checkpoints[default_import_model_name],
      analyse=False,
      search_corpus_key="dev-other",
      load_ignore_missing_vars=True,  # otherwise RETURNN will complain about missing length model params
    )


def center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model_blank_penalty(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    for blank_penalty in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4):
      alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/no_length_model_blank_penalty/win-size-%d_penalty-%f" % (
        default_import_model_name, win_size, blank_penalty
      )
      config_builder = get_center_window_att_config_builder(
        win_size=win_size,
        use_weight_feedback=True,
        length_scale=0.0,
        blank_penalty=blank_penalty
      )

      recog_center_window_att_import_global(
        alias=alias,
        config_builder=config_builder,
        checkpoint=external_checkpoints[default_import_model_name],
        analyse=False,
        search_corpus_key="dev-other",
        load_ignore_missing_vars=True,  # otherwise RETURNN will complain about missing length model params
      )
