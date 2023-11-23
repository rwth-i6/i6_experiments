import copy
from typing import Dict, List, Any, Optional, Tuple

from i6_core.returnn.training import Checkpoint

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_ROOT, RETURNN_EXE
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import LibrispeechConformerGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import GlobalTrainExperiment, SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog import ReturnnDecodingExperimentV2
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import recog_center_window_att_import_global, get_center_window_att_config_builder
from i6_experiments.users.schmitt.alignment.alignment import AlignmentAddEosJob


default_import_model_name = "glob.conformer.mohammad.5.6"


def glob_att_import_global_no_finetuning():
  alias = "models/ls_conformer/import_%s/glob_att_diff_epochs_no_finetuning" % (default_import_model_name,)
  config_builder = get_global_att_config_builder(use_weight_feedback=True)

  recog_global_att_import_global(
    alias=alias,
    config_builder=config_builder,
    checkpoint=external_checkpoints[default_import_model_name],
    analyse=True,
    search_corpus_key="dev-other",
    att_weight_seq_tags=[
      "dev-other/3660-6517-0005/3660-6517-0005",
      "dev-other/6467-62797-0001/6467-62797-0001",
      "dev-other/6467-62797-0002/6467-62797-0002",
      "dev-other/7697-105815-0015/7697-105815-0015",
      "dev-other/7697-105815-0051/7697-105815-0051",
    ],
  )


def glob_att_import_global_diff_epochs_diff_lrs(
        n_epochs_list: Tuple[int] = (10, 100),
        const_lr_list: Tuple[float] = (1e-4,),
):
  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      alias = "models/ls_conformer/import_%s/glob_att_diff_epochs/%d-epochs_const-lr-%f" % (default_import_model_name, n_epochs, const_lr)
      config_builder = get_global_att_config_builder(use_weight_feedback=True)

      train_exp = GlobalTrainExperiment(
        config_builder=config_builder,
        alias=alias,
        n_epochs=n_epochs,
        import_model_train_epoch1=external_checkpoints[default_import_model_name],
        lr_opts={
          "type": "const_then_linear",
          "const_lr": const_lr,
          "const_frac": 1/3,
          "final_lr": 1e-6,
          "num_epochs": n_epochs
        }
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      recog_global_att_import_global(
        alias=alias,
        config_builder=config_builder,
        checkpoint=checkpoints[n_epochs],
        analyse=True,
        search_corpus_key="dev-other",
        att_weight_seq_tags=[
          "dev-other/3660-6517-0005/3660-6517-0005",
          "dev-other/6467-62797-0001/6467-62797-0001",
          "dev-other/6467-62797-0002/6467-62797-0002",
          "dev-other/7697-105815-0015/7697-105815-0015",
          "dev-other/7697-105815-0051/7697-105815-0051",
          "dev-other/6467-94831-0006/6467-94831-0006",  # global 2 err, win-size-8 + win-size-128 + seg correct
          "dev-other/8254-84205-0021/8254-84205-0021",  # seg + win-size-8 + win-size-128 2 err, global correct
          "dev-other/6123-59150-0002/6123-59150-0002",  # seg + win-size-8 2 err, win-size-128 1 err, global correct
          "dev-other/1585-131718-0027/1585-131718-0027",  # global 2 err, win-size-8 + win-size-128 + seg correct
          "dev-other/1585-157660-0007/1585-157660-0007",  # seg 5 err, win-size-8 + win-size-128 4 err, global correct
          "dev-other/6123-59150-0008/6123-59150-0008",  # global 2 err, win-size-8 + win-size-128 1 err, seg correct
          "dev-other/1650-167613-0026/1650-167613-0026",  # seg 2 err, win-size-8 + win-size-128 3 err, global correct
          "dev-other/1686-142278-0018/1686-142278-0018",  # global 2 err, win-size-8 + win-size-128 + seg correct
          "dev-other/1701-141759-0026/1701-141759-0026",  # seg + win-size-8 + win-size-128 2 err, global correct
          "dev-other/2506-11278-0017/2506-11278-0017",  # all correct
          "dev-other/2506-11278-0025/2506-11278-0025",  # all correct
          "dev-other/2506-13150-0004/2506-13150-0004",  # all correct
          "dev-other/3660-172182-0035/3660-172182-0035",  # seg + win-size-8 + win-size-128 2 err, global correct
          "dev-other/4153-186222-0014/4153-186222-0014",  # global 3 err, win-size-8 + win-size-128 1 err, seg correct
          "dev-other/4570-14911-0000/4570-14911-0000",  # global 2 err, win-size-8 + win-size-128 1 err, seg correct
          "dev-other/5849-50873-0033/5849-50873-0033",  # seg 2 err, global + win-size-8 + win-size-128 correct
          "dev-other/6123-59186-0009/6123-59186-0009",  # seg 1 err, win-size-8 1 err, global + win-size-128 correct
          "dev-other/6267-65525-0049/6267-65525-0049",  # global 2 err, win-size-8 + win-size-128 + seg correct
          "dev-other/8288-274162-0025/8288-274162-0025",  # global 3 err, win-size-8 + win-size-128 + seg correct
        ],
      )


def glob_att_import_global_concat_recog(
        n_epochs_list: Tuple[int] = (10, 100),
        const_lr_list: Tuple[float] = (1e-4,),
        concat_nums: Tuple[int] = (2, 4),
):
  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      alias = "models/ls_conformer/import_%s/glob_att_concat_recog/%d-epochs_const-lr-%f" % (default_import_model_name, n_epochs, const_lr)
      config_builder = get_global_att_config_builder(use_weight_feedback=True)

      train_exp = GlobalTrainExperiment(
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
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      for concat_num in concat_nums:
        if concat_num == 8:
          batch_size = 2518800
          search_rqmt = None
          # search_rqmt = {
          #   "gpu": 1,
          #   "cpu": 2,
          #   "mem": 4,
          #   "time": 1,
          #   "sbatch_args": ["-p", "gpu_24gb", "-w", "cn-501"]
          # }
        else:
          batch_size = None
          search_rqmt = None

        recog_global_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=checkpoints[n_epochs],
          analyse=False,
          search_corpus_key="dev-other",
          concat_num=concat_num,
          batch_size=batch_size,
          search_rqmt=search_rqmt,
        )


def center_window_att_import_global_do_label_sync_search(
        win_size_list: Tuple[int, ...] = (4, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        weight_feedback_list: Tuple[bool, ...] = (True, False),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        center_window_use_eos: bool = False,
):
  for win_size in win_size_list:
    for weight_feedback in weight_feedback_list:
      for n_epochs in n_epochs_list:
        for const_lr in const_lr_list:
          alias = "models/ls_conformer/import_%s/center_window_train_global_recog/win-size-%d/%s/%s/%d-epochs_const-lr-%f" % (
            default_import_model_name,
            win_size,
            "w-weight-feedback" if weight_feedback else "wo-weight-feedback",
            "w-eos" if center_window_use_eos else "wo-eos",
            n_epochs,
            const_lr
          )
          center_window_att_alias = alias + "/center_window_att_train_recog"
          global_att_alias = alias + "/global_att_recog"

          center_window_config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=weight_feedback,
            search_remove_eos=center_window_use_eos,
          )

          if center_window_use_eos:
            align_targets = {
              corpus_key: AlignmentAddEosJob(
                hdf_align_path=alignment_path,
                segment_file=center_window_config_builder.dependencies.segment_paths.get(corpus_key, None),
                blank_idx=center_window_config_builder.dependencies.model_hyperparameters.blank_idx,
                eos_idx=center_window_config_builder.dependencies.model_hyperparameters.sos_idx,
                returnn_python_exe=RETURNN_EXE,
                returnn_root=RETURNN_ROOT,
              ).out_align for corpus_key, alignment_path in ctc_aligns.global_att_ctc_align.ctc_alignments.items()
            }
          else:
            align_targets = ctc_aligns.global_att_ctc_align.ctc_alignments

          center_window_train_exp = SegmentalTrainExperiment(
            config_builder=center_window_config_builder,
            alias=center_window_att_alias,
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
            alias=center_window_att_alias,
            config_builder=center_window_config_builder,
            checkpoint=center_window_checkpoints[n_epochs],
            analyse=True,
            search_corpus_key="dev-other"
          )

          global_config_builder = get_global_att_config_builder(use_weight_feedback=weight_feedback)

          recog_global_att_import_global(
            alias=global_att_alias,
            checkpoint=center_window_checkpoints[n_epochs],
            config_builder=global_config_builder,
            search_corpus_key="dev-other",
            analyse=True
          )


def get_global_att_config_builder(use_weight_feedback: bool = True):
  model_type = "librispeech_conformer_glob_att"
  variant_name = "glob.conformer.mohammad.5.6"
  variant_params = copy.deepcopy(models[model_type][variant_name])
  variant_params["network"]["use_weight_feedback"] = use_weight_feedback

  config_builder = LibrispeechConformerGlobalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  return config_builder


# def train_global_att_import_global(
#         alias: str,
#         config_builder: LibrispeechConformerGlobalAttentionConfigBuilder,
#         n_epochs: int,
#         import_model_name: str,
#         const_lr: float = 1e-4,
#         const_frac: float = 1/3,
#         final_lr: float = 1e-6,
#         cleanup_old_models: Optional[Dict] = None,
# ):
#   cleanup_old_models = cleanup_old_models if cleanup_old_models is not None else {"keep_best_n": 1, "keep_last_n": 1}
#
#   train_opts = {
#     "cleanup_old_models": cleanup_old_models,
#     "lr_opts": {
#       "type": "const_then_linear",
#       "const_lr": const_lr,
#       "const_frac": const_frac,
#       "final_lr": final_lr,
#       "num_epochs": n_epochs
#     },
#     "import_model_train_epoch1": external_checkpoints[import_model_name],
#     "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
#     "max_seq_length": {"targets": 75}
#   }
#
#   checkpoints, model_dir, learning_rates = run_train(
#     config_builder=config_builder,
#     variant_params=config_builder.variant_params,
#     n_epochs=n_epochs,
#     train_opts=train_opts,
#     alias=alias
#   )
#
#   return checkpoints, model_dir, learning_rates


def recog_global_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerGlobalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        search_corpus_key: str,
        concat_num: Optional[int] = None,
        search_rqmt: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        analyse: bool = False,
        att_weight_seq_tags: Optional[List[str]] = None,
):
  recog_exp = ReturnnDecodingExperimentV2(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    corpus_key=search_corpus_key,
    concat_num=concat_num,
    search_rqmt=search_rqmt,
    batch_size=batch_size
  )
  recog_exp.run_eval()

  if analyse:
    if concat_num is not None:
      raise NotImplementedError
    analaysis_corpus_key = "cv"
    forward_recog_opts = {"search_corpus_key": analaysis_corpus_key}

    recog_exp.run_analysis(
      ground_truth_hdf=None,
      att_weight_ref_alignment_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[search_corpus_key],
      att_weight_ref_alignment_blank_idx=10025,
      att_weight_seq_tags=att_weight_seq_tags,
    )

    # run_analysis(
    #   config_builder=config_builder,
    #   variant_params=config_builder.variant_params,
    #   ground_truth_hdf=None,
    #   att_weight_ref_alignment_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[analaysis_corpus_key],
    #   corpus_key=analaysis_corpus_key,
    #   forward_recog_opts=forward_recog_opts,
    #   checkpoint=checkpoint,
    #   alias=alias,
    #   att_weight_ref_alignment_blank_idx=10025,  # TODO: change to non-hardcoded index
    # )


# def train_recog_glob_att_import_global(
#         alias: str,
#         n_epochs: int,
#         import_model_name: str,
#         cleanup_old_models: Optional[Dict] = None,
#         const_lr: float = 1e-4,
#         const_frac: float = 1/3,
#         final_lr: float = 1e-6,
#         analyse=True,
# ):
#   config_builder = get_global_att_config_builder()
#
#   train_exp = GlobalTrainExperiment(
#     config_builder=config_builder,
#     alias=alias,
#     n_epochs=n_epochs,
#     import_model_name=import_model_name,
#     const_lr=const_lr,
#     const_frac=const_frac,
#     final_lr=final_lr,
#     cleanup_old_models=cleanup_old_models,
#   )
#
#   checkpoints, model_dir, learning_rates = train_exp.run_train()
#
#   recog_global_att_import_global(
#     alias=alias,
#     config_builder=config_builder,
#     checkpoint=checkpoints[n_epochs],
#     analyse=analyse,
#     search_corpus_key="dev-other"
#   )
#
#   return checkpoints
