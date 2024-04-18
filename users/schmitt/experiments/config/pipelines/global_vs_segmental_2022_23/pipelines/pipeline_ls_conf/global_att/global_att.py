import copy
from typing import Dict, List, Any, Optional, Tuple

from i6_core.returnn.training import Checkpoint

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import LibrispeechConformerGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import GlobalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog import ReturnnDecodingExperimentV2
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnGlobalAttDecodingExperiment, ReturnnGlobalAttDecodingPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  returnn_recog_center_window_att_import_global,
  get_center_window_att_config_builder,
  train_center_window_att_import_global
)

default_import_model_name = "glob.conformer.mohammad.5.6"


def glob_att_import_global_no_finetuning():
  alias = "models/ls_conformer/import_%s/glob_att/no_finetuning" % (default_import_model_name,)
  config_builder = get_global_att_config_builder(use_weight_feedback=True)

  recog_exp = ReturnnGlobalAttDecodingExperiment(
    alias=alias,
    config_builder=config_builder,
    checkpoint=external_checkpoints[default_import_model_name],
    checkpoint_alias="best-4-avg",
    recog_opts={"search_corpus_key": "dev-other"},
    analysis_opts={"att_weight_seq_tags": [
      "dev-other/3660-6517-0005/3660-6517-0005",
      "dev-other/6467-62797-0001/6467-62797-0001",
      "dev-other/6467-62797-0002/6467-62797-0002",
      "dev-other/7697-105815-0015/7697-105815-0015",
      "dev-other/7697-105815-0051/7697-105815-0051",
    ]}
  )
  recog_exp.run_eval()
  recog_exp.run_analysis()


def glob_att_import_global_diff_epochs_diff_lrs(
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float] = (1e-4,),
        analysis_checkpoint_alias: Optional[str] = None,
        att_weight_seq_tags: Optional[List[str]] = None
):
  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      alias = "models/ls_conformer/import_%s/glob_att/diff_epochs/%d-epochs_const-lr-%f" % (default_import_model_name, n_epochs, const_lr)
      config_builder = get_global_att_config_builder(use_weight_feedback=True)

      train_exp = GlobalTrainExperiment(
        config_builder=config_builder,
        alias=alias,
        num_epochs=n_epochs,
        train_opts={
          "import_model_train_epoch1": external_checkpoints[default_import_model_name],
          "lr_opts": {
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1/3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
          "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
          "max_seq_length": {"targets": 75}
        }
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      for checkpoint_alias in ("last", "best", "best-4-avg"):
        recog_exp = ReturnnGlobalAttDecodingExperiment(
          alias=alias,
          config_builder=config_builder,
          checkpoint={
            "model_dir": model_dir,
            "learning_rates": learning_rates,
            "key": "dev_score_output/output_prob",
            "checkpoints": checkpoints,
            "n_epochs": n_epochs
          },
          checkpoint_alias=checkpoint_alias,
          recog_opts={
            "search_corpus_key": "dev-other",
          },
          analysis_opts={"att_weight_seq_tags": att_weight_seq_tags}
        )
        recog_exp.run_eval()
        if checkpoint_alias == analysis_checkpoint_alias:
          recog_exp.run_analysis()


def glob_att_import_global_diff_epochs_diff_lrs_recog_wav_no_peak_norm_same_static_padding(
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float] = (1e-4,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        att_weight_seq_tags: Optional[List[str]] = None,
        run_analysis: bool = False
):
  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      alias = "models/ls_conformer/import_%s/glob_att/diff_epochs_recog_wav_no_peak_norm_same_static_padding/%d-epochs_const-lr-%f" % (default_import_model_name, n_epochs, const_lr)
      config_builder = get_global_att_config_builder(use_weight_feedback=True)

      train_exp = GlobalTrainExperiment(
        config_builder=config_builder,
        alias=alias,
        num_epochs=n_epochs,
        train_opts={
          "import_model_train_epoch1": external_checkpoints[default_import_model_name],
          "lr_opts": {
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1/3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
          "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
          "max_seq_length": {"targets": 75}
        }
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      for checkpoint_alias in checkpoint_aliases:
        recog_exp = ReturnnGlobalAttDecodingExperiment(
          alias=alias,
          config_builder=config_builder,
          checkpoint={
            "model_dir": model_dir,
            "learning_rates": learning_rates,
            "key": "dev_score_output/output_prob",
            "checkpoints": checkpoints,
            "n_epochs": n_epochs
          },
          checkpoint_alias=checkpoint_alias,
          recog_opts={
            "search_corpus_key": "dev-other",
            "dataset_opts": {
              "peak_normalization": False,
              "oggzip_paths": {
                "dev-other": [
                  config_builder.variant_params["dataset"]["corpus"].oggzip_paths_wav["dev-other"].out_ogg_zip]}
            },
            "use_same_static_padding": True
          },
          analysis_opts={"att_weight_seq_tags": att_weight_seq_tags},
        )
        recog_exp.run_eval()
        if run_analysis:
          recog_exp.run_analysis()


def glob_att_import_global_diff_epochs_diff_lrs_freeze_encoder(
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float] = (1e-4,),
        checkpoint_aliases: Tuple[str, ...] = ("last", "best", "best-4-avg"),
        att_weight_seq_tags: Optional[List[str]] = None,
        run_analysis: bool = False
):
  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      alias = "models/ls_conformer/import_%s/glob_att/diff_epochs_freeze_encoder/%d-epochs_const-lr-%f" % (default_import_model_name, n_epochs, const_lr)
      config_builder = get_global_att_config_builder(use_weight_feedback=True)

      train_exp = GlobalTrainExperiment(
        config_builder=config_builder,
        alias=alias,
        num_epochs=n_epochs,
        train_opts={
          "import_model_train_epoch1": external_checkpoints[default_import_model_name],
          "lr_opts": {
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1/3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
          "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
          "max_seq_length": {"targets": 75},
          "freeze_encoder": True
        }
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      for checkpoint_alias in checkpoint_aliases:
        recog_exp = ReturnnGlobalAttDecodingExperiment(
          alias=alias,
          config_builder=config_builder,
          checkpoint={
            "model_dir": model_dir,
            "learning_rates": learning_rates,
            "key": "dev_score_output/output_prob",
            "checkpoints": checkpoints,
            "n_epochs": n_epochs
          },
          checkpoint_alias=checkpoint_alias,
          recog_opts={
            "search_corpus_key": "dev-other",
          },
          analysis_opts={"att_weight_seq_tags": att_weight_seq_tags},
        )
        recog_exp.run_eval()
        if run_analysis:
          recog_exp.run_analysis()


def glob_att_import_global_bpe_lm(
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        lm_scale_list: Tuple[float, ...] = (0.4,),
        lm_type: str = "trafo",
        ilm_scale_list: Tuple[float, ...] = (0.0,),
        ilm_type: Optional[str] = None,
        lm_recog_checkpoint_alias: str = "last",
        beam_size_list: Tuple[int, ...] = (12,)
):
  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      alias = "models/ls_conformer/import_%s/glob_att/bpe_lm/%d-epochs_const-lr-%f" % (default_import_model_name, n_epochs, const_lr)
      config_builder = get_global_att_config_builder(use_weight_feedback=True)

      train_exp = GlobalTrainExperiment(
        config_builder=config_builder,
        alias=alias,
        num_epochs=n_epochs,
        train_opts={
          "import_model_train_epoch1": external_checkpoints[default_import_model_name],
          "lr_opts": {
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1/3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
          "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
          "max_seq_length": {"targets": 75}
        }
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      ilm_opts = {"type": ilm_type}
      if ilm_type == "mini_att":
        ilm_opts.update({
          "use_se_loss": False,
          "correct_eos": True,
        })

      ReturnnGlobalAttDecodingPipeline(
        alias=alias,
        config_builder=config_builder,
        checkpoint={
          "model_dir": model_dir,
          "learning_rates": learning_rates,
          "key": "dev_score_output/output_prob",
          "checkpoints": checkpoints,
          "n_epochs": n_epochs
        },
        checkpoint_aliases=(lm_recog_checkpoint_alias,),
        beam_sizes=beam_size_list,
        lm_scales=lm_scale_list,
        lm_opts={"type": lm_type},
        ilm_scales=ilm_scale_list,
        ilm_opts=ilm_opts
      ).run()


def glob_att_import_global_concat_recog(
        n_epochs_list: Tuple[int] = (10, 100),
        const_lr_list: Tuple[float] = (1e-4,),
        concat_nums: Tuple[int] = (2, 4),
):
  # TODO: i restarted the search jobs and something seems to have changed in RETURNN since the last time, because
  # it now throws an error:
  """
    File "/u/schmitt/src/returnn_new/returnn/datasets/audio.py", line 346, in OggZipDataset._get_ref_seq_idx
    line: return self._seq_order[seq_idx]
    locals:
      self = <local> <OggZipDataset 'dataset_id140515693976880_subdataset_zip_dataset' epoch=None>
      self._seq_order = <local> None
      seq_idx = <local> 0
  """

  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      alias = "models/ls_conformer/import_%s/glob_att/concat_recog/%d-epochs_const-lr-%f" % (default_import_model_name, n_epochs, const_lr)
      config_builder = get_global_att_config_builder(use_weight_feedback=True)

      train_exp = GlobalTrainExperiment(
        config_builder=config_builder,
        alias=alias,
        num_epochs=n_epochs,
        train_opts={
          "import_model_train_epoch1": external_checkpoints[default_import_model_name],
          "lr_opts": {
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1/3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
          "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
          "max_seq_length": {"targets": 75}
        }
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      for concat_num in concat_nums:
        if concat_num == 8:
          batch_size = 2518800
          search_rqmt = None
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
            align_targets = ctc_aligns.global_att_ctc_align.ctc_alignments_with_eos(
              segment_paths=center_window_config_builder.dependencies.segment_paths,
              blank_idx=center_window_config_builder.dependencies.model_hyperparameters.blank_idx,
              eos_idx=center_window_config_builder.dependencies.model_hyperparameters.sos_idx
            )
          else:
            align_targets = ctc_aligns.global_att_ctc_align.ctc_alignments

          center_window_checkpoints, _, _ = train_center_window_att_import_global(
            alias=center_window_att_alias,
            config_builder=center_window_config_builder,
            train_opts={"num_epochs": n_epochs, "const_lr": const_lr, "align_targets": align_targets},
          )

          returnn_recog_center_window_att_import_global(
            alias=center_window_att_alias,
            config_builder=center_window_config_builder,
            checkpoint=center_window_checkpoints[n_epochs],
            recog_opts={"analyse": True, "search_corpus_key": "dev-other"},
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


def recog_global_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerGlobalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        search_corpus_key: str,
        concat_num: Optional[int] = None,
        search_rqmt: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        lm_opts: Optional[Dict] = None,
        ilm_correction_opts: Optional[Dict] = None,
        analyse: bool = False,
        att_weight_seq_tags: Optional[List[str]] = None,
        checkpoint_alias: str = "last"
):
  recog_exp = ReturnnDecodingExperimentV2(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    corpus_key=search_corpus_key,
    concat_num=concat_num,
    search_rqmt=search_rqmt,
    batch_size=batch_size,
    lm_opts=lm_opts,
    ilm_correction_opts=ilm_correction_opts,
    checkpoint_alias=checkpoint_alias,
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
