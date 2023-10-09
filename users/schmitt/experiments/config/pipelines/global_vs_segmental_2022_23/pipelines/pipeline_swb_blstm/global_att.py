import copy
from typing import Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_LABELS, LibrispeechBPE10025_CTC_ALIGNMENT

from sisyphus import tk

from i6_core.returnn.training import ReturnnTrainingJob, Checkpoint
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import SWBBlstmGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_swb_blstm import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import GlobalTrainExperiment, SegmentalTrainExperiment


def glob_att():
  n_epochs = 150

  alias = "models/swb_blstm/no_import/glob_att_best"
  config_builder = get_global_att_config_builder()

  train_exp = GlobalTrainExperiment(
    config_builder=config_builder,
    alias=alias,
    n_epochs=n_epochs,
  )
  checkpoints, model_dir, learning_rates = train_exp.run_train()

  # recog_global_att_import_global(
  #   alias=alias,
  #   config_builder=config_builder,
  #   checkpoint=external_checkpoints[default_import_model_name],
  #   analyse=True,
  #   search_corpus_key="dev-other",
  #   att_weight_seq_tags=[
  #     "dev-other/3660-6517-0005/3660-6517-0005",
  #     "dev-other/6467-62797-0001/6467-62797-0001",
  #     "dev-other/6467-62797-0002/6467-62797-0002",
  #     "dev-other/7697-105815-0015/7697-105815-0015",
  #     "dev-other/7697-105815-0051/7697-105815-0051",
  #   ],
  # )


def get_global_att_config_builder():
  model_type = "swb_blstm_glob_att"
  variant_name = "glob.blstm.best"
  variant_params = copy.deepcopy(models[model_type][variant_name])

  config_builder = SWBBlstmGlobalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  return config_builder


# def recog_global_att_import_global(
#         alias: str,
#         config_builder: SWBBlstmGlobalAttentionConfigBuilder,
#         checkpoint: Checkpoint,
#         search_corpus_key: str,
#         concat_num: Optional[int] = None,
#         search_rqmt: Optional[Dict[str, Any]] = None,
#         batch_size: Optional[int] = None,
#         analyse: bool = False,
#         att_weight_seq_tags: Optional[List[str]] = None,
# ):
#   recog_exp = ReturnnDecodingExperimentV2(
#     alias=alias,
#     config_builder=config_builder,
#     checkpoint=checkpoint,
#     corpus_key=search_corpus_key,
#     concat_num=concat_num,
#     search_rqmt=search_rqmt,
#     batch_size=batch_size
#   )
#   recog_exp.run_eval()
#
#   if analyse:
#     if concat_num is not None:
#       raise NotImplementedError
#     analaysis_corpus_key = "cv"
#     forward_recog_opts = {"search_corpus_key": analaysis_corpus_key}
#
#     recog_exp.run_analysis(
#       ground_truth_hdf=None,
#       att_weight_ref_alignment_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[search_corpus_key],
#       att_weight_ref_alignment_blank_idx=10025,
#       att_weight_seq_tags=att_weight_seq_tags,
#     )


# def run_pipeline():
#   # for every model, store the checkpoint
#   checkpoints = {}
#
#   num_epochs = [618, 2035]
#
#   model_dir_alias = "models/test_new_pipeline"
#
#   """
#   Train standard global attention models
#   """
#   for model_type, model_variants in [("swb_blstm_glob_att", models["swb_blstm_glob_att"])]:
#     for variant_name, variant_params in model_variants.items():
#       base_alias = "%s/%s/%s" % (model_dir_alias, model_type, variant_name)
#
#       config_builder = SWBBlstmGlobalAttentionConfigBuilder(
#         dependencies=variant_params["dependencies"],
#         variant_params=variant_params,
#       )
#
#       train_job = ReturnnTrainingJob(
#         config_builder.get_train_config(
#           opts={"cleanup_old_models": {"keep_best_n": 0, "keep_last_n": 1}}),
#         num_epochs=40,
#         keep_epochs=[40],
#         log_verbosity=5,
#         returnn_python_exe=variant_params["returnn_python_exe"],
#         returnn_root=variant_params["returnn_python_root"],
#         mem_rqmt=24,
#         time_rqmt=12)
#       train_job.add_alias(base_alias + "/train")
#       alias = train_job.get_one_alias()
#       tk.register_output(alias + "/models", train_job.out_model_dir)
#       tk.register_output(alias + "/plot_lr", train_job.out_plot_lr)
