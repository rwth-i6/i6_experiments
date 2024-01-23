import copy

from i6_core.returnn.training import ReturnnTrainingJob, Checkpoint
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import SWBBlstmSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_swb_blstm import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment


def seg_att():
  n_epochs = 150

  alias = "models/swb_blstm/no_import/seg_att_best"
  config_builder = get_seg_att_config_builder()

  train_exp = SegmentalTrainExperiment(
    config_builder=config_builder,
    alias=alias,
    n_epochs=n_epochs,
    align_targets=config_builder.dependencies.hdf_targets
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


def get_seg_att_config_builder():
  model_type = "swb_blstm_seg_att"
  variant_name = "seg.blstm.best"
  variant_params = copy.deepcopy(models[model_type][variant_name])

  config_builder = SWBBlstmSegmentalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  return config_builder


# def run_pipeline():
#   # for every model, store the checkpoint
#   checkpoints = {}
#
#   num_epochs = [618, 2035]
#
#   model_dir_alias = "models/test_new_pipeline"
#
#   for model_type, model_variants in [("swb_blstm_seg_att", models["swb_blstm_seg_att"])]:
#     for variant_name, variant_params in model_variants.items():
#       base_alias = "%s/%s/%s" % (model_dir_alias, model_type, variant_name)
#
#       config_builder = SWBBlstmSegmentalAttentionConfigBuilder(
#         dependencies=variant_params["dependencies"],
#         variant_params=variant_params,
#       )
#
#       train_job = ReturnnTrainingJob(
#         config_builder.get_train_config(
#           opts={
#             "cleanup_old_models": {"keep_best_n": 0, "keep_last_n": 1},
#             "chunking": {
#               "chunk_size_targets": 60,
#               "chunk_size_data": 360,
#               "chunk_step_targets": 30,
#               "chunk_step_data": 180,
#             }
#           }),
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



