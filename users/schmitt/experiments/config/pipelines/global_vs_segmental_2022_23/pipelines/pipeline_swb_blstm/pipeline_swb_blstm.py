from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_LABELS, LibrispeechBPE10025_CTC_ALIGNMENT

from sisyphus import tk

from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import SWBBlstmGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import SWBBlstmSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_swb_blstm import models

global_dependencies = {
  "bpe": LibrispeechBPE10025_LABELS
}

segmental_dependencies = {
  "bpe-ctc-align": LibrispeechBPE10025_CTC_ALIGNMENT
}


def run_pipeline():
  # for every model, store the checkpoint
  checkpoints = {}

  num_epochs = [618, 2035]

  model_dir_alias = "models/test_new_pipeline"

  """
  Train standard global attention models
  """
  for model_type, model_variants in [("swb_blstm_glob_att", models["swb_blstm_glob_att"])]:
    for variant_name, variant_params in model_variants.items():
      base_alias = "%s/%s/%s" % (model_dir_alias, model_type, variant_name)

      config_builder = SWBBlstmGlobalAttentionConfigBuilder(
        dependencies=variant_params["dependencies"],
        variant_params=variant_params,
      )

      train_job = ReturnnTrainingJob(
        config_builder.get_train_config(
          opts={"cleanup_old_models": {"keep_best_n": 0, "keep_last_n": 1}}),
        num_epochs=40,
        keep_epochs=[40],
        log_verbosity=5,
        returnn_python_exe=variant_params["returnn_python_exe"],
        returnn_root=variant_params["returnn_python_root"],
        mem_rqmt=24,
        time_rqmt=12)
      train_job.add_alias(base_alias + "/train")
      alias = train_job.get_one_alias()
      tk.register_output(alias + "/models", train_job.out_model_dir)
      tk.register_output(alias + "/plot_lr", train_job.out_plot_lr)

      # GlobalReturnnDecodingExperiment(
      #   dependencies=variant_params["dependencies"],
      #   returnn_config=config_builder.get_recog_config(search_corpus_key="dev"),
      #   variant_params=variant_params,
      #   checkpoint=Checkpoint(
      #     Path("")
      #   ),
      #   dump_best_traces=False,
      #   corpus_key="dev",
      #   base_alias=base_alias + "/returnn_label_sync_beam_search")


  for model_type, model_variants in [("swb_blstm_seg_att", models["swb_blstm_seg_att"])]:
    for variant_name, variant_params in model_variants.items():
      base_alias = "%s/%s/%s" % (model_dir_alias, model_type, variant_name)

      config_builder = SWBBlstmSegmentalAttentionConfigBuilder(
        dependencies=variant_params["dependencies"],
        variant_params=variant_params,
      )

      train_job = ReturnnTrainingJob(
        config_builder.get_train_config(
          opts={
            "cleanup_old_models": {"keep_best_n": 0, "keep_last_n": 1},
            "chunking": {
              "chunk_size_targets": 60,
              "chunk_size_data": 360,
              "chunk_step_targets": 30,
              "chunk_step_data": 180,
            }
          }),
        num_epochs=40,
        keep_epochs=[40],
        log_verbosity=5,
        returnn_python_exe=variant_params["returnn_python_exe"],
        returnn_root=variant_params["returnn_python_root"],
        mem_rqmt=24,
        time_rqmt=12)
      train_job.add_alias(base_alias + "/train")
      alias = train_job.get_one_alias()
      tk.register_output(alias + "/models", train_job.out_model_dir)
      tk.register_output(alias + "/plot_lr", train_job.out_plot_lr)



