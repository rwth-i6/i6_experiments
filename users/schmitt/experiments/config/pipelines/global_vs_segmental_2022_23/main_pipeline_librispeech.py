from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_LABELS
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.librispeech.train_recog.global_ import GlobalTrainRecogPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants_librispeech import models, global_model_variants
from i6_experiments.users.schmitt.alignment.alignment import extract_ctc_alignment

global_dependencies = {
  "bpe": LibrispeechBPE10025_LABELS
}


def run_pipeline():
  # for every model, store the checkpoint
  checkpoints = {}

  num_epochs = [618, 2035]

  model_dir_alias = "models/librispeech"

  """
  Train standard global attention models
  """
  for model_type, model_variants in global_model_variants.items():
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = global_dependencies[variant_params["label_type"]]

      base_alias = "%s/%s/%s" % (model_dir_alias, model_type, variant_name)

      global_train_recog_pipeline = GlobalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=num_epochs,
        base_alias=base_alias,
        do_recog=True
      )
      global_train_recog_pipeline.run()
      checkpoints[variant_name] = global_train_recog_pipeline.checkpoints
