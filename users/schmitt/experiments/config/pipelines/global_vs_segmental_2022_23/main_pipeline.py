# dependencies
import copy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import SegmentalLabelDefinition, GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.label_singletons import RNA_BPE, RNA_BPE_SIL_BASE, RNA_BPE_SIL_TIME_RED_6, RNA_BPE_SPLIT_SIL_TIME_RED_6, BPE_LABELS

# experiments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.training import SegmentalTrainExperiment, GlobalTrainExperiment

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.recognition.segmental import run_returnn_simple_segmental_decoding, run_rasr_segmental_decoding
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.realignment import run_rasr_segmental_realignment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.recognition.global_ import run_returnn_label_sync_decoding
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.train_recog.segmental import SegmentalTrainRecogPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.train_recog.global_ import GlobalTrainRecogPipeline

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants import build_alias, models, global_model_variants, segmental_model_variants, segmental_model_variants_w_length_model, segmental_model_variants_wo_length_model

import time

segmental_dependencies = {
    "rna-bpe": RNA_BPE,
    "rna-bpe-sil-time-red-6": RNA_BPE_SIL_TIME_RED_6,
    "rna-bpe-split-sil-time-red-6": RNA_BPE_SPLIT_SIL_TIME_RED_6,
  }

global_dependencies = {
    "bpe": BPE_LABELS
  }


def check_names():
  # for each model, check if name is according to my naming conventions
  for _, model_variants in models.items():
    for variant_name, variant_params in model_variants.items():
      check_name = "" + build_alias(**variant_params["config"])
      assert variant_name == check_name, "\n{} \t should be \n{}".format(variant_name, check_name)


def run_pipeline():
  # for each model, check if name is according to my naming conventions
  check_names()

  ######################################## Segmental Model Training ########################################

  # for every model, store the checkpoint
  checkpoints = {}
  # for some models, store a realignment
  realignments = {}

  num_epochs = [40, 80, 120, 150]

  for model_type, model_variants in segmental_model_variants_w_length_model.items():
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=num_epochs,
        base_alias=base_alias,
        recog_type="standard",
        do_recog=True,
        rasr_recog_epochs=(150,),
        returnn_recog_epochs=(40, 150),
        num_retrain=2,
        realignment_length_scale=0.
      ).run()

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=num_epochs,
        base_alias=base_alias,
        recog_type="returnn_w_recomb",
        do_recog=True,
        rasr_recog_epochs=(150,),
        returnn_recog_epochs=(40, 150),
        num_retrain=2,
        realignment_length_scale=0.
      ).run()

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=num_epochs,
        base_alias=base_alias,
        recog_type="huge_beam",
        do_recog=True).run()

  for model_type, model_variants in segmental_model_variants_wo_length_model.items():
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=num_epochs,
        base_alias=base_alias,
        recog_type="huge_beam",
        do_recog=False
      ).run()

  # test realign/retrain pipeline with silence models
  for model_type, model_variants in (
          ("segmental_w_split_silence", segmental_model_variants["segmental_w_split_silence"]),
          ("segmental_w_silence", segmental_model_variants["segmental_w_silence"]),
          ("segmental", segmental_model_variants["segmental"])
  ):
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[40],
        base_alias=base_alias,
        recog_type="standard",
        do_recog=True,
        rasr_recog_epochs=(),
        returnn_recog_epochs=(40,),
        num_retrain=2,
        realignment_length_scale=0.,
        retrain_load_checkpoint=True,
      ).run()

  for model_type, model_variants in global_model_variants.items():
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = global_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      global_train_recog_pipeline = GlobalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=num_epochs,
        base_alias=base_alias,
        do_recog=False
      )
      global_train_recog_pipeline.run()
      checkpoints[variant_name] = global_train_recog_pipeline.checkpoints

  for model_type, model_variants in (
          ("segmental", segmental_model_variants["segmental"]),
  ):
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      global_import_model_name = "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.no-prev-target-in-readout.pretrain-like-seg"

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[6, 12],
        base_alias=base_alias,
        recog_type="standard",
        do_recog=True,
        num_retrain=0,
        import_model_train_epoch1=checkpoints[global_import_model_name]["train"][150],
        import_model_train_epoch1_alias="global-train-150"
      ).run()
