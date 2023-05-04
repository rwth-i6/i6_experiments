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

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants import build_alias, models, global_model_variants, segmental_model_variants, segmental_model_variants_w_length_model

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

  """
    Train models with KL divergence
  """
  for model_type, model_variants in (("segmental_w_kl_div", models["segmental_w_kl_div"]),):
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
        num_retrain=0, ).run()
      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[40],
        base_alias=base_alias,
        recog_type="rasr_wo_length_model",
        do_recog=True,
        num_retrain=0, ).run()

  for model_type, model_variants in (("segmental", models["segmental"]),):
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[40, 150],
        rasr_recog_epochs=(40,),
        base_alias=base_alias,
        recog_type="rasr_wo_length_model",
        do_recog=True,
        num_retrain=0, ).run()

  """
    Train conformer models
  """
  for model_type, model_variants in (("segmental_conformer", models["segmental_conformer"]),):
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[40, 150],
        base_alias=base_alias,
        recog_type="standard",
        do_recog=False,
        num_retrain=0, ).run()

  """
    Train conformer models
  """
  for model_type, model_variants in (("global_conformer", models["global_conformer"]),):
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = global_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      GlobalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[40, 150],
        base_alias=base_alias,
        recog_type="standard",
        do_recog=False).run()

  """
    Train model without chunking fix
  """
  for model_type, model_variants in (
          ("segmental_wo_chunk_fix", models["segmental_wo_chunk_fix"]),
          ("center-window_wo_chunk_fix", models["center-window_wo_chunk_fix"])):
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[40, 150],
        base_alias=base_alias,
        recog_type="standard",
        do_recog=True,
        num_retrain=0, ).run()

  """
  Standard segmental and center-window variants
  """
  for model_type, model_variants in segmental_model_variants_w_length_model.items():
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      segmental_train_recog_pipeline = SegmentalTrainRecogPipeline(
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
      )
      segmental_train_recog_pipeline.run()
      checkpoints[variant_name] = segmental_train_recog_pipeline.checkpoints

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
        realignment_length_scale=1.
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

  """
  Realign and retrain after 40 epochs for 2 iterations
  """
  for model_type, model_variants in (
          ("segmental_w_split_silence", segmental_model_variants["segmental_w_split_silence"]),
          ("segmental_w_silence", segmental_model_variants["segmental_w_silence"]),
          # ("segmental", segmental_model_variants["segmental"])
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

  """
  Train standard global attention models
  """
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
        do_recog=True
      )
      global_train_recog_pipeline.run()
      checkpoints[variant_name] = global_train_recog_pipeline.checkpoints

      global_import_model_name = "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.no-prev-target-in-readout.pretrain-like-seg"

      if variant_name == global_import_model_name:
        GlobalTrainRecogPipeline(
          dependencies=dependencies,
          model_type=model_type,
          variant_name=variant_name,
          variant_params=variant_params,
          num_epochs=[6, 12, 18, 24, 30, 36, 42],
          base_alias=base_alias,
          do_recog=True,
          import_model_train_epoch1=checkpoints[global_import_model_name]["train"][150],
          import_model_train_epoch1_alias="global-train-150"
        ).run()


  """
  Import global attention model parameters into segmental model and train for a few epochs
  """
  for model_type, model_variants in (
          ("segmental", segmental_model_variants["segmental"]),
  ):
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      global_import_model_name = "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.no-prev-target-in-readout.pretrain-like-seg"
      segmental_import_model_name = "seg.rna-bpe.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg"

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[6, 12, 18, 24, 30, 36, 42],
        base_alias=base_alias,
        recog_type="standard",
        do_recog=True,
        num_retrain=0,
        import_model_train_epoch1=checkpoints[global_import_model_name]["train"][150],
        import_model_train_epoch1_alias="global-train-150"
      ).run()

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[40, 150],
        base_alias=base_alias,
        recog_type="standard",
        do_recog=True,
        num_retrain=1,
        realignment_length_scale=0.,
        retrain_load_checkpoint=False,
        import_model_train_epoch1=checkpoints[global_import_model_name]["train"][150],
        import_model_train_epoch1_alias="global-train-150",
        import_model_do_initial_realignment=True,
        import_model_is_global=True
      ).run()

      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[12],
        base_alias=base_alias,
        recog_type="standard",
        do_recog=True,
        num_retrain=8,
        realignment_length_scale=0.,
        retrain_load_checkpoint=True,
        import_model_train_epoch1=checkpoints[segmental_import_model_name]["train"][40],
        import_model_train_epoch1_alias="segmental-train-40",
        import_model_do_initial_realignment=True
      ).run()

  """
    Start from epoch 40 and train/realign after 10 epochs
  """
  for model_type, model_variants in (
          ("segmental_w_split_silence", segmental_model_variants["segmental_w_split_silence"]),
  ):
    for variant_name, variant_params in model_variants.items():
      # select the correct dependencies for the current model
      dependencies = segmental_dependencies[variant_params["config"]["label_type"]]

      base_alias = "models/%s/%s" % (model_type, variant_name)

      segmental_import_model_name = "seg.rna-bpe-split-sil-time-red-6.lr-meas-lab.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg"

      run_rasr_segmental_realignment(
        dependencies=dependencies,
        variant_params=variant_params,
        base_alias=base_alias + "_realignment_test",
        checkpoint=checkpoints[segmental_import_model_name]["train"][40],
        corpus_key="cv_test",
        length_scale=0.,
        length_norm=False,
        tags_for_analysis=["switchboard-1/sw02022A/sw2022A-ms98-a-0002"],
        label_pruning_limit=5000,
        time_rqmt=30
      )
