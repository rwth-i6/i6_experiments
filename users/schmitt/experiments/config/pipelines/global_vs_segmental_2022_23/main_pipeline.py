# dependencies
import copy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.label_singletons import RNA_BPE, RNA_BPE_SIL_TIME_RED_6, RNA_BPE_SPLIT_SIL_TIME_RED_6, BPE_LABELS
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.label_singletons import SWBBPE534_LABELS

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.train_recog.segmental import SegmentalTrainRecogPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.train_recog.global_ import GlobalTrainRecogPipeline

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants import build_alias, models, global_model_variants, segmental_model_variants, segmental_model_variants_w_length_model

segmental_dependencies = {
    "rna-bpe": RNA_BPE,
    "rna-bpe-sil-time-red-6": RNA_BPE_SIL_TIME_RED_6,
    "rna-bpe-split-sil-time-red-6": RNA_BPE_SPLIT_SIL_TIME_RED_6,
  }

global_dependencies = {
    "bpe": BPE_LABELS,
    "bpe534": SWBBPE534_LABELS
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
        rasr_recog_epochs=(40, 150),
        base_alias=base_alias,
        recog_type="rasr_wo_length_model",
        do_recog=True,
        num_retrain=0, ).run()
      SegmentalTrainRecogPipeline(
        dependencies=dependencies,
        model_type=model_type,
        variant_name=variant_name,
        variant_params=variant_params,
        num_epochs=[40,  150],
        rasr_recog_epochs=(40, 150),
        base_alias=base_alias,
        recog_type="rasr_wo_length_model_force_nb_in_last_frame",
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
        num_epochs=[900],
        base_alias=base_alias,
        returnn_recog_epochs=[900],
        recog_type="standard",
        do_recog=False).run()

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
        num_retrain=0,
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
        num_retrain=0,
        realignment_length_scale=1.
      ).run()

  """
    Center window with diff win sizes
  """
  for model_type, model_variants in (
          ("center-window_diff_sizes", models["center-window_diff_sizes"]),
  ):
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
      )
      segmental_train_recog_pipeline.run()
      checkpoints[variant_name] = segmental_train_recog_pipeline.checkpoints

  """
    Center window with diff win sizes w/o chunking
  """
  for model_type, model_variants in (
          ("center-window_diff_sizes_wo_chunking", models["center-window_diff_sizes_wo_chunking"]),
  ):
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
      )
      segmental_train_recog_pipeline.run()
      checkpoints[variant_name] = segmental_train_recog_pipeline.checkpoints

  """
    Segmental models with KL divergence
  """
  for model_type, model_variants in (
          ("segmental_kl_div", models["segmental_kl_div"]),
  ):
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
      )
      segmental_train_recog_pipeline.run()
      checkpoints[variant_name] = segmental_train_recog_pipeline.checkpoints

  """
    Segmental models with alignment augmentation
  """
  for model_type, model_variants in (
          ("segmental_align_augment", models["segmental_align_augment"]),
  ):
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
      )
      segmental_train_recog_pipeline.run()
      checkpoints[variant_name] = segmental_train_recog_pipeline.checkpoints

  """
  Realign and retrain after 40 epochs for 2 iterations
  """
  for model_type, model_variants in (
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


  """
  Import global attention model parameters into segmental model and train for a few epochs
  """
  for model_type, model_variants in (
          ("segmental", segmental_model_variants["segmental"]),
          ("center-window", segmental_model_variants["center-window"]),
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
        num_epochs=[40, 150],
        base_alias=base_alias,
        recog_type="standard",
        do_recog=True,
        num_retrain=1,
        realignment_length_scale=0.,
        retrain_load_checkpoint=False,
        import_model_initial_realignment=checkpoints[global_import_model_name]["train"][150],
        import_model_initial_realignment_alias="global-train-150",
        import_model_initial_realignment_is_global=True
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
        import_model_initial_realignment=checkpoints[segmental_import_model_name]["train"][40],
        import_model_initial_realignment_alias="segmental-train-40"
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
        import_model_initial_realignment=checkpoints[segmental_import_model_name]["train"][40],
        import_model_initial_realignment_alias="segmental-train-40",
        retrain_reset_lr=False,
        import_model_train_epoch1_initial_lr=0.00049
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
        import_model_initial_realignment=checkpoints[segmental_import_model_name]["train"][40],
        import_model_initial_realignment_alias="segmental-train-40",
        retrain_reset_lr=False,
        import_model_train_epoch1_initial_lr=0.00049,
        retrain_choose_best_alignment=True
      ).run()

