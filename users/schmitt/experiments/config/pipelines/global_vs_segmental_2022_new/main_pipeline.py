# dependencies
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.general import SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.bpe.rna import RNABPE

# experiments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.experiments.segmental_experiments import ReturnnFrameWiseSimpleBeamSearchPipeline, RasrFrameWiseSegmentalBeamSearchPipeline
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.experiments.training import SegmentalTrainExperiment

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.model_variants import build_alias, seg_model_variants

segmental_dependencies = {
    "rna-bpe": RNABPE()
  }
segmental_dependency_alias_mapping = {
  "bpe": "rna-bpe"
}


def select_segmental_dependencies(alias: str) -> SegmentalLabelDefinition:
  assert alias in segmental_dependency_alias_mapping
  return segmental_dependencies[segmental_dependency_alias_mapping[alias]]


def run_pipeline():
  # for each model, check if name is according to my naming conventions
  for variant_name, variant_params in seg_model_variants.items():
    check_name = "" + build_alias(**variant_params["config"])
    assert variant_name == check_name, "\n{} \t should be \n{}".format(variant_name, check_name)

  ######################################## Training ########################################

  # for every model, store the checkpoint
  checkpoints = {}

  num_epochs = [1]

  for variant_name, variant_params in seg_model_variants.items():
    # select the correct dependencies for the current model
    dependencies = select_segmental_dependencies(variant_params["config"]["label_type"])

    checkpoints[variant_name] = {}
    checkpoints[variant_name]["train1"] = SegmentalTrainExperiment(
      dependencies=dependencies,
      variant_params=variant_params,
      variant_name=variant_name,
      num_epochs=num_epochs,
      alias="train1").run_training()

  ######################################## Recognition ########################################

  for variant_name, variant_params in seg_model_variants.items():
    # select the correct dependencies for the current model
    dependencies = select_segmental_dependencies(variant_params["config"]["label_type"])

    ######################################## RETURNN FRAME-WISE SIMPLE BEAM SEARCH ####################################
    """
    We do the simple RETURNN search for all stored checkpoints, as it is really fast.
    """

    for i, (epoch, checkpoint) in enumerate(checkpoints[variant_name]["train1"].items()):
      if i == len(checkpoints[variant_name]["train1"]) - 1:
        calc_search_errors, plot_att_weights, compare_to_ground_truth_align = True, True, True
      else:
        calc_search_errors, plot_att_weights, compare_to_ground_truth_align = False, False, False

      ReturnnFrameWiseSimpleBeamSearchPipeline(
        dependencies=dependencies,
        variant_params=variant_params,
        variant_name=variant_name,
        checkpoint=checkpoint,
        epoch=epoch,
        calc_search_errors=calc_search_errors,
        plot_att_weights=plot_att_weights,
        compare_to_ground_truth_align=compare_to_ground_truth_align
      ).run()

    ######################################## RASR FRAME-WISE SEGMENTAL BEAM SEARCH ####################################
    """
    We do the more complex RASR search only for the last stored checkpoint, as it is computationally expensive
    """

    last_epoch, last_checkpoint = list(checkpoints[variant_name]["train1"].items())[-1]
    RasrFrameWiseSegmentalBeamSearchPipeline(
      length_norm=False,
      label_pruning=None,
      label_pruning_limit=12,
      word_end_pruning=None,
      word_end_pruning_limit=12,
      full_sum_decoding=False,
      allow_recombination=True,
      max_segment_len=20,
      dependencies=dependencies,
      variant_params=variant_params,
      variant_name=variant_name,
      checkpoint=last_checkpoint,
      epoch=last_epoch,
      calc_search_errors=True,
      plot_att_weights=True,
      compare_to_ground_truth_align=True
    )

