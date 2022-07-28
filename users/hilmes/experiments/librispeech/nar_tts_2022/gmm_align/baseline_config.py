"""
Definition of the pipeline in terms of inputs and steps that are executed
"""
from sisyphus import gs, tk

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.gmm_align.default_tools import RASR_BINARY_PATH

from i6_private.users.hilmes.nar_tts.GMM import baseline_args
from i6_private.users.hilmes.nar_tts.GMM.data import get_corpus_data_inputs, get_synth_corpus_data_inputs


def run_librispeech_100_common_baseline(
  alias_prefix="baselines/librispeech/ls100/gmm_align/common_baseline",
):
  """baseline for gmm_align model with librispeech-100"""
  stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
  gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

  hybrid_init_args = baseline_args.get_init_args()
  mono_args = baseline_args.get_monophone_args()
  # no unknown question needed when G2P is used
  cart_args = baseline_args.get_cart_args(add_unknown=False)
  tri_args = baseline_args.get_triphone_args()
  vtln_args = baseline_args.get_vtln_args()
  sat_args = baseline_args.get_sat_args()
  vtln_sat_args = baseline_args.get_vtln_sat_args()

  final_output_args = OutputArgs("final")
  final_output_args.define_corpus_type("train-clean-100", "train")

  steps = RasrSteps()
  steps.add_step("extract", hybrid_init_args.feature_extraction_args)
  steps.add_step("mono", mono_args)
  # steps.add_step("cart", cart_args)
  # steps.add_step("tri", tri_args)
  # steps.add_step("vtln", vtln_args)
  # steps.add_step("sat", sat_args)
  # steps.add_step("vtln+sat", vtln_sat_args)
  steps.add_step("output", final_output_args)

  corpus_data = get_corpus_data_inputs()

  system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
  system.init_system(
    hybrid_init_args=hybrid_init_args,
    train_data=corpus_data.train_data,
    dev_data=corpus_data.dev_data,
    test_data=corpus_data.test_data,
  )
  system.run(steps)
  system.align(
    name="tts_align", corpus="tts_align", feature_scorer=("train-clean-100", "train_mono"), flow="mfcc+deriv+norm"
  )

  gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
  return system.alignments["tts_align"]["tts_align"].alternatives["bundle"], system.allophone_files["base"]


def run_librispeech_100_synthesized_training(alias_prefix, synth_corpus):
  """
  synthetic training for gmm_align model with librispeech-100

  :param alias_prefix:
  :param synth_corpus
  """
  stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
  gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

  hybrid_init_args = baseline_args.get_init_args()
  mono_args = baseline_args.get_monophone_args()
  # no unknown question needed when G2P is used
  cart_args = baseline_args.get_cart_args(add_unknown=False)
  tri_args = baseline_args.get_triphone_args()
  vtln_args = baseline_args.get_vtln_args()
  sat_args = baseline_args.get_sat_args()
  vtln_sat_args = baseline_args.get_vtln_sat_args()

  final_output_args = OutputArgs("final")
  final_output_args.define_corpus_type("train-clean-100", "train")

  steps = RasrSteps()
  steps.add_step("extract", hybrid_init_args.feature_extraction_args)
  steps.add_step("mono", mono_args)
  # steps.add_step("cart", cart_args)
  # steps.add_step("tri", tri_args)
  # steps.add_step("vtln", vtln_args)
  # steps.add_step("sat", sat_args)
  # steps.add_step("vtln+sat", vtln_sat_args)
  steps.add_step("output", final_output_args)

  corpus_data = get_synth_corpus_data_inputs(synth_corpus)

  system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
  system.init_system(
    hybrid_init_args=hybrid_init_args,
    train_data=corpus_data.train_data,
    dev_data=corpus_data.dev_data,
    test_data=corpus_data.test_data,
  )
  system.run(steps)

  gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
