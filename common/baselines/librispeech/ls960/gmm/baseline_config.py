"""
Definition of the pipeline in terms of inputs and steps that are executed
"""
from sisyphus import gs

# global imports for "static" components

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

# relative imports for files that should be copied for new setups

from ...data import get_corpus_data_inputs
from . import baseline_args
from ...default_tools import RASR_BINARY_PATH


def run_librispeech_960_common_baseline(
    alias_prefix="baselines/librispeech/ls960/gmm/common_baseline",
):

    # the RASR-System pipelines need global alias and output settings
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    # ******************** GMM parameters ********************

    rasr_init_args = baseline_args.get_init_args()
    mono_args = baseline_args.get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = baseline_args.get_cart_args(add_unknown=False)
    tri_args = baseline_args.get_triphone_args()
    vtln_args = baseline_args.get_vtln_args()
    sat_args = baseline_args.get_sat_args()
    vtln_sat_args = baseline_args.get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train-other-960", "train")
    final_output_args.define_corpus_type("dev-clean", "dev")
    final_output_args.define_corpus_type("dev-other", "dev")

    # **************** GMM step definitions *****************

    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    # ******************** Data ********************

    corpus_data = get_corpus_data_inputs(corpus_key="train-other-960", use_g2p_training=True, use_stress_marker=False)

    # ******************** GMM System ********************

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )

    # run everything
    system.run(steps)

    # recover alias and output path settings
    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
