"""
Definition of the pipeline in terms of inputs and steps that are executed
"""
from sisyphus import gs, tk

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from . import baseline_args_v5
from . import baseline_args_v6
from .data import get_corpus_data_inputs

from ..hybrid.switchboard.default_tools import RASR_BINARY_PATH


@tk.block(name="SWDB_LDC_v5")
def run_switchboard_baseline_ldc_v5(
        alias_prefix="baselines/switchboard/gmm_ldc_v5/",
):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    rasr_init_args = baseline_args_v5.get_init_args()
    mono_args = baseline_args_v5.get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = baseline_args_v5.get_cart_args()
    tri_args = baseline_args_v5.get_triphone_args()
    cart_reestimation_args = baseline_args_v5.get_cart_reestimation_args()
    tri_second_pass_args = baseline_args_v5.get_triphone_second_pass_args()
    sat_args = baseline_args_v5.get_sat_args()
    vtln_args = baseline_args_v5.get_vtln_args()
    vtln_sat_args = baseline_args_v5.get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("switchboard", "train")
    final_output_args.define_corpus_type("hub5e00", "dev")
    #final_output_args.define_corpus_type("dev-other", "dev")
    # enable this if you want to create features for the following training, e.g. Hybrid
    final_output_args.add_feature_to_extract("gt")

    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("cart_reestimate", cart_reestimation_args)
    steps.add_step("tri2", tri_second_pass_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    corpus_data = get_corpus_data_inputs(use_legacy=False, use_legacy_lexicon=False, normalize_pronunciation=False)

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )
    system.run(steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir

    return system


@tk.block(name="SWDB_LDC_v6")
def run_switchboard_baseline_ldc_v6(
        alias_prefix="baselines/switchboard/gmm_ldc_v6/",
):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    rasr_init_args = baseline_args_v6.get_init_args()
    mono_args = baseline_args_v6.get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = baseline_args_v6.get_cart_args()
    tri_args = baseline_args_v6.get_triphone_args()
    cart_reestimation_args = baseline_args_v6.get_cart_reestimation_args()
    tri_second_pass_args = baseline_args_v6.get_triphone_second_pass_args()
    sat_args = baseline_args_v6.get_sat_args()
    vtln_args = baseline_args_v6.get_vtln_args()
    vtln_sat_args = baseline_args_v6.get_vtln_sat_args()

    #final_output_args = OutputArgs("final")
    #final_output_args.define_corpus_type("train-clean-100", "train")
    #final_output_args.define_corpus_type("dev-clean", "dev")
    #final_output_args.define_corpus_type("dev-other", "dev")
    # enable this if you want to create features for the following training, e.g. Hybrid
    # final_output_args.add_feature_to_extract("gt")

    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("cart_reestimate", cart_reestimation_args)
    steps.add_step("tri2", tri_second_pass_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    #steps.add_step("output", final_output_args)

    corpus_data = get_corpus_data_inputs(use_legacy=False, use_legacy_lexicon=False, normalize_pronunciation=False)

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )
    system.run(steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
