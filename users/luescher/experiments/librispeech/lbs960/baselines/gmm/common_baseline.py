import os

# -------------------- Sisyphus --------------------

from sisyphus import gs

# -------------------- Recipes --------------------

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util
import i6_experiments.users.luescher.setups.librispeech.pipeline_base_args as lbs_gmm_setups

from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode


def run_librispeech_960_common_baseline():
    # ******************** Settings ********************

    filename_handle = os.path.splitext(os.path.basename(__file__))[0]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    rasr_binary_path = compile_rasr_binaries_i6mode()

    # ******************** GMM Init ********************

    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_gmm_setups.get_data_inputs(
        use_eval_data_subset=True,
    )
    hybrid_init_args = lbs_gmm_setups.get_init_args()
    mono_args = lbs_gmm_setups.get_monophone_args()
    cart_args = lbs_gmm_setups.get_cart_args(max_leaves=12001)
    tri_args = lbs_gmm_setups.get_triphone_args()
    vtln_args = lbs_gmm_setups.get_vtln_args()
    sat_args = lbs_gmm_setups.get_sat_args()
    vtln_sat_args = lbs_gmm_setups.get_vtln_sat_args()
    final_output_args = lbs_gmm_setups.get_final_output()

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", hybrid_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    # ******************** GMM System ********************

    lbs_gmm_system = gmm_system.GmmSystem(rasr_binary_path=rasr_binary_path)
    lbs_gmm_system.init_system(
        hybrid_init_args=hybrid_init_args,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    lbs_gmm_system.run(steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""
