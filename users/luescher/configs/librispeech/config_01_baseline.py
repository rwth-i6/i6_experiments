import os

# -------------------- Sisyphus --------------------

from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as hybrid_util
import i6_experiments.users.luescher.setups.librispeech.pipeline_base_args as lbs_setups

# -------------------- Init --------------------

filename_handle = os.path.splitext(os.path.basename(__file__))[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


def py():
    train_data_inputs, dev_data_inputs, test_data_inputs = lbs_setups.get_data_inputs()
    hybrid_init_args = lbs_setups.get_init_args()
    mono_args = lbs_setups.get_monophone_args(allow_zero_weights=True)
    cart_args = lbs_setups.get_cart_args()
    tri_args = lbs_setups.get_triphone_args()
    vtln_args = lbs_setups.get_vtln_args()
    sat_args = lbs_setups.get_sat_args()
    vtln_sat_args = lbs_setups.get_vtln_sat_args()

    steps = hybrid_util.RasrSteps()
    steps.add_step("extract", hybrid_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    #steps.add_step("vtln", vtln_args)
    #steps.add_step("sat", sat_args)
    #steps.add_step("vtln+sat", vtln_sat_args)

    # -------------------- System --------------------

    lbs_system = gmm_system.GmmSystem()
    lbs_system.init_system(
        hybrid_init_args=hybrid_init_args,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    lbs_system.run(steps)
