import os
from IPython import embed

# -------------------- Sisyphus --------------------

from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.rasr as rasr

from i6_core.tools import CloneGitRepositoryJob

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.hybrid_system as nn_system
import i6_experiments.common.setups.rasr.util as rasr_util
import i6_experiments.users.luescher.setups.librispeech.pipeline_base_args as lbs_setups
import i6_experiments.users.raissi.experiments.librispeech.data_preparation.clean_100h.data as data_setups


def run():
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** GMM Init ********************

    mfcc_cepstrum_options = {
        "normalize": False,
        "outputs": 16,
        "add_epsilon": True,
        "epsilon": 1e-10,
    }

    gt_options_extra_args = {
        "normalize": False,
    }

    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = data_setups.get_corpus_data_inputs()
    init_args = lbs_setups.get_init_args(
        dc_detection=False,
        mfcc_cepstrum_options=mfcc_cepstrum_options,
        gt_options_extra_args=gt_options_extra_args,
    )
    mono_args = lbs_setups.get_monophone_args(train_align_iter=90, allow_zero_weights=True)

    cart_args = lbs_setups.get_cart_args()
    tri_args = lbs_setups.get_triphone_args()
    vtln_args = lbs_setups.get_vtln_args(allow_zero_weights=True)
    sat_args = lbs_setups.get_sat_args(allow_zero_weights=True)
    vtln_sat_args = lbs_setups.get_vtln_sat_args()
    final_output_args = lbs_setups.get_final_output()

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    # ******************** GMM System ********************

    lbs_gmm_system = gmm_system.GmmSystem()
    lbs_gmm_system.init_system(
        hybrid_init_args=init_args,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    lbs_gmm_system.run(steps)
