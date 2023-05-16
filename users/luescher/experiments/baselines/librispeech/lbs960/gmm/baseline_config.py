"""
Definition of the pipeline in terms of inputs and steps that are executed
"""

import os
from sisyphus import gs

# ******************** global imports for "static" components ********************

import i6_core.rasr as rasr

from i6_experiments.users.luescher.setups.rasr.gmm_system import GmmSystem
import i6_experiments.users.luescher.setups.rasr.util as rasr_util

# ******************** (relative) imports for files that should be copied for new setups ********************

from i6_experiments.users.luescher.experiments.baselines.librispeech.data import get_corpus_data_inputs
from i6_experiments.users.luescher.experiments.baselines.librispeech.default_tools import RASR_BINARY_PATH
from . import baseline_args


def run_librispeech_960_gmm_baseline():
    # ******************** Settings ********************

    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    filename_handle = os.path.splitext(os.path.basename(__file__))[0]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** GMM Init ********************

    rasr_init_args = baseline_args.get_init_args()
    mono_args = baseline_args.get_monophone_args()
    cart_args = baseline_args.get_cart_args(max_leaves=9001, add_unknown=True)
    tri_args = baseline_args.get_triphone_args()
    vtln_args = baseline_args.get_vtln_args()
    sat_args = baseline_args.get_sat_args()
    vtln_sat_args = baseline_args.get_vtln_sat_args()
    align_args = baseline_args.get_align_dev_args()
    final_output_args = baseline_args.get_final_output()

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    #steps.add_step("forced_align", align_args)
    steps.add_step("output", final_output_args)

    # ******************** Data ********************

    corpus_data = get_corpus_data_inputs(
        corpus_key="train-other-960", use_g2p_training=True, use_stress_marker=False, add_unknown_phoneme_and_mapping=True,
    )

    # ******************** GMM System ********************

    gmm_system = GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    gmm_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )

    # run everything
    gmm_system.run(steps)

    # recover alias and output path settings
    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir

    return gmm_system
