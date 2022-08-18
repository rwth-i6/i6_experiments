from sisyphus import gs

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

from .data import get_corpus_data_inputs
from . import baseline_args_chris

from .system_collection import add_gmm_system

def run_tina_baseline():
    # ******************** GMM Init ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_gmm/gmm_tina'

    mfcc_cepstrum_options = {
        "normalize": False,
        "outputs": 16,
        "add_epsilon": True,
        "epsilon": 1e-10,
    }

    #gt_options_extra_args = {
    #    "normalize": False,
    #}

    train_data_inputs, dev_data_inputs, test_data_inputs = get_corpus_data_inputs(use_stress_marker=False)
    init_args = baseline_args_chris.get_init_args(
        mfcc_cepstrum_options=mfcc_cepstrum_options,
        #gt_options_extra_args=gt_options_extra_args,
        dc_detection=False,
    )
    mono_args = baseline_args_chris.get_monophone_args(
        train_align_iter=90,
        allow_zero_weights=True
    )
    cart_args = baseline_args_chris.get_cart_args()
    tri_args = baseline_args_chris.get_triphone_args()
    vtln_args = baseline_args_chris.get_vtln_args(allow_zero_weights=True)
    sat_args = baseline_args_chris.get_sat_args(allow_zero_weights=True)
    vtln_sat_args = baseline_args_chris.get_vtln_sat_args()

    mono_args.training_args["dump_alignment_score_report"] = True
    tri_args.training_args["dump_alignment_score_report"] = True
    vtln_args.training_args["dump_alignment_score_report"] = True
    sat_args.training_args["dump_alignment_score_report"] = True
    vtln_sat_args.training_args["dump_alignment_score_report"] = True

    final_output_args = rasr_util.OutputArgs("final")
    final_output_args.define_corpus_type("train-clean-100", "train")
    final_output_args.define_corpus_type("dev-other", "dev")
    final_output_args.add_feature_to_extract("gt")

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    # SAT pipeline was changed
    # steps.add_step("sat", sat_args)
    # steps.add_step("vtln+sat", vtln_sat_args)
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

    add_gmm_system("gmm_tina", lbs_gmm_system, steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ''


def run_tina_baseline_g2p():
    # ******************** GMM Init ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_gmm/gmm_tina_g2p'

    mfcc_cepstrum_options = {
        "normalize": False,
        "outputs": 16,
        "add_epsilon": True,
        "epsilon": 1e-10,
    }

    #gt_options_extra_args = {
    #    "normalize": False,
    #}

    train_data_inputs, dev_data_inputs, test_data_inputs = get_corpus_data_inputs(use_g2p_training=True, use_stress_marker=False)
    init_args = baseline_args_chris.get_init_args(
        mfcc_cepstrum_options=mfcc_cepstrum_options,
        #gt_options_extra_args=gt_options_extra_args,
        dc_detection=False,
    )
    mono_args = baseline_args_chris.get_monophone_args(
        train_align_iter=90,
        allow_zero_weights=True
    )

    cart_args = baseline_args_chris.get_cart_args()
    tri_args = baseline_args_chris.get_triphone_args()
    vtln_args = baseline_args_chris.get_vtln_args(allow_zero_weights=True)
    sat_args = baseline_args_chris.get_sat_args(allow_zero_weights=True)
    vtln_sat_args = baseline_args_chris.get_vtln_sat_args()

    final_output_args = rasr_util.OutputArgs("final")
    final_output_args.define_corpus_type("train-clean-100", "train")
    final_output_args.define_corpus_type("dev-other", "dev")
    final_output_args.add_feature_to_extract("gt")

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    # SAT pipeline was changed
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

    add_gmm_system("gmm_tina_g2p", lbs_gmm_system, steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ''

