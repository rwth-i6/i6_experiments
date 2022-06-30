import time

from sisyphus import gs

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from .args import get_init_args, get_monophone_args, get_cart_args, get_triphone_args, get_vtln_args, get_sat_args, get_vtln_sat_args
from .data import get_corpus_data_inputs

from ..default_tools import RASR_BINARY_PATH

def run_ls100_gmm_aligner():

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/alignment_analysis_tts/ls_gmm_aligner/default_pipeline'
    hybrid_init_args = get_init_args()
    mono_args = get_monophone_args()
    cart_args = get_cart_args(use_stress_marker=False)  # folded = without stress marker
    tri_args = get_triphone_args()
    vtln_args = get_vtln_args()
    sat_args = get_sat_args()
    vtln_sat_args = get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train-clean-100", "train")
    # final_output_args.define_corpus_type("dev-clean", "dev")
    # final_output_args.define_corpus_type("dev-other", "dev")
    # final_output_args.add_feature_to_extract("gt")

    steps = RasrSteps()
    steps.add_step("extract", hybrid_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)
    train, dev, test = get_corpus_data_inputs()

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    start = time.time()
    system.init_system(hybrid_init_args=get_init_args(),
                       train_data=train,
                       dev_data=dev,
                       test_data=test)
    print("init_system took: %.1f" % (time.time()-start))
    system.run(steps)
    gs.ALIAS_AND_OUTPUT_SUBDIR = ''
