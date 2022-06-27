from sisyphus import gs

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from i6_experiments.users.rossenbach.experiments.baselines.librispeech.ls100.gmm import baseline_args
from i6_experiments.users.rossenbach.experiments.baselines.librispeech.data import get_corpus_data_inputs

from ...default_tools import RASR_BINARY_PATH


def run_librispeech_100_common_baseline(
    alias_prefix="baselines/librispeech/ls100/gmm/common_baseline",
):

    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    hybrid_init_args = baseline_args.get_init_args()
    mono_args = baseline_args.get_monophone_args()
    cart_args = baseline_args.get_cart_args()
    tri_args = baseline_args.get_triphone_args()
    vtln_args = baseline_args.get_vtln_args()
    sat_args = baseline_args.get_sat_args()
    vtln_sat_args = baseline_args.get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train-clean-100", "train")
    final_output_args.define_corpus_type("dev-clean", "dev")
    final_output_args.define_corpus_type("dev-other", "dev")
    # enable this if you want to create features for the following training, e.g. Hybrid
    # final_output_args.add_feature_to_extract("gt")

    steps = RasrSteps()
    steps.add_step("extract", hybrid_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    # steps.add_step("cart", cart_args)
    # steps.add_step("tri", tri_args)
    # steps.add_step("vtln", vtln_args)
    # steps.add_step("sat", sat_args)
    # steps.add_step("vtln+sat", vtln_sat_args)
    # steps.add_step("output", final_output_args)

    corpus_data = get_corpus_data_inputs("train-clean-100", use_g2p_training=True)

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        hybrid_init_args=hybrid_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )
    system.run(steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""
