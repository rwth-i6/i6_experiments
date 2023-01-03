from sisyphus import gs

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from i6_experiments.common.baselines.tedlium2.gmm import baseline_args
from i6_experiments.common.baselines.tedlium2.data import get_corpus_data_inputs

from ..default_tools import RASR_BINARY_PATH


def run_tedlium2_common_baseline(
    alias_prefix="baselines/tedlium2/gmm/common_baseline",
):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    rasr_init_args = baseline_args.get_init_args()
    mono_args = baseline_args.get_monophone_args()
    cart_args = baseline_args.get_cart_args()
    tri_args = baseline_args.get_triphone_args()
    vtln_args = baseline_args.get_vtln_args()
    sat_args = baseline_args.get_sat_args()
    vtln_sat_args = baseline_args.get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train", "train")
    final_output_args.define_corpus_type("dev", "dev")
    final_output_args.define_corpus_type("test", "test")
    # final_output_args.add_feature_to_extract("gt")

    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    corpus_data = get_corpus_data_inputs()

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data["train"],
        dev_data={},  # corpus_data["dev"],
        test_data={},  # corpus_data["test"],
    )
    system.run(steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
