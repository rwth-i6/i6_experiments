"""
Definition of the pipeline in terms of inputs and steps that are executed
"""
from sisyphus import gs, tk

import copy
from i6_core.rasr.config import RasrConfig
from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from . import baseline_args
from . import baseline_args_v2
from . import baseline_args_v4
from .data import get_corpus_data_inputs

from ..default_tools import RASR_BINARY_PATH



def run_switchboard_baseline_ldc_v2(
        alias_prefix="baselines/switchboard/gmm_ldc_v3/",
):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    rasr_init_args = baseline_args_v2.get_init_args()
    mono_args = baseline_args_v2.get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = baseline_args_v2.get_cart_args()
    tri_args = baseline_args_v2.get_triphone_args()
    vtln_args = baseline_args_v2.get_vtln_args()
    vtln_sat_args = baseline_args_v2.get_vtln_sat_args()

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
    steps.add_step("vtln", vtln_args)
    #steps.add_step("sat", sat_args)
    #steps.add_step("vtln+sat", vtln_sat_args)
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

    # if use_legacy_lexicon is True and normalize_pronunciation is True:
    #     extra_config = RasrConfig()
    #     # extra_config.flf_lattice_tool.global_cache.file = tk.Path("/u/michel/setups/SWB_sis/work/recognition/advanced_tree_search/AdvancedTreeSearchLmImageAndGlobalCacheJob.3pCmRyPgYPl1/output/global.cache", cached=True)
    #     extra_config.flf_lattice_tool.lexicon.file = tk.Path("/u/corpora/speech/switchboard-1/lexicon/train.lex.v1_0_4.ci.gz", cached=True)
    #     extra_config.flf_lattice_tool.network.recognizer.acoustic_model.allophones.add_from_file = tk.Path("/u/michel/setups/SWB_sis/work/allophones/StoreAllophones.wNiR4cF7cdOE/output/allophones")
    #     extra_config.flf_lattice_tool.network.recognizer.acoustic_model.mixture_set.file = tk.Path("/u/michel/setups/SWB_sis/work/mm/mixtures/EstimateMixturesJob.accumulate.sDN8IK78DmO7/output/am.mix", cached=True)
    #     extra_config.flf_lattice_tool.network.recognizer.acoustic_model.state_tying.file = tk.Path("/u/michel/setups/SWB_sis/work/cart/estimate/EstimateCartJob.Wxfsr7efOgnu/output/cart.tree.xml.gz")
    #     recog_args = copy.deepcopy(tri_args.recognition_args)
    #     recog_args["iters"] = [10]
    #     system.recognition(
    #         name="tri_michel_am",
    #         corpus_key="hub5e00",
    #         feature_scorer_key=("switchboard", "train_tri"),
    #         extra_config=extra_config,
    #         **recog_args,
    #     )

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir

def run_switchboard_baseline_ldc_v4(
        alias_prefix="baselines/switchboard/gmm_ldc_v4/",
):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    rasr_init_args = baseline_args_v4.get_init_args()
    mono_args = baseline_args_v4.get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = baseline_args_v4.get_cart_args()
    tri_args = baseline_args_v4.get_triphone_args()
    sat_args = baseline_args_v4.get_sat_args()
    vtln_args = baseline_args_v4.get_vtln_args()
    vtln_sat_args = baseline_args_v4.get_vtln_sat_args()

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
