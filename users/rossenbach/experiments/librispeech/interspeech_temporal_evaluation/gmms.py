"""
Definition of the pipeline in terms of inputs and steps that are executed
"""
from sisyphus import gs, tk

from typing import Any, Dict, Optional

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from i6_experiments.common.baselines.librispeech.ls100.gmm import baseline_args

from .data import get_corpus_data_inputs
from i6_experiments.common.baselines.librispeech.default_tools import RASR_BINARY_PATH


def run_interspeech_synthetic_data_gmm(
    corpus_data,
    extract_additional_rasr_features: Optional[Dict[str, Any]] = None,
) -> gmm_system.GmmSystem:
    """

    :param alias_prefix:
    :param extract_additional_rasr_features: a dict of <feature_key>: <feature_option_dict>,
        used to extract additional RASR-based features to be used on subsequent systems such as Hybrid
    """



    rasr_init_args = baseline_args.get_init_args()

    if extract_additional_rasr_features is not None:
        for feature_key, feature_options in extract_additional_rasr_features.items():
            rasr_init_args.feature_extraction_args[feature_key] = feature_options

    mono_args = baseline_args.get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = baseline_args.get_cart_args(add_unknown=False)
    tri_args = baseline_args.get_triphone_args()
    vtln_args = baseline_args.get_vtln_args()
    sat_args = baseline_args.get_sat_args()
    vtln_sat_args = baseline_args.get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train-clean-100", "train")
    final_output_args.define_corpus_type("dev-clean", "dev")
    final_output_args.define_corpus_type("dev-other", "dev")

    if extract_additional_rasr_features:
        for feature_key in extract_additional_rasr_features.keys():
            final_output_args.add_feature_to_extract(feature_key)

    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)



    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )
    system.run(steps)


    return system


def experiments():
    alias_prefix = "experiments/librispeech/interspeech_temporal_evaluation/sat_gauss_pred"
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix


    corpus_data = get_corpus_data_inputs(
        synthetic_data_bliss=tk.Path("/u/hilmes/experiments/tts_new_sis/output/paper_nick/sat_gauss_pred/real_tags_corpus.xml.gz"), use_g2p_training=True, use_stress_marker=False
    )
    run_interspeech_synthetic_data_gmm(
        corpus_data=corpus_data,
    )


    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
