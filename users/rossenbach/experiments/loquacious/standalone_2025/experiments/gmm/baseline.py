"""
Definition of the pipeline in terms of inputs and steps that are executed
"""
from sisyphus import gs, tk

from typing import Any, Dict, Optional

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from ...default_tools import RASR_BINARY_PATH, MINI_RETURNN_ROOT

from .config import get_init_args, get_monophone_args, get_cart_args, get_triphone_args, get_cart2_args, get_triphone2_args
from ...data.gmm import get_corpus_data_inputs
from ...storage import add_label_alignment

def run_loquacious_gmm_baseline(
        alias_prefix="experiments/loquacious/special_gmm/baseline",
        extract_additional_rasr_features: Optional[Dict[str, Any]] = None,
) -> gmm_system.GmmSystem:
    """

    :param alias_prefix:
    :param extract_additional_rasr_features: a dict of <feature_key>: <feature_option_dict>,
        used to extract additional RASR-based features to be used on subsequent systems such as Hybrid
    """

    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    rasr_init_args = get_init_args()

    if extract_additional_rasr_features is not None:
        for feature_key, feature_options in extract_additional_rasr_features.items():
            rasr_init_args.feature_extraction_args[feature_key] = feature_options

    mono_args = get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = get_cart_args(add_unknown=False)
    tri_args = get_triphone_args()
    cart2_args = get_cart2_args(add_unknown=False)
    tri2_args = get_triphone2_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train.small", "train")
    final_output_args.define_corpus_type("dev.short", "dev")

    if extract_additional_rasr_features:
        for feature_key in extract_additional_rasr_features.keys():
            final_output_args.add_feature_to_extract(feature_key)

    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("cart2", cart2_args)
    steps.add_step("tri2", tri2_args)
    steps.add_step("output", final_output_args)

    corpus_data = get_corpus_data_inputs(corpus_key="train.small", use_g2p_training=True, use_stress_marker=False)

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )
    system.run(steps)


    # print(system.outputs["train.small"]["final"].alignments.alternatives["task_dependent"].hidden_paths.values())
    print(system.allophone_files["train.small"])
    print(system.jobs["train.small"]["state_tying_cart_mono"].out_state_tying)

    from i6_core.returnn.hdf import RasrAlignmentDumpHDFJob
    dump_hdf = RasrAlignmentDumpHDFJob(
        alignment_caches=list(system.outputs["train.small"]["final"].alignments.alternatives["task_dependent"].hidden_paths.values()),
        allophone_file=system.allophone_files["train.small"],
        state_tying_file=system.jobs["train.small"]["state_tying_cart_mono"].out_state_tying,
        returnn_root=MINI_RETURNN_ROOT,
    )
    tk.register_output("gmm_test_align.hdf", dump_hdf.out_hdf_files[0])
    add_label_alignment("triphone_train.small", dump_hdf.out_hdf_files)






    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix + "_novariants"
    #### same with new lex
    corpus_data = get_corpus_data_inputs(corpus_key="train.small", use_g2p_training=True, use_stress_marker=False, variant=2)

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )
    system.run(steps)
    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir

    return system