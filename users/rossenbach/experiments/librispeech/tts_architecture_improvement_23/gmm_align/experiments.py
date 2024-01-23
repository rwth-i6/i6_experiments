"""
Definition of the pipeline in terms of inputs and steps that are executed
"""
import os
from sisyphus import gs, tk

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from . import baseline_args
from .data import get_corpus_data_inputs
from ..default_tools import RASR_GMM_BINARY_PATH


def run_librispeech_100_common_tts_baseline(
    alias_prefix="experiments/librispeech/tts_architecture_improvement_23/gmm_align/baseline",
):

    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    hybrid_init_args = baseline_args.get_init_args()
    mono_args = baseline_args.get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = baseline_args.get_cart_args(add_unknown=False)
    tri_args = baseline_args.get_triphone_args()
    # vtln_args = baseline_args.get_vtln_args()
    sat_args = baseline_args.get_sat_args()
    # vtln_sat_args = baseline_args.get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train-clean-100-tts-train", "train")
    final_output_args.define_corpus_type("tts_align", "test")
    final_output_args.define_corpus_type("dev-other", "dev")
    # enable this if you want to create features for the following training, e.g. Hybrid
    # final_output_args.add_feature_to_extract("gt")

    steps = RasrSteps()
    steps.add_step("extract", hybrid_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    # steps.add_step("forced_align_mono_train",
    #   {"name": "tts_align_mono_train", "target_corpus_key": "train-clean-100-tts-train", "flow": "mfcc+deriv+norm",
    #    "feature_scorer": ("train-clean-100-tts-train", "train_mono"), "corpus_keys": ["train-clean-100-tts-train"]})

    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("sat", sat_args)
    steps.add_step("forced_align_dev_sat",
                   {"name": "tts_align_dev_sat", "target_corpus_key": "train-clean-100-tts-dev",
                    "flow": "mfcc+context+lda",
                    "feature_scorer": ("train-clean-100-tts-train", "train_sat"),
                    "corpus_keys": ["train-clean-100-tts-dev"]})
    steps.add_step("output", final_output_args)

    corpus_data = get_corpus_data_inputs()

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_GMM_BINARY_PATH)
    system.init_system(
        rasr_init_args=hybrid_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )
    system.run(steps)

    #system.align(
    #    name="tts_align",
    #    corpus="tts_align",
    #    feature_scorer=("train-clean-100", "train_mono"),
    #    flow="mfcc+deriv+norm",
    #)

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
    alignments = {}
    from i6_core.text.processing import ConcatenateJob
    for align in ["sat"]:
         align_train = system.alignments["train-clean-100-tts-train"]["train_" + align][0].alternatives["bundle"]
         align_dev = system.alignments["train-clean-100-tts-dev"]["tts_align_dev_" + align].alternatives["bundle"]
         alignments[align] = ConcatenateJob([align_train, align_dev], zip_out=False, out_name="alignment.bundle").out
         tk.register_output(os.path.join(alias_prefix, align + ".bundle"), alignments[align])
    return (
        alignments,
        system.allophone_files["train-clean-100-tts-train"],
    )
