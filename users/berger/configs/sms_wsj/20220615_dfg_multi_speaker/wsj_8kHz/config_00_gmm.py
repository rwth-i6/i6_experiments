import os
from typing import Dict

from sisyphus import gs, tk

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

import i6_experiments.users.berger.args.jobs.gmm_args as gmm_args
from i6_experiments.users.berger.corpus.sms_wsj.data import (
    get_data_inputs,
    get_final_gmm_output,
)
from i6_experiments.users.berger.corpus.sms_wsj.cart_questions import WSJCartQuestions
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

rasr_binary_path = tk.Path("/u/berger/rasr_tf2/arch/linux-x86_64-standard")

train_key = "train_si284"
dev_key = "cv_dev93"
test_key = "test_eval92"

speechsource_train_key = "sms_train_si284_speechsource"
speechsource_dev_key = "sms_cv_dev93_speechsource"

frequency = 8
num_inputs = 40
num_classes = 9001


def run_exp() -> Dict[str, rasr_util.GmmOutput]:

    # ********** GMM System **********

    (train_data_inputs, dev_data_inputs, test_data_inputs, align_data_inputs,) = get_data_inputs(
        train_keys=[train_key],
        dev_keys=[dev_key],
        test_keys=[test_key],
        align_keys=[train_key, dev_key, speechsource_train_key, speechsource_dev_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        add_all_allophones=False,
    )

    init_args = get_init_args(sample_rate_kHz=frequency)
    mono_args = gmm_args.get_monophone_args()
    cart_args = gmm_args.get_cart_args(WSJCartQuestions(max_leaves=num_classes))
    tri_args = gmm_args.get_triphone_args()
    vtln_args = gmm_args.get_vtln_args(allow_zero_weights=True)
    sat_args = gmm_args.get_sat_args(allow_zero_weights=True)
    vtln_sat_args = gmm_args.get_vtln_sat_args()
    final_output_args = get_final_gmm_output(align_keys=align_data_inputs.keys())

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    wsj_gmm_system = gmm_system.GmmSystem(rasr_binary_path=rasr_binary_path)
    wsj_gmm_system.init_system(
        rasr_init_args=init_args,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
        align_data=align_data_inputs,
    )
    wsj_gmm_system.run(steps)

    return {corpus: output["final"] for corpus, output in wsj_gmm_system.outputs.items()}


def py() -> Dict[str, rasr_util.GmmOutput]:
    dir_handle = os.path.dirname(__file__).split("config/")[1]
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"

    return run_exp()
