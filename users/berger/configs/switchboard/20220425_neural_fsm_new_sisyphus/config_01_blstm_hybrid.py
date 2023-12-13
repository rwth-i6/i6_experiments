import os

from sisyphus import gs

import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr
import i6_core.text as text

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.hybrid_system as hybrid_system
import i6_experiments.common.setups.rasr.util as rasr_util

import i6_experiments.users.berger.args.jobs.gmm_args as gmm_args
import i6_experiments.users.berger.args.jobs.hybrid_args as hybrid_args
from i6_experiments.users.berger.network.models.blstm_hybrid import (
    make_blstm_hybrid_model,
    make_blstm_hybrid_recog_model,
)
from i6_experiments.users.berger.corpus.switchboard.data import (
    get_data_inputs,
    get_final_gmm_output,
)
from i6_experiments.users.berger.corpus.switchboard.cart_questions import (
    SWBCartQuestions,
)
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args


def py():
    # ********** Settings **********

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    train_key = "switchboard-300h"
    dev_key = "hub5e-00"
    test_key = "hub5-01"

    stm_file = "/u/corpora/speech/hub5e_00/xml/hub5e_00.stm"
    glm_file = "/u/corpora/speech/hub5e_00/xml/glm"

    # train_alignment = "/work/asr4/berger/dependencies/switchboard/alignments/switchboard-300h_tuske/alignment.cache.bundle"
    # dev_alignment = "/work/asr4/berger/dependencies/switchboard/alignments/hub5-00_tuske/alignment.cache.bundle"

    total_train_num_segments = 249623
    cv_size = 3000 / total_train_num_segments

    num_inputs = 40
    num_classes = 9001

    # ********** GMM System **********

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs()
    init_args = get_init_args(sample_rate_kHz=8, scorer="hub5")
    mono_args = gmm_args.get_monophone_args()
    cart_args = gmm_args.get_cart_args(SWBCartQuestions(max_leaves=num_classes))
    tri_args = gmm_args.get_triphone_args()
    vtln_args = gmm_args.get_vtln_args(allow_zero_weights=True)
    sat_args = gmm_args.get_sat_args(allow_zero_weights=True)
    vtln_sat_args = gmm_args.get_vtln_sat_args()
    final_output_args = get_final_gmm_output()

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    swb_gmm_system = gmm_system.GmmSystem()
    swb_gmm_system.init_system(
        hybrid_init_args=init_args,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    swb_gmm_system.stm_files[dev_key] = stm_file
    swb_gmm_system.glm_files[dev_key] = glm_file
    swb_gmm_system.run(steps)

    # ********** Data splits **********

    train_corpus_path = swb_gmm_system.corpora[train_key].corpus_file

    all_segments = corpus_recipe.SegmentCorpusJob(train_corpus_path, 1).out_single_segment_files[1]

    splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_segments, {"train": 1 - cv_size, "cv": cv_size}
    )
    train_segments = splitted_segments_job.out_segments["train"]
    cv_segments = splitted_segments_job.out_segments["cv"]
    # devtrain_segments = text.TailJob(
    #     train_segments, num_lines=1000, zip_output=False
    # ).out

    nn_train_data = swb_gmm_system.outputs[train_key]["final"].as_returnn_rasr_data_input(shuffle_data=True)
    nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
    nn_train_data_inputs = {f"{train_key}.train": nn_train_data}

    nn_cv_data = swb_gmm_system.outputs[train_key]["final"].as_returnn_rasr_data_input()
    nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
    nn_cv_data_inputs = {f"{train_key}.cv": nn_cv_data}

    # nn_devtrain_data = swb_gmm_system.outputs[train_key]["final"].as_returnn_rasr_data_input()
    # nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
    # nn_devtrain_data_inputs = {
    #     f"{train_key}.devtrain": nn_devtrain_data
    #     }
    nn_devtrain_data_inputs = {}

    nn_dev_data = swb_gmm_system.outputs[dev_key]["final"].as_returnn_rasr_data_input()
    nn_dev_data_inputs = {dev_key: nn_dev_data}

    nn_test_data_inputs = {}

    # ********** Hybrid System **********

    train_blstm_net, train_python_code = make_blstm_hybrid_model(num_outputs=num_classes)
    recog_blstm_net, recog_python_code = make_blstm_hybrid_recog_model(num_outputs=num_classes)

    train_networks = {"BLSTM_hybrid": train_blstm_net}
    recog_networks = {"BLSTM_hybrid": recog_blstm_net}

    nn_args = hybrid_args.get_nn_args(
        train_networks=train_networks,
        recog_networks=recog_networks,
        num_inputs=num_inputs,
        num_outputs=num_classes,
        num_epochs=180,
        returnn_train_config_args={"extra_python": train_python_code},
        returnn_recog_config_args={"extra_python": recog_python_code},
        train_args={"partition_epochs": 6},
        recog_keys=[dev_key],
        recog_args={"epochs": [60, 120, 150, 156, 162, 168, 174, 180]},
    )

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("nn", nn_args)

    swb_hybrid_system = hybrid_system.HybridSystem()
    swb_hybrid_system.init_system(
        hybrid_init_args=init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[(f"{train_key}.train", f"{train_key}.cv")],
    )
    swb_hybrid_system.stm_files[dev_key] = stm_file
    swb_hybrid_system.glm_files[dev_key] = glm_file

    swb_hybrid_system.run(nn_steps)
