import copy
from sisyphus import tk, gs

from .data import get_corpus_data_inputs_oggzip
from .baseline_args import get_nn_args as get_nn_args_baseline

import i6_core.rasr as rasr
from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem

from .default_tools import RASR_BINARY_PATH, RETURNN_ROOT, RETURNN_EXE


def run_gmm_system_from_common():
    from ...gmm.baseline.baseline_config import run_switchboard_baseline_ldc_v5

    system = run_switchboard_baseline_ldc_v5(recognition=False)
    return system


def run_baseline_gt():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/hybrid/feat/"

    gmm_system = run_gmm_system_from_common()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_oggzip(
        gmm_system,
        partition_epoch={"train": 6, "dev": 1},
        context_window={"classes": 1, "data": 441},
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
    )

    nn_args = get_nn_args_baseline(num_epochs=260)
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    hybrid_nn_gt_system = HybridSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        rasr_binary_path=RASR_BINARY_PATH,
    )
    hybrid_nn_gt_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["switchboard.train", "switchboard.cv"])],
    )
    hybrid_nn_gt_system.run(nn_steps)
