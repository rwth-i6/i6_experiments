import copy
from sisyphus import gs, tk

from i6_core.features import FilterbankJob

from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.common.baselines.tedlium2.default_tools import RETURNN_RC_ROOT, RASR_BINARY_PATH

from .data import get_corpus_data_inputs
from .baseline_args import get_log_mel_feature_extraction_args
from .nn_config.nn_args import get_nn_args


def run_gmm_system():
    from i6_experiments.common.baselines.tedlium2.gmm.baseline_config import (
        run_tedlium2_common_baseline,
    )

    system = run_tedlium2_common_baseline()
    return system


def run_tedlium2_hybrid_baseline():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "baselines/tedlium2/hybrid/baseline"

    gmm_system = run_gmm_system()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_log_mel_feature_extraction_args()
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs(
        gmm_system,
        rasr_init_args.feature_extraction_args["fb"],
        FilterbankJob,
        alias_prefix="experiments/tedlium2/hybrid/wei_baseline",
    )
    # image only, so just python3
    returnn_exe = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    blas_lib = tk.Path("/lib/x86_64-linux-gnu/liblapack.so.3")
    blas_lib.hash_overwrite = "TEDLIUM2_DEFAULT_RASR_BINARY_PATH"
    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    gmm_system.run(steps)
    nn_args = get_nn_args(num_epochs=160)
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    tedlium_nn_system = HybridSystem(
        returnn_root=RETURNN_RC_ROOT,
        returnn_python_exe=returnn_exe,
        blas_lib=blas_lib,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH,
    )
    tedlium_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train.train", "dev.cv"])],
    )
    tedlium_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""