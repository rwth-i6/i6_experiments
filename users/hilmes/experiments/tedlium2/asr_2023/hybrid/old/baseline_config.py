import copy
from sisyphus import gs, tk

from i6_core.features import FilterbankJob

from i6_experiments.users.hilmes.common.setups.rasr.util import RasrSteps
from i6_experiments.users.hilmes.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.common.baselines.librispeech.default_tools import RASR_BINARY_PATH

from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.data import get_corpus_data_inputs
from .baseline_args import get_log_mel_feature_extraction_args
from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.wei_tedlium2_setup.nn_args import get_nn_args
from i6_experiments.users.luescher.experiments.baselines.librispeech.lbs960.hybrid.default_tools import RETURNN_RC_ROOT


def run_gmm_system():
    from i6_experiments.users.hilmes.common.tedlium2.gmm.baseline_config import (
        run_tedlium2_common_baseline,
    )

    system = run_tedlium2_common_baseline()
    return system


def run_tedlium2_hybrid_baseline():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/tedlium2/hybrid/wei_baseline"

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
        rasr_init_args.feature_extraction_args,
        FilterbankJob,
        alias_prefix="experiments/tedlium2/hybrid/wei_baseline",
    )

    nn_args = get_nn_args(num_epochs=125)

    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    blas_lib = tk.Path(
        "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so",
        hash_overwrite="TF23_MKL_BLAS",
    )

    lbs_nn_system = HybridSystem(
        returnn_root=RETURNN_RC_ROOT,
        returnn_python_exe=returnn_exe,
        blas_lib=blas_lib,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH,
    )
    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train.train", "dev.cv"])],
    )
    lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""
