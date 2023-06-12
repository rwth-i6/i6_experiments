import copy
from sisyphus import gs, tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.features import FilterbankJob

from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.common.baselines.tedlium2.default_tools import RASR_BINARY_PATH, RETURNN_RC_ROOT

from .data import get_corpus_data_inputs
from .baseline_args import get_log_mel_feature_extraction_args
from i6_experiments.common.baselines.tedlium2.hybrid.torch_args import get_nn_args
from i6_experiments.users.rossenbach.common_setups.rasr.pytorch_onnx_hybrid_system import PyTorchOnnxHybridSystem


def run_gmm_system():
    from i6_experiments.common.baselines.tedlium2.gmm.baseline_config import (
        run_tedlium2_common_baseline,
    )

    system = run_tedlium2_common_baseline()
    return system


def run_tedlium2_torch_conformer():
    prefix = "experiments/tedlium2/hybrid/conformer_baseline"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )

    gmm_system = run_gmm_system()

    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_log_mel_feature_extraction_args()

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs(gmm_system, rasr_init_args.feature_extraction_args, FilterbankJob, alias_prefix=prefix)

    nn_args = get_nn_args(num_epochs=125)

    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # image only, so just python3
    returnn_exe = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    blas_lib = tk.Path(
        "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so",
        hash_overwrite="TF23_MKL_BLAS",
    )

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="04c14d6a3745f90f0bd10b28ce4d51b1d61afb82",
    ).out_repository

    tedlium_nn_system = PyTorchOnnxHybridSystem(
        returnn_root=returnn_root,
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
