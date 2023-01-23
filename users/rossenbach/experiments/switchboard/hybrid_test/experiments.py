import copy
from sisyphus import tk, gs

from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.common.baselines.librispeech.default_tools import RASR_BINARY_PATH as RASR_BINARY_PATH_LS

from ..default_tools import RASR_BINARY_PATH, RETURNN_ROOT

from .baseline_args import get_nn_args
from .baseline_args_jingjing import get_nn_args as get_nn_args_jingjing
from .data import  get_corpus_data_inputs_newcv


def run_gmm_system_from_common():
    from ..gmm_michel.baseline_config import run_switchboard_baseline_ldc_v5
    system = run_switchboard_baseline_ldc_v5()
    return system


def run_hybrid_baseline_newcv():

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/switchboard/hybrid_test/common_baseline_newcv'

    gmm_system = run_gmm_system_from_common()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    nn_args = get_nn_args(num_epochs=125, extra_exps=True)
    nn_args.training_args ["partition_epochs"] = {"train": 20, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6").out_repository
    blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so", hash_overwrite="TF23_MKL_BLAS")


    lbs_nn_system = HybridSystem(returnn_root=returnn_root, returnn_python_exe=returnn_exe, blas_lib=blas_lib,
                                 rasr_arch="linux-x86_64-standard", rasr_binary_path=RASR_BINARY_PATH_LS)
    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["switchboard.train", "switchboard.cv"])],
    )
    lbs_nn_system.run(nn_steps)


def run_hybrid_baseline_jingjing():

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/switchboard/hybrid_test/common_baseline_newcv'

    gmm_system = run_gmm_system_from_common()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    nn_args = get_nn_args_jingjing(num_epochs=240)
    nn_args.training_args ["partition_epochs"] = {"train": 6, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so", hash_overwrite="TF23_MKL_BLAS")


    lbs_nn_system = HybridSystem(returnn_root=RETURNN_ROOT, returnn_python_exe=returnn_exe, blas_lib=blas_lib,
                                 rasr_arch="linux-x86_64-standard", rasr_binary_path=RASR_BINARY_PATH)
    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["switchboard.train", "switchboard.cv"])],
    )
    lbs_nn_system.run(nn_steps)