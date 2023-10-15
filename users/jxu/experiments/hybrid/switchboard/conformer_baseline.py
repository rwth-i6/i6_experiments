import copy
from sisyphus import tk, gs

from .data import get_corpus_data_inputs_newcv
from recipe.i6_experiments.users.jxu.experiments.hybrid.switchboard.config.tf_baseline_args_jingjing import get_nn_args as get_tf_nn_args_jingjing
from recipe.i6_experiments.users.jxu.experiments.hybrid.switchboard.config.torch_baseline_args_jingjing import get_nn_args as get_torch_nn_args_jingjing
from recipe.i6_experiments.users.jxu.experiments.hybrid.switchboard.config.torch_baseline_args_tune_specaug import get_nn_args as get_torch_nn_args_new_specaug

from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem

from recipe.i6_experiments.users.jxu.experiments.hybrid.switchboard.default_tools import RASR_BINARY_PATH, RETURNN_ROOT


def run_gmm_system_from_common():
    from ...gmm_michel.baseline_config import run_switchboard_baseline_ldc_v5

    system = run_switchboard_baseline_ldc_v5()
    return system


def run_tf_hybrid_baseline_jingjing(peak_lr):

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/hybrid/{}".format(
        str("%.0E" % peak_lr).replace("-", "_").replace("E", "e").replace(".", "")
    )

    gmm_system = run_gmm_system_from_common()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    nn_args = get_tf_nn_args_jingjing(num_epochs=260, peak_lr=peak_lr)
    nn_args.training_args["partition_epochs"] = {"train": 6, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    blas_lib = tk.Path(
        "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so",
        hash_overwrite="TF23_MKL_BLAS",
    )

    hybrid_nn_system = HybridSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=returnn_exe,
        blas_lib=blas_lib,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH,
    )
    hybrid_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["switchboard.train", "switchboard.cv"])],
    )
    hybrid_nn_system.run(nn_steps)



def run_torch_hybrid_baseline_jingjing(peak_lr):
    from i6_experiments.users.rossenbach.common_setups.rasr.pytorch_onnx_hybrid_system import PyTorchOnnxHybridSystem

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/mfcc/{}".format(
        str("%.0E" % peak_lr).replace("-", "_").replace("E", "e").replace(".", "")
    )

    gmm_system = run_gmm_system_from_common()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    nn_args = get_torch_nn_args_jingjing(num_epochs=260, peak_lr=peak_lr)
    nn_args.training_args["partition_epochs"] = {"train": 6, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path(
        "/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    blas_lib = tk.Path(
        "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so",
        hash_overwrite="TF23_MKL_BLAS",
    )

    RASR_BINARY_PATH = tk.Path("/work/asr4/rossenbach/rescale/pytorch_mixed_precision/onnx_extended_rasr/arch/linux-x86_64-standard")
    RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"
    RETURNN_ROOT = tk.Path("/u/jxu/setups/switchboard/2023-02-07-hybrid-new-recipe/tools/returnn", hash_overwrite="RETURNN_ROOT",)
    returnn_exe = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER",)


    hybrid_nn_system = PyTorchOnnxHybridSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=returnn_exe,
        blas_lib=blas_lib,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH,
    )
    hybrid_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["switchboard.train", "switchboard.cv"])],
    )
    hybrid_nn_system.run(nn_steps)


def run_torch_hybrid_baseline_new_specaug(peak_lr):
    from i6_experiments.users.rossenbach.common_setups.rasr.pytorch_onnx_hybrid_system import PyTorchOnnxHybridSystem

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/new_specaug/{}".format(
        str("%.0E" % peak_lr).replace("-", "_").replace("E", "e").replace(".", "")
    )

    gmm_system = run_gmm_system_from_common()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    nn_args = get_torch_nn_args_new_specaug(num_epochs=260, peak_lr=peak_lr)
    nn_args.training_args["partition_epochs"] = {"train": 6, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path(
        "/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    blas_lib = tk.Path(
        "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so",
        hash_overwrite="TF23_MKL_BLAS",
    )

    RASR_BINARY_PATH = tk.Path("/work/asr4/rossenbach/rescale/pytorch_mixed_precision/onnx_extended_rasr/arch/linux-x86_64-standard")
    RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"
    RETURNN_ROOT = tk.Path("/u/jxu/setups/switchboard/2023-02-07-hybrid-new-recipe/tools/returnn", hash_overwrite="RETURNN_ROOT",)
    returnn_exe = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER",)


    hybrid_nn_system = PyTorchOnnxHybridSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=returnn_exe,
        blas_lib=blas_lib,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH,
    )
    hybrid_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["switchboard.train", "switchboard.cv"])],
    )
    hybrid_nn_system.run(nn_steps)