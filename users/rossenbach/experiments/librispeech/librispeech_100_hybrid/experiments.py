import copy
from sisyphus import tk, gs

from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.common.baselines.librispeech.default_tools import RASR_BINARY_PATH


from .data import get_corpus_data_inputs
from .configs.legacy_baseline import get_nn_args, get_feature_extraction_args
from .configs.rc_baseline import get_nn_args as get_rc_nn_args
from .configs.pytorch_baseline import get_nn_args as get_pytorch_nn_args
from .default_tools import RETURNN_RC_ROOT, RASR_BINARY_PATH_U22, RASR_BINARY_PATH_APPTEK_APPTAINER

from .data import  get_corpus_data_inputs_newcv, get_corpus_data_inputs_newcv_hdf


def run_gmm_system():
    from .gmm_baseline import run_librispeech_100_common_baseline
    system = run_librispeech_100_common_baseline(
        extract_additional_rasr_features=get_feature_extraction_args()
    )
    return system

def run_gmm_system_v2():
    from .gmm_baseline import run_librispeech_100_common_baseline
    system = run_librispeech_100_common_baseline(
        extract_additional_rasr_features=get_feature_extraction_args(fix_features_output=True)
    )
    return system


def run_hybrid_baseline():

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_hybrid/common_baseline'

    gmm_system = run_gmm_system()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_feature_extraction_args()

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs(gmm_system)

    nn_args = get_nn_args()
    nn_args_2 = get_nn_args()
    nn_args_2.training_args["test_prefix"] = True
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)
    #nn2_args = get_nn_args()
    #nn2_args.recognition_args["dev-other"]["search_parameters"]["beam-pruning"] = 16.0
    #nn_steps.add_step("nn2", nn_args_2)

    # ******************** NN System ********************

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6").out_repository
    blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so", hash_overwrite="TF23_MKL_BLAS")


    lbs_nn_system = HybridSystem(returnn_root=returnn_root, returnn_python_exe=returnn_exe, blas_lib=blas_lib,
                                 rasr_arch="linux-x86_64-standard", rasr_binary_path=RASR_BINARY_PATH)
    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-clean-100.train", "train-clean-100.cv"])],
    )
    lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""


def run_hybrid_baseline_newcv():

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_hybrid/common_baseline_newcv'

    gmm_system = run_gmm_system()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_feature_extraction_args()

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    nn_args = get_nn_args(num_epochs=125, extra_exps=True)
    nn_args.training_args ["partition_epochs"] = {"train": 10, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6").out_repository
    blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so", hash_overwrite="TF23_MKL_BLAS")


    lbs_nn_system = HybridSystem(returnn_root=returnn_root, returnn_python_exe=returnn_exe, blas_lib=blas_lib,
                                 rasr_arch="linux-x86_64-standard", rasr_binary_path=RASR_BINARY_PATH)
    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-clean-100.train", "train-clean-100.cv"])],
    )
    lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""


def run_hybrid_baseline_rc():

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_hybrid/common_baseline_rc_newcv'

    gmm_system = run_gmm_system()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_feature_extraction_args()
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    nn_args = get_rc_nn_args(num_epochs=125)
    nn_args.training_args ["partition_epochs"] = {"train": 10, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so", hash_overwrite="TF23_MKL_BLAS")

    lbs_nn_system = HybridSystem(returnn_root=RETURNN_RC_ROOT, returnn_python_exe=returnn_exe, blas_lib=blas_lib,
                                 rasr_arch="linux-x86_64-standard", rasr_binary_path=RASR_BINARY_PATH)
    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-clean-100.train", "train-clean-100.cv"])],
    )
    lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""
    
    
def run_hybrid_baseline_pytorch():

    from i6_experiments.users.rossenbach.common_setups.rasr.pytorch_onnx_hybrid_system import PyTorchOnnxHybridSystem
    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_hybrid/common_baseline_pytorch'

    gmm_system = run_gmm_system()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_feature_extraction_args()
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    nn_args = get_pytorch_nn_args(num_epochs=125, debug=True)
    nn_args.training_args ["partition_epochs"] = {"train": 10, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_pt20_experimental.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so", hash_overwrite="TF23_MKL_BLAS")

    returnn_root = tk.Path("/u/rossenbach/src/returnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")
    lbs_nn_system = PyTorchOnnxHybridSystem(returnn_root=returnn_root, returnn_python_exe=returnn_exe, blas_lib=blas_lib,
                                 rasr_arch="linux-x86_64-standard", rasr_binary_path=RASR_BINARY_PATH_APPTEK_APPTAINER)
    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-clean-100.train", "train-clean-100.cv"])],
    )
    lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""


def run_hybrid_baseline_rc_hdf():

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_hybrid/common_baseline_rc_newcv_hdf'

    gmm_system = run_gmm_system_v2()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv_hdf(gmm_system)

    nn_args = get_rc_nn_args(num_epochs=125, use_rasr_returnn_training=False)
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so", hash_overwrite="TF23_MKL_BLAS")

    lbs_nn_system = HybridSystem(returnn_root=RETURNN_RC_ROOT, returnn_python_exe=returnn_exe, blas_lib=blas_lib,
                                 rasr_arch="linux-x86_64-standard", rasr_binary_path=RASR_BINARY_PATH)
    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-clean-100.train", "train-clean-100.cv"])],
    )
    lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""
