import copy
from sisyphus import tk, gs

from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.common.baselines.librispeech.default_tools import RASR_BINARY_PATH
# from i6_experiments.common.baselines.librispeech.ls100.gmm.baseline_config import run_librispeech_100_common_baseline

from .data import get_corpus_data_inputs
from .baseline_args import get_nn_args, get_feature_extraction_args
from .rc_baseline_config import get_nn_args as get_rc_nn_args
from .default_tools import RETURNN_RC_ROOT

#### gmm imports
from functools import lru_cache

from i6_experiments.common.setups.rasr.util import OutputArgs

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.baselines.librispeech.ls100.gmm import baseline_args
from i6_experiments.common.baselines.librispeech.data import get_corpus_data_inputs as get_corpus_data_inputs_gmm
from .data import  get_corpus_data_inputs_newcv
####


@lru_cache
def run_librispeech_100_common_baseline(
        alias_prefix="baselines/librispeech/ls100/gmm/common_baseline",
):


    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    rasr_init_args = baseline_args.get_init_args()
    rasr_init_args.feature_extraction_args.update(
        get_feature_extraction_args()
    )
    mono_args = baseline_args.get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = baseline_args.get_cart_args(add_unknown=False)
    tri_args = baseline_args.get_triphone_args()
    vtln_args = baseline_args.get_vtln_args()
    sat_args = baseline_args.get_sat_args()
    vtln_sat_args = baseline_args.get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train-clean-100", "train")
    final_output_args.define_corpus_type("dev-clean", "dev")
    final_output_args.define_corpus_type("dev-other", "dev")
    # enable this if you want to create features for the following training, e.g. Hybrid
    final_output_args.add_feature_to_extract("gt")

    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    corpus_data = get_corpus_data_inputs_gmm(
        corpus_key="train-clean-100", use_g2p_training=True, use_stress_marker=False
    )
    tk.register_output("rasr_binaries_librispeech", RASR_BINARY_PATH)
    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )
    system.run(steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
    return system


def run_hybrid_baseline():

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_hybrid/common_baseline'

    gmm_system = run_librispeech_100_common_baseline()
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
    #nn_steps.add_step("extract_gt", rasr_init_args.feature_extraction_args)
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

    gmm_system = run_librispeech_100_common_baseline()
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

    gmm_system = run_librispeech_100_common_baseline()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_feature_extraction_args()
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs(gmm_system)

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
