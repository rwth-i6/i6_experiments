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
from .default_tools import RETURNN_RC_ROOT, RASR_BINARY_PATH_APPTAINER

from .data import get_corpus_data_inputs_newcv, get_corpus_data_inputs_newcv_hdf


def run_gmm_system():
    from .gmm_baseline import run_librispeech_100_common_baseline

    system = run_librispeech_100_common_baseline(extract_additional_rasr_features=get_feature_extraction_args())
    return system


def run_gmm_system_v2():
    from .gmm_baseline import run_librispeech_100_common_baseline

    system = run_librispeech_100_common_baseline(
        extract_additional_rasr_features=get_feature_extraction_args(fix_features_output=True)
    )
    return system


def run_hybrid_baseline_pytorch():
    from i6_experiments.users.rossenbach.common_setups.rasr.pytorch_onnx_hybrid_system import PyTorchOnnxHybridSystem

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_hybrid/common_baseline_pytorch"

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
    nn_args.training_args["partition_epochs"] = {"train": 10, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    # image only, so just python3
    returnn_exe = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

    # returnn_root = tk.Path("/u/rossenbach/src/returnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")
    returnn_root_experimental = tk.Path(
        "/u/lukas.rilling/github/MiniReturnn", hash_overwrite="LIBRISPEECH_DEFAULT_MINIRETURNN_ROOT"
    )
    lbs_nn_system = PyTorchOnnxHybridSystem(
        returnn_root=returnn_root_experimental,
        returnn_python_exe=returnn_exe,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH_APPTAINER,
    )
    # manually override RASR binary for trainer
    # lbs_nn_system.crp["base"].nn_trainer_exe = RASR_BINARY_PATH_U22.join_right(f"nn-trainer.linux-x86_64-standard")
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
