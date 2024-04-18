import copy
from sisyphus import gs, tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.features import FilterbankJob

from i6_experiments.users.hilmes.common.setups.rasr.util import RasrSteps

from i6_experiments.users.hilmes.common.tedlium2.hybrid.data import get_corpus_data_inputs
from i6_experiments.users.hilmes.common.tedlium2.hybrid.baseline_args import get_log_mel_feature_extraction_args
from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.torch_baselines.whisper_args_v2 import get_nn_args
from i6_experiments.users.hilmes.modules.pytorch_onnx_hybrid_system import PyTorchOnnxHybridSystem


def run_gmm_system():
    from i6_experiments.users.hilmes.common.tedlium2.gmm.baseline_config import (
        run_tedlium2_common_baseline,
    )

    system = run_tedlium2_common_baseline()
    return system


def run_tedlium2_torch_whisper():
    prefix = "experiments/tedlium2/hybrid/whisper_v2"
    gs.ALIAS_AND_OUTPUT_SUBDIR = prefix

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
        gmm_system, rasr_init_args.feature_extraction_args["fb"], FilterbankJob, alias_prefix=prefix
    )
    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    gmm_system.run(steps)
    nn_args = get_nn_args(num_epochs=250)

    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # image only, so just python3
    returnn_exe = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    # blas_lib = tk.Path(
    #     "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so",
    #     hash_overwrite="TF23_MKL_BLAS",
    # )
    rasr_binary = tk.Path(
        "/work/asr4/hilmes/dev/rasr_onnx_115_24_01_24/arch/linux-x86_64-standard")
    rasr_binary.hash_overwrite = "TEDLIUM2_DEFAULT_RASR_BINARY_PATH"

    # rasr_binary = tk.Path(
    #     "/u/hilmes/dev/rasr/arch/linux-x86_64-standard")
    # rasr_binary.hash_overwrite = "TEDLIUM2_DEFAULT_RASR_BINARY_PATH"

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="bd7cbbc5cb9efa018547e3179795b0416b61c236",
    ).out_repository.copy()
    returnn_root.hash_overwrite = "TEDLIUM_RETURNN_UPDATEABLE_COMMIT"

    tedlium_nn_system = PyTorchOnnxHybridSystem(
        returnn_root=returnn_root,
        returnn_python_exe=returnn_exe,
        blas_lib=None,
        rasr_arch="linux-x86_64-standard",
        rasr_binary_path=rasr_binary,
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

