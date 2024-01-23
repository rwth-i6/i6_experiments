import copy
from sisyphus import gs, tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.features import FilterbankJob
from i6_core.features import filter_width_from_channels

from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.baselines.tedlium2.default_tools import RASR_BINARY_PATH

from i6_experiments.users.jxu.experiments.hybrid.tedlium.data import get_corpus_data_inputs
from i6_experiments.users.jxu.experiments.hybrid.tedlium.config.torch_conformer_baseline_args import get_pytorch_nn_args
from i6_experiments.users.rossenbach.common_setups.rasr.pytorch_onnx_hybrid_system import PyTorchOnnxHybridSystem
from i6_core.features import filter_width_from_channels


def get_log_mel_feature_extraction_args():
    return {
        "fb": {
            "filterbank_options": {
                "warping_function": "mel",
                "filter_width": filter_width_from_channels(channels=80, warping_function="mel", f_max=8000),
                "normalize": True,
                "normalization_options": None,
                "without_samples": False,
                "samples_options": {
                    "audio_format": "wav",
                    "dc_detection": False,
                },
                "fft_options": None,
                "add_features_output": True,
                "apply_log": True,
                "add_epsilon": True,
            }
        }
    }

def run_gmm_system():
    from i6_experiments.common.baselines.tedlium2.gmm.baseline_config import (
        run_tedlium2_common_baseline,
    )

    system = run_tedlium2_common_baseline()
    return system


def run_tedlium2_torch_conformer_epochs_260():
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
    ) = get_corpus_data_inputs(gmm_system, rasr_init_args.feature_extraction_args['fb'], FilterbankJob, alias_prefix=prefix)
    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    gmm_system.run(steps)
    nn_args = get_pytorch_nn_args(num_epochs=260)

    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # image only, so just python3
    returnn_exe = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    blas_lib = tk.Path(
        "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so",
        hash_overwrite="TF23_MKL_BLAS",
    )

    # returnn_root = CloneGitRepositoryJob(
    #     "https://github.com/rwth-i6/returnn",
    #     commit="1dcca35aa9745e9d3d5b5b08436787bcfcc9ccf9",
    # ).out_repository
    # returnn_root.hash_overwrite="LIBRISPEECH_DEFAULT_NORETURNN_ROOT"

    returnn_root = tk.Path("/u/jxu/setups/librispeech-100/2023-06-10--torch-model/tools/20230620-returnn",
            hash_overwrite="LIBRISPEECH_DEFAULT_NORETURNN_ROOT")

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