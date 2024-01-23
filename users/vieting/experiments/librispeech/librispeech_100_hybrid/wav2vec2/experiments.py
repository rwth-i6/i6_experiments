import copy
from sisyphus import tk, gs

from i6_core.features.common import samples_flow
from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.users.vieting.tools.conda import InstallMinicondaJob, CreateCondaEnvJob
from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.data import get_corpus_data_inputs
from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.configs.legacy_baseline import (
    get_feature_extraction_args
)
from i6_experiments.common.baselines.librispeech.ls100.gmm.baseline_config import run_librispeech_100_common_baseline
from .configs.config_01_baseline import get_nn_args as get_pytorch_nn_args
from .data import get_ls100_oggzip_hdf_data, get_ls100_oggzip_hdf_data_split_train_cv
from .default_tools import RASR_BINARY_PATH_ONNX_APPTAINER
from .onnx_precomputed_hybrid_system import OnnxPrecomputedHybridSystem


def run_gmm_system():
    system = run_librispeech_100_common_baseline(rasr_binary_path=RASR_BINARY_PATH_ONNX_APPTAINER)
    flow = samples_flow(dc_detection=False, input_options={"block-size": "1"}, scale_input=2**-15)
    system.extract_features(
        feat_args={"samples": {"feature_flow": flow, "port_name": "samples"}},
        corpus_list=system.dev_corpora + system.test_corpora,
    )
    return system


def run_hybrid_baseline_pytorch():
    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_hybrid/wav2vec2'

    gmm_system = run_gmm_system()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_feature_extraction_args()

    nn_args = get_pytorch_nn_args(evaluation_epochs=[40, 80, 160, 200], debug=True, use_rasr_returnn_training=False)
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************
    conda = InstallMinicondaJob()
    packages = {
        "numpy": "==1.23.5",
        "pytorch": "==2.0.0",
        "torchaudio": "==2.0.0",
        "torchdata": "==0.6.0",
        "pytorch-cuda": "==11.7",
        "pysoundfile": "==0.11.0",
        "matplotlib": "==3.7.1",
        "h5py": "==3.7.0",
        "typing": "==3.10.0.0",
        "black": "==22.3.0",
        "flask": "==2.2.2",
        "ipdb": "==0.13.11",
        "PyYAML": "==6.0",
        "dill": "==0.3.6",
        "bitarray": "==2.5.1",
        "cython": "==0.29.33",
        "pybind11": "==2.10.1",
        "hydra-core": "==1.3.2",
        "omegaconf": "==2.3.0",
        "sacrebleu": "==2.3.1",
        "scikit-learn": "==1.2.2",
        "tqdm": "==4.65.0",
    }
    conda_env_job = CreateCondaEnvJob(
        conda.out_conda_exe, python_version="3.10", channels=["pytorch", "nvidia", "conda-forge"], packages=packages,
    )
    conda_env_job.add_alias("tools/conda_envs/returnn_training")
    returnn_exe = conda_env_job.out_env_exe
    # returnn_exe = tk.Path("/work/asr4/vieting/programs/conda/20230126/anaconda3/envs/py310_tf210/bin/python", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    blas_lib = tk.Path(
        "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so",
        hash_overwrite="TF23_MKL_BLAS",
    )

    # returnn_root = tk.Path("/u/rossenbach/src/returnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")
    returnn_root = tk.Path("/u/vieting/testing/returnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")
    lbs_nn_system = OnnxPrecomputedHybridSystem(
        returnn_root=returnn_root, returnn_python_exe=returnn_exe, blas_lib=blas_lib, rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH_ONNX_APPTAINER)

    data = get_ls100_oggzip_hdf_data_split_train_cv(gmm_system)
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs(gmm_system)

    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data={"train-clean-100.train": data["train"]},
        cv_data={"train-clean-100.cv": data["cv"]},
        # devtrain_data={"train-clean-100.devtrain": data["devtrain"]},
        dev_data=nn_dev_data_inputs,
        # test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-clean-100.train", "train-clean-100.cv"])],
    )
    lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""
