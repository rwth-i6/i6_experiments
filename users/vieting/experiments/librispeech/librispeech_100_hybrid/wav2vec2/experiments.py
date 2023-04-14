import copy
import numpy as np
from sisyphus import tk, gs

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.features.common import samples_flow
from i6_experiments.common.setups.rasr.util import RasrSteps, OggZipHdfDataInput
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.common.baselines.librispeech.default_tools import RASR_BINARY_PATH
from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict

from i6_experiments.users.vieting.experiments.librispeech.librispeech_100_attention.stoch_feat.pipeline import (
    build_training_datasets, build_test_dataset, training, search, search_single, get_average_checkpoint_v2
)
from i6_experiments.users.vieting.tools.conda import InstallMinicondaJob, CreateCondaEnvJob
from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.data import get_corpus_data_inputs
from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.configs.legacy_baseline import get_nn_args, get_feature_extraction_args
from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.configs.rc_baseline import get_nn_args as get_rc_nn_args
from .configs.config_01_baseline import get_nn_args as get_pytorch_nn_args
from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.default_tools import RETURNN_RC_ROOT, RASR_BINARY_PATH_U22, RASR_BINARY_PATH_APPTEK_APPTAINER

from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.data import  get_corpus_data_inputs_newcv, get_corpus_data_inputs_newcv_hdf


def run_gmm_system():
    from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.gmm_baseline import run_librispeech_100_common_baseline
    flow = samples_flow(dc_detection=False, input_options={"block-size": "1"}, scale_input=2**-15)
    system = run_librispeech_100_common_baseline(
        extract_additional_rasr_features={"samples": {"feature_flow": flow}}
    )
    return system


def get_ls100_oggzip_hdf_data():
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn", commit="45fad83c785a45fa4abfeebfed2e731dd96f960c").out_repository
    returnn_root.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

    gmm_system = run_gmm_system()
    from i6_core.returnn.hdf import RasrAlignmentDumpHDFJob
    from i6_core.lexicon.allophones import DumpStateTyingJob
    state_tying = DumpStateTyingJob(gmm_system.outputs["train-clean-100"]["final"].crp)
    train_align_job = RasrAlignmentDumpHDFJob(
        alignment_caches=gmm_system.outputs["train-clean-100"]["final"].alignments.hidden_paths,  # TODO: needs to be list
        state_tying_file=state_tying.out_state_tying,
        allophone_file=gmm_system.outputs["train-clean-100"]["final"].crp.acoustic_model_post_config.allophones.add_from_file,
        data_type=np.int16,
        returnn_root=returnn_root,
    )
    # import ipdb
    # ipdb.set_trace()

    ogg_zip_dict = get_ogg_zip_dict(returnn_python_exe=returnn_exe, returnn_root=returnn_root)
    ogg_zip_base_args = dict(
        alignments=train_align_job.out_hdf_files,
        audio={"features": "raw", "peak_normalization": True, "preemphasis": None},
        meta_args={
            "data_map": {"classes": ("hdf", "data"), "data": ("ogg", "data")},
            "context_window": {"classes": 1, "data": 400},
        },
        ogg_args={"targets": None},
        acoustic_mixtures=gmm_system.outputs["train-clean-100"]["final"].acoustic_mixtures,
    )
    nn_data_inputs = {}
    nn_data_inputs["train"] = OggZipHdfDataInput(
        oggzip_files=[ogg_zip_dict["train-clean-100"]],
        partition_epoch=3,
        **ogg_zip_base_args,
    )
    nn_data_inputs["cv"] = OggZipHdfDataInput(
        oggzip_files=[ogg_zip_dict["dev-clean"]],
        seq_ordering="sorted_reverse",
        **ogg_zip_base_args,
    )
    nn_data_inputs["devtrain"] = OggZipHdfDataInput(
        oggzip_files=[ogg_zip_dict["dev-clean"]],
        seq_ordering="sorted_reverse",
        **ogg_zip_base_args,
    )
    return nn_data_inputs


def run_hybrid_baseline_pytorch():
    from i6_experiments.users.rossenbach.common_setups.rasr.pytorch_onnx_hybrid_system import PyTorchOnnxHybridSystem
    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_hybrid/common_baseline_pytorch'

    gmm_system = run_gmm_system()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_feature_extraction_args()

    nn_args = get_pytorch_nn_args(num_epochs=10, debug=True, use_rasr_returnn_training=False)
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
    lbs_nn_system = PyTorchOnnxHybridSystem(
        returnn_root=returnn_root, returnn_python_exe=returnn_exe, blas_lib=blas_lib, rasr_arch="linux-x86_64-standard",
        rasr_binary_path=RASR_BINARY_PATH_APPTEK_APPTAINER)

    data = get_ls100_oggzip_hdf_data()
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
