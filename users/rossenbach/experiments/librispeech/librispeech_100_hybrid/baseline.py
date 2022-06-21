import copy
from sisyphus import tk, gs

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.tools.compile import MakeJob

from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem

from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_gmm.system_collection import (
    get_system_and_steps,
)

from .data import get_corpus_data_inputs
from .baseline_args import get_nn_args


def run_hybrid_baseline():

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'experiments/librispeech/librispeech_100_hybrid/common_baseline'


    rasr_repo = CloneGitRepositoryJob("https://github.com/rwth-i6/rasr", commit="d7eaf1a37f4dcb18bc3a78295eb4b1d72a0eccfb").out_repository
    rasr_binaries = MakeJob(
        folder=rasr_repo,
        make_sequence=["build", "install"],
        configure_opts=["--i6"],
        num_processes=8,
        output_folder_name=None,
        link_outputs={'binaries': "arch/linux-x86_64-standard/"},
    ).out_links['binaries']

    tk.register_output("rasr_binaries", rasr_binaries)

    gmm_system, steps = get_system_and_steps("common_baseline")
    hybrid_init_args = copy.deepcopy(gmm_system.hybrid_init_args)

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs(gmm_system)

    nn_args = get_nn_args()
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)
    nn2_args = get_nn_args()
    nn2_args.recognition_args["dev-other"]["search_parameters"]["beam-pruning"] = 16.0
    nn_steps.add_step("nn2", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="20ff1f8fd866ec680854dc6563ff348c9c647674").out_repository
    blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so", hash_overwrite="MKL_BLA")


    lbs_nn_system = HybridSystem(returnn_root=returnn_root, returnn_python_exe=returnn_exe, blas_lib=blas_lib)
    lbs_nn_system.crp['base'].set_executables(rasr_binaries)
    lbs_nn_system.init_system(
        hybrid_init_args=hybrid_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-clean-100.train", "train-clean-100.cv"])],
    )
    #lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""
