import copy
from typing import Any, Dict, Optional
from sisyphus import gs, tk
import ipdb
from i6_core.features.common import samples_flow
from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from ..wav2vec2.data import get_ls100_oggzip_hdf_data_split_train_cv
from .baseline_args import get_nn_args as get_nn_args_baseline
from .default_tools import RETURNN_ROOT, RETURNN_EXE, RASR_BINARY_PATH
from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.data import get_corpus_data_inputs
from .gmm_baseline import run_librispeech_100_common_baseline

def run_gmm_system():
    system = run_librispeech_100_common_baseline()
    flow = samples_flow(dc_detection=False, input_options={"block-size": "1"}, scale_input=2**-15)
    system.extract_features(
        feat_args={"samples": {"feature_flow": flow, "port_name": "samples"}},
        corpus_list=system.dev_corpora + system.test_corpora,
    )
    return system

def get_hybrid_nn_system(
    context_window: int,
    train_seq_ordering: Optional[str] = None,
    audio_opts: Optional[Dict[str, Any]] = None,
):
    gmm_system = run_gmm_system()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)


    data = get_ls100_oggzip_hdf_data_split_train_cv(gmm_system)
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs(gmm_system)
    if train_seq_ordering:
        nn_train_data_inputs["librespeech.train"].seq_ordering = train_seq_ordering
    if audio_opts:
        nn_train_data_inputs["librespeech.train"].audio = audio_opts
        nn_cv_data_inputs["librespeech.cv"].audio = audio_opts
        nn_devtrain_data_inputs["librespeech.devtrain"].audio = audio_opts
    #returnn_root = tk.Path("/u/vieting/testing/returnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")
    hybrid_nn_system = HybridSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        rasr_binary_path=RASR_BINARY_PATH,
    )
    import ipdb 
    ipdb.set_trace()
    hybrid_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data={"train-clean-100.train": data["train"].get_data_dict()},
        cv_data={"train-clean-100.cv": data["cv"].get_data_dict()},
        # devtrain_data={"train-clean-100.devtrain": data["devtrain"]},
        dev_data=nn_dev_data_inputs,
        # test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-clean-100.train", "train-clean-100.cv"])],
    )
    hybrid_nn_system.datasets = data
    return hybrid_nn_system


def run_baseline_mel():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librespeech/hybrid/feat/"
    log_mel_args_8khz = {
        "class": "LogMelNetwork",
        "wavenorm": True,
        "frame_size": 200,
        "frame_shift": 80,
        "fft_size": 256,
    }    
    hybrid_nn_system = get_hybrid_nn_system(context_window=441)
    nn_args = get_nn_args_baseline(
        nn_base_args={
            "lm80_fft256": dict(
                returnn_args={
                    "batch_size": 5000,
                    },
                feature_args=log_mel_args_8khz,
            ),
        },
        num_epochs=125,
        prefix="bs5k_",
    )
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)
    hybrid_nn_system.run(nn_steps)
    for train_job in hybrid_nn_system.jobs["train-clean-100.train_train-clean-100.cv"].values():
        # noinspection PyUnresolvedReferences
        train_job.rqmt.update({"gpu_mem": 11, "mem": 10})

def py():
    """
    called if the file is passed to sis manager, used to run all experiments (replacement for main)
    """
    run_baseline_mel()