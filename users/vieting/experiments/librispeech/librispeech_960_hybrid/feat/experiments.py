import copy
import numpy as np
from typing import Any, Dict, Optional
from sisyphus import gs, tk
from i6_core.features.common import samples_flow
from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from ..wav2vec2.data import get_ls100_oggzip_hdf_data_split_train_cv
from .baseline_args import get_nn_args as get_nn_args_baseline
from .default_tools import RETURNN_ROOT, RETURNN_EXE, RASR_BINARY_PATH
from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.data import get_corpus_data_inputs
from .gmm_baseline import run_librispeech_100_common_baseline
from i6_core.returnn.config import CodeWrapper

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

    data = get_librispeech_data(returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    #data = get_ls100_oggzip_hdf_data_split_train_cv(gmm_system, sync_ogg=True, context_window=context_window)
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs(gmm_system)
    if train_seq_ordering:
        nn_train_data_inputs["librispeech.train"].seq_ordering = train_seq_ordering
    if audio_opts:
        nn_train_data_inputs["librispeech.train"].audio = audio_opts
        nn_cv_data_inputs["librispeech.cv"].audio = audio_opts
        nn_devtrain_data_inputs["librispeech.devtrain"].audio = audio_opts
    #returnn_root = tk.Path("/u/vieting/testing/returnn", hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT")
    hybrid_nn_system = HybridSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        rasr_binary_path=RASR_BINARY_PATH,
    )
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


def run_baseline_mel_legacy():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/hybrid/feat/"
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
                feature_args=log_mel_args_8khz,
                returnn_args={
                    "batch_size": 5000,
                    "extra_args": {
                        "chunking": (
                            {"classes": 250, "data": 40000},
                            {"classes": 500, "data": 80000},
                        ),
                    },
                },
            ),
        },
        num_epochs=250,
        prefix="bs5k_",
        datasets=hybrid_nn_system.datasets,
        evaluation_epochs=[64, 128, 200, 230, 240, 250],
    )
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)
    hybrid_nn_system.run(nn_steps)
    for train_job in hybrid_nn_system.jobs["train-clean-100.train_train-clean-100.cv"].values():
        # noinspection PyUnresolvedReferences
        train_job.rqmt.update({"gpu_mem": 24, "mem": 10})


def run_baseline_mel():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/hybrid/feat/"
    log_mel_args = {
        "class": "LogMelNetwork",
        "wavenorm": True,
        "frame_size": 400,
        "frame_shift": 160,
        "fft_size": 512,
    }    
    hybrid_nn_system = get_hybrid_nn_system(context_window=241)
    nn_args = get_nn_args_baseline(
        nn_base_args={
            "v0_2x5k": dict(
                feature_args=log_mel_args,
                returnn_args={
                "batch_size": 5000,
                "extra_args": {
                    "accum_grad_multiple_step": 2,
                    },
                },
            ),
            "v2_2x5k_experiment": dict(
                feature_args=log_mel_args,
                returnn_args={
                "batch_size": 5000,
                "extra_args": {
                    "accum_grad_multiple_step": 2,
                    "optimizer": {"class": "nadam", "epsilon": 1e-8},
                    "gradient_noise": 0.3,
                    "learning_rates": list(np.linspace(2.5e-5, 3e-4, 50)) + list(np.linspace(3e-4, 2.5e-5, 50)),
                    "learning_rate_control": "newbob_multi_epoch",
                    "learning_rate_control_min_num_epochs_per_new_lr": 3,
                    "learning_rate_control_relative_error_relative_lr": True,
                    "newbob_learning_rate_decay": 0.707,
                    "newbob_multi_num_epochs": 40,
                    "newbob_multi_update_interval": 1,
                    },
                },
            ),
            # "v1": dict(
            #     feature_args=log_mel_args,
            #     returnn_args={
            #         "batch_size": 5000,
            #         "extra_args": {
            #             "chunking": (
            #                 {"classes": 250, "data": 250 * 160},
            #                 {"classes": 200, "data": 200 * 160},
            #             ),
            #             "gradient_clip": 1.0,
            #             "gradient_noise": 0.3,
            #             "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
            #                 np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #             "optimizer": {"class": "adam", "epsilon": 1e-08},
            #         },
            #         "conformer_args": {
            #             "encoder_layers": 12,
            #             "conv_filter_size": (31,),
            #         },
            #     },
            # ),
            # "v1_nogradnoise": dict(
            #     feature_args=log_mel_args,
            #     returnn_args={
            #         "batch_size": 5000,
            #         "extra_args": {
            #             "chunking": (
            #                 {"classes": 250, "data": 250 * 160},
            #                 {"classes": 200, "data": 200 * 160},
            #             ),
            #             "gradient_clip": 1.0,
            #             "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
            #                 np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #             "optimizer": {"class": "adam", "epsilon": 1e-08},
            #         },
            #         "conformer_args": {
            #             "encoder_layers": 12,
            #             "conv_filter_size": (31,),
            #         },
            #     },
            # ),
            # "v1_nogradnoise_nadam": dict(
            #     feature_args=log_mel_args,
            #     returnn_args={
            #         "batch_size": 5000,
            #         "extra_args": {
            #             "chunking": (
            #                 {"classes": 250, "data": 250 * 160},
            #                 {"classes": 200, "data": 200 * 160},
            #             ),
            #             "gradient_clip": 1.0,
            #             "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
            #                 np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #             "optimizer": {"class": "nadam", "epsilon": 1e-08},
            #         },
            #         "conformer_args": {
            #             "encoder_layers": 12,
            #             "conv_filter_size": (31,),
            #         },
            #     },
            # ),
        },
        num_epochs=250,
        prefix="lm80_bs5k_",
        datasets=hybrid_nn_system.datasets,
        evaluation_epochs=[64, 128, 200, 230, 240, 250],
    )
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)
    hybrid_nn_system.run(nn_steps)
    for train_job in hybrid_nn_system.jobs["train-clean-100.train_train-clean-100.cv"].values():
        # noinspection PyUnresolvedReferences
        train_job.rqmt.update({"gpu_mem": 11, "mem": 10})


def run_baseline_scf():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/hybrid/feat/"
    scf_args = {"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2}   
    hybrid_nn_system = get_hybrid_nn_system(context_window=441)
    nn_args = get_nn_args_baseline(
        nn_base_args={
            "scf": dict(
                returnn_args={
                    "batch_size": 5000,
                    "extra_args": {
                        "accum_grad_multiple_step": 2,
                        },
                    },
                feature_args=scf_args,
            ),
            "scf_v2": dict(
                returnn_args={
                    "batch_size": 5000,
                    "extra_args": {
                        "accum_grad_multiple_step": 2,
                        "optimizer": {"class": "nadam", "epsilon": 1e-8},
                        "gradient_noise": 0.3,
                        "learning_rates": list(np.linspace(2.5e-5, 3e-4, 50)) + list(np.linspace(3e-4, 2.5e-5, 50)),
                        "learning_rate_control": "newbob_multi_epoch",
                        "learning_rate_control_min_num_epochs_per_new_lr": 3,
                        "learning_rate_control_relative_error_relative_lr": True,
                        "newbob_learning_rate_decay": 0.707,
                        "newbob_multi_num_epochs": 40,
                        "newbob_multi_update_interval": 1,
                        },
                    # Does not fit on 11GB GPU. Needs 24GB. Maybe try later if baseline stays bad.
                    "conformer_args": {
                        "encoder_layers": 12,
                        "conv_filter_size": (31,),
                    },
                },
                feature_args=scf_args,
            ),
            "scf_v3_experiment": dict(
                feature_args=scf_args,
                returnn_args={
                "batch_size": 5000,
                "extra_args": {
                    "accum_grad_multiple_step": 2,
                    "optimizer": {"class": "nadam", "epsilon": 1e-8},
                    "gradient_noise": 0.3,
                    "learning_rates": list(np.linspace(2.5e-5, 3e-4, 50)) + list(np.linspace(3e-4, 2.5e-5, 50)),
                    "learning_rate_control": "newbob_multi_epoch",
                    "learning_rate_control_min_num_epochs_per_new_lr": 3,
                    "learning_rate_control_relative_error_relative_lr": True,
                    "newbob_learning_rate_decay": 0.707,
                    "newbob_multi_num_epochs": 40,
                    "newbob_multi_update_interval": 1,
                    },
                },
            ),
        },
        num_epochs=260,
        prefix="bs2x5k_",
        datasets=hybrid_nn_system.datasets,
        evaluation_epochs=[128, 200, 230, 240, 250, 260],
    )
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)
    hybrid_nn_system.run(nn_steps)
    for train_job in hybrid_nn_system.jobs["train-clean-100.train_train-clean-100.cv"].values():
        # noinspection PyUnresolvedReferences
        train_job.rqmt.update({"gpu_mem": 11, "mem": 10})


def run_scf_perturbation():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/hybrid/feat/"

    scf_args = {"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2}   

    # second run
   # perturbation_args = [
    #    {"codecs": [{"encoding": "ULAW","prob": 0.3}]},
     #   {"codecs": [{"encoding": "ULAW","prob": 0.5}]},
      #  {"codecs": [{"encoding": "ULAW","prob": 0.7}]},
       # {"tempo": {"prob": 0.3, "minimum": 0.83, "maximum": 1.17}},
     #   {"tempo": {"prob": 0.8, "minimum": 0.83, "maximum": 1.17}},
      #  {"tempo": {"prob": 0.3, "minimum": 0.7, "maximum": 1.3}},
       # {"tempo": {"prob": 0.8, "minimum": 0.7, "maximum": 1.3}},
       # {"speed": {"prob": 0.5, "minimum": 0.88, "maximum": 1.12}},
     #   {"speed": {"prob": 0.3, "minimum": 0.88, "maximum": 1.12}},
      #  {"speed": {"prob": 0.7, "minimum": 0.88, "maximum": 1.12}},
       # {"speed": {"prob": 0.6, "minimum": 0.8, "maximum": 1.2}},
        #{"speed": {"prob": 0.6, "minimum": 0.9, "maximum": 1.1}},
     #   {"non_linearity": {"prob": 0.7, "minimum": 0.9, "maximum": 1.1}},
      #  {"non_linearity": {"prob": 0.3, "minimum": 0.9, "maximum": 1.1}},
   # ]


    # seperated because the training needs a different network without preemphasis
    perturbation_args = [
        {"preemphasis": {"prob": 1, "minimum": 0.9, "maximum": 1}},
        {"preemphasis": {"prob": 0.7, "minimum": 0.9, "maximum": 1}},
        {"preemphasis": {"prob": 0.3, "minimum": 0.9, "maximum": 1}},
    ]

    def process_args(args: Dict[str, Any]):
        """
        Process the argument dictionary to generate a key string and a report string.

        Returns:
            tuple: A tuple containing the key string and the report string.
        """

        key_string = ""
        report_dict = {}

        for key, value in args.items():

            if key in ["speed", "tempo", "preemphasis", "non_linearity"]:
                key_component = f"{key}_{value['prob']}_{value['minimum']}_{value['maximum']}"
                key_string += key_component
                report_dict[key] = f"{value['prob']}_{value['minimum']}_{value['maximum']}"
            elif key == "codecs":
                codecs_str = "_".join([f"{codec['encoding']}_{codec['prob']}" for codec in value])
                key_string += f"{key}_{codecs_str}_"
                report_dict[key] = codecs_str
            else:
                raise ValueError(f"Unknown argument name: {key}")

        return key_string, report_dict

    nn_base_args = {}

    returnn_args = {
        "batch_size": 5000,
        "audio_perturbation": True,
        "use_multi_proc_dataset": True,
        "pre_process": CodeWrapper("audio_perturb_runner.run")
    }

    for args in perturbation_args:
        exp_name_suffix, report_dict = process_args(args)

        # Construct the exp_name and report_args
        exp_name = f"scf_perturb_{exp_name_suffix}"
        report_args = report_dict
        nn_base_args[exp_name] = dict(
            returnn_args={
                "extra_args": {
                    "audio_perturb_args": args,
                    "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
                },
                **returnn_args,
            },
            feature_args=scf_args,
        )

    hybrid_nn_system = get_hybrid_nn_system(context_window=441)
    nn_args = get_nn_args_baseline(
        nn_base_args,
        num_epochs=260,
        prefix="bs5k_",
        datasets=hybrid_nn_system.datasets,
        evaluation_epochs=[64, 128, 200, 230, 240, 250, 260],
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
