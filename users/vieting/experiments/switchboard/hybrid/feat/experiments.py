import copy
from typing import Any, Dict, Optional
from sisyphus import gs

from i6_core.features.common import samples_flow
from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem

from .data import get_corpus_data_inputs_oggzip
from .baseline_args import get_nn_args as get_nn_args_baseline
from .default_tools import RASR_BINARY_PATH, RETURNN_ROOT, RETURNN_EXE


def run_gmm_system_from_common():
    from ...gmm.baseline.baseline_config import run_switchboard_baseline_ldc_v5

    flow = samples_flow(dc_detection=False, input_options={"block-size": "1"}, scale_input=2**-15)
    system = run_switchboard_baseline_ldc_v5(recognition=True)
    system.extract_features(
        feat_args={"samples": {"feature_flow": flow}},
        corpus_list=system.dev_corpora + system.test_corpora,
    )
    return system


def get_hybrid_nn_system(
        context_window: int,
        train_seq_ordering: Optional[str] = None,
        audio_opts: Optional[Dict[str, Any]] = None,
):
    gmm_system = run_gmm_system_from_common()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)

    # noinspection PyTypeChecker
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_oggzip(
        gmm_system,
        partition_epoch={"train": 6, "dev": 1},
        context_window={"classes": 1, "data": context_window},
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
    )
    if train_seq_ordering:
        nn_train_data_inputs["switchboard.train"].seq_ordering = train_seq_ordering
    if audio_opts:
        nn_train_data_inputs["switchboard.train"].audio = audio_opts
        nn_cv_data_inputs["switchboard.cv"].audio = audio_opts
        nn_devtrain_data_inputs["switchboard.devtrain"].audio = audio_opts

    hybrid_nn_system = HybridSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        rasr_binary_path=RASR_BINARY_PATH,
    )
    hybrid_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["switchboard.train", "switchboard.cv"])],
    )

    return hybrid_nn_system


def run_baseline_gt():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/hybrid/feat/"

    nn_args = get_nn_args_baseline(
        nn_base_args={
            # comment out because hash changed because freq_max and freq_min are added for GT
            # "gt40_oldFreqMax": dict(
            #     returnn_args=dict(batch_size=14000),
            #     feature_args={"class": "GammatoneNetwork", "sample_rate": 8000, "output_dim": 40},
            # ),
            # "gt40_oldFreqMax_win640": dict(
            #     returnn_args=dict(batch_size=14000),
            #     feature_args={
            #         "class": "GammatoneNetwork", "sample_rate": 8000, "output_dim": 40, "gt_filterbank_size": 0.08,
            #         "temporal_integration_size": 0.05,
            #     },
            # ),
            # "gt50_oldFreqMax": dict(
            #     returnn_args=dict(batch_size=14000),
            #     feature_args={"class": "GammatoneNetwork", "sample_rate": 8000, "output_dim": 50},
            # ),
            # "gt40_oldFreqMax_minchunk2": dict(
            #     returnn_args=dict(batch_size=14000, extra_args=dict(min_chunk_size={"classes": 2, "data": 160})),
            #     feature_args={"class": "GammatoneNetwork", "sample_rate": 8000, "output_dim": 40},
            # ),
            "gt40_pe": dict(
                returnn_args=dict(batch_size=14000),
                feature_args={
                    "class": "GammatoneNetwork", "sample_rate": 8000, "freq_max": 3800., "output_dim": 40,
                    "preemphasis": 1.0,
                },
            ),
        },
        num_epochs=260,
        prefix="conformer_bs14k_"
    )
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)
    hybrid_nn_system = get_hybrid_nn_system(context_window=441)
    hybrid_nn_system.run(nn_steps)

    # disable peak normalization, add wave norm
    audio_opts = {"features": "raw", "sample_rate": 8000}
    nn_args = get_nn_args_baseline(
        nn_base_args={
            "gt40_pe_wavenorm": dict(
                returnn_args=dict(batch_size=14000),
                feature_args={
                    "class": "GammatoneNetwork", "sample_rate": 8000, "freq_max": 3800., "output_dim": 40,
                    "preemphasis": 1.0, "wave_norm": True,
                },
            ),
        },
        num_epochs=260,
        prefix="conformer_bs14k_"
    )
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    hybrid_nn_system = get_hybrid_nn_system(context_window=441, audio_opts=audio_opts)
    hybrid_nn_system.run(nn_steps)
    for train_job in hybrid_nn_system.jobs["switchboard.train_switchboard.cv"].values():
        # noinspection PyUnresolvedReferences
        train_job.rqmt.update({"gpu_mem": 24, "mem": 10})


def run_baseline_mel():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/hybrid/feat/"

    nn_args = get_nn_args_baseline(
        nn_base_args={
            "lm80_fft256": dict(
                returnn_args=dict(batch_size=14000),
                feature_args={
                    "class": "LogMelNetwork", "wavenorm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256
                },
            ),
            "lm80_fft256_lr8e-4": dict(
                returnn_args=dict(batch_size=14000),
                feature_args={
                    "class": "LogMelNetwork", "wavenorm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256
                },
                peak_lr=8e-4,
            ),
            "lm80_fft256_lr9e-4": dict(
                returnn_args=dict(batch_size=14000),
                feature_args={
                    "class": "LogMelNetwork", "wavenorm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256
                },
                peak_lr=9e-4,
            ),
            "lm80_fft256_lr15e-4": dict(
                returnn_args=dict(batch_size=14000),
                feature_args={
                    "class": "LogMelNetwork", "wavenorm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256
                },
                peak_lr=15e-4,
            ),
            "lm80_fft512": dict(
                returnn_args=dict(batch_size=14000),
                feature_args={"class": "LogMelNetwork", "wavenorm": True, "frame_shift": 80},
            ),
        },
        num_epochs=260,
        prefix="conformer_bs14k_"
    )
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    hybrid_nn_system = get_hybrid_nn_system(context_window=441)
    hybrid_nn_system.run(nn_steps)
    for train_job in hybrid_nn_system.jobs["switchboard.train_switchboard.cv"].values():
        # noinspection PyUnresolvedReferences
        train_job.rqmt.update({"gpu_mem": 24, "mem": 10})


def run_baseline_scf():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/hybrid/feat/"

    nn_args = get_nn_args_baseline(
        nn_base_args={
            "scf": dict(
                returnn_args=dict(batch_size=14000),
                feature_args={"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2}
            )
        },
        prefix="conformer_bs14k_",
        num_epochs=260,
    )
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    hybrid_nn_system = get_hybrid_nn_system(context_window=249)
    hybrid_nn_system.run(nn_steps)
    for train_job in hybrid_nn_system.jobs["switchboard.train_switchboard.cv"].values():
        # noinspection PyUnresolvedReferences
        train_job.rqmt.update({"gpu_mem": 24, "mem": 10})

def run_specaug_scf():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/hybrid/feat/"

    nn_args = get_nn_args_baseline(
        nn_base_args={
            "scf": dict(returnn_args=dict(batch_size=7000, specaug_mask_sorting=True, extra_args=dict(accum_grad_multiple_step=2)),
                feature_args={"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2}
            ),
            "scf_divisor-4": dict(
                returnn_args=dict(
                    batch_size=3500, 
                    specaug_mask_sorting=True,
                    mask_divisor=4,
                    extra_args=dict(accum_grad_multiple_step=4)
                ),
                feature_args={"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2}
            ),
            "scf_divisor-6": dict(
                returnn_args=dict(
                    batch_size=3500,
                    specaug_mask_sorting=True,
                    mask_divisor=6,
                    extra_args=dict(accum_grad_multiple_step=4)
                ),
                feature_args={"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2}
            ),
        },
        prefix="conformer_bs14k_specaug_sorted_",
        num_epochs=260,
    )
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    hybrid_nn_system = get_hybrid_nn_system(context_window=249)
    hybrid_nn_system.run(nn_steps)
    for train_job in hybrid_nn_system.jobs["switchboard.train_switchboard.cv"].values():
        # noinspection PyUnresolvedReferences
        train_job.rqmt.update({"gpu_mem": 24, "mem": 10})


def run_all():
    run_baseline_gt()
    run_baseline_mel()
    run_baseline_scf()
    run_specaug_scf()
