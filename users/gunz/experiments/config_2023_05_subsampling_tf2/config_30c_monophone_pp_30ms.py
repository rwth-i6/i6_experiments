__all__ = ["run", "run_single"]

import copy
import dataclasses
import typing
from dataclasses import dataclass
import itertools

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------

from i6_core import rasr, returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.fh import system as fh_system
from ...setups.fh.decoder.config import PriorInfo
from ...setups.fh.network import subsampling
from ...setups.fh.network.subsampling import PoolingReduction, SelectOneReduction, TemporalReduction
from ...setups.fh.factored import PhoneticContext
from ...setups.fh.network.augment import (
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING_30MS,
    CONF_FH_DECODING_TENSOR_CONFIG,
    CONF_FOCAL_LOSS,
    RASR_ARCH,
    RASR_ROOT_NO_TF,
    RASR_ROOT_TF2,
    RETURNN_PYTHON,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_NO_TF, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH")
RASR_TF_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_TF2, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH_TF2")
RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON, hash_overwrite="RETURNN_PYTHON_EXE")

train_key = "train-other-960"


@dataclass(frozen=True)
class Experiment:
    alignment_name: str
    decode_all_corpora: bool
    init_from_system: fh_system.FactoredHybridSystem
    dc_detection: bool
    temporal_reduction_mode: TemporalReduction

    filter_segments: typing.Optional[typing.List[str]] = None
    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path, init_from_system: fh_system.FactoredHybridSystem):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    configs = [
        Experiment(
            alignment_name="scratch",
            dc_detection=False,
            decode_all_corpora=False,
            init_from_system=init_from_system,
            temporal_reduction_mode=m,
        )
        for m in [
            PoolingReduction.avg(),
            SelectOneReduction(take_i=0),
            SelectOneReduction(take_i=1),
            SelectOneReduction(take_i=2),
        ]
    ]
    for exp in configs:
        run_single(
            alignment_name=exp.alignment_name,
            dc_detection=exp.dc_detection,
            init_from_system=exp.init_from_system,
            returnn_root=returnn_root,
            filter_segments=exp.filter_segments,
            temporal_reduction_mode=exp.temporal_reduction_mode,
        )


def run_single(
    *,
    alignment_name: str,
    dc_detection: bool,
    init_from_system: fh_system.FactoredHybridSystem,
    returnn_root: tk.Path,
    filter_segments: typing.Optional[typing.List[str]],
    temporal_reduction_mode: TemporalReduction,
    conf_model_dim: int = 512,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-1-pp:{temporal_reduction_mode}"
    print(f"fh {name}")

    ss_factor = 3

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()
    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True, dc_detection=dc_detection)
    data_preparation_args = gmm_setups.get_final_output(name="data_preparation")
    # *********** System Instantiation *****************
    steps = rasr_util.RasrSteps()
    steps.add_step("init", None)  # you can create the label_info and pass here
    s = fh_system.FactoredHybridSystem(
        rasr_binary_path=RASR_BINARY_PATH,
        rasr_init_args=rasr_init_args,
        returnn_root=returnn_root,
        returnn_python_exe=RETURNN_PYTHON_EXE,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    s.label_info = dataclasses.replace(s.label_info, n_states_per_phone=1)
    s.lm_gc_simple_hash = True
    s.train_key = train_key
    if filter_segments is not None:
        s.filter_segments = filter_segments
    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=False,
        input_key="data_preparation",
        chunk_size=CONF_CHUNKING_30MS,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
    )

    # ---------------------- returnn config---------------
    s.set_experiment_dict("fh", alignment_name, "mono", postfix_name=name)

    class FakeReturnnJob:
        def __init__(self, epoch: int, ckpt: returnn.Checkpoint):
            self.out_checkpoints = {epoch: ckpt}

    s.experiments["fh"]["train_job"] = FakeReturnnJob(
        600, init_from_system.experiments["fh"]["train_job"].out_checkpoints[600]
    )
    s.experiments["fh"]["priors"] = PriorInfo.from_monophone_job(
        "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/priors/post-process-10ms"
    )

    tensor_config = dataclasses.replace(
        CONF_FH_DECODING_TENSOR_CONFIG,
        in_seq_length="extern_data/placeholders/data/data_dim0_size",
        out_right_context="right__output__ss/output_batch_major",
        out_left_context="left__output__ss/output_batch_major",
        out_center_state="center__output__ss/output_batch_major",
    )

    returnn_config = subsampling.reduce_output_step_rate(
        init_from_system.experiments["fh"]["returnn_config"],
        init_from_system.label_info,
        s.label_info,
        temporal_reduction=temporal_reduction_mode,
    )
    returnn_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)
    aux_layers = [l for l in returnn_config.config["network"].keys() if l.startswith("aux")]
    for l in aux_layers:
        returnn_config.config["network"].pop(l)

    s.set_returnn_config_for_experiment("fh", copy.deepcopy(returnn_config))

    for ep, crp_k in itertools.product([600], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        # s.set_mono_priors_returnn_rasr(
        #     key="fh",
        #     epoch=ep,
        #     train_corpus_key=s.crp_names["train"],
        #     dev_corpus_key=s.crp_names["cvtrain"],
        #     smoothen=True,
        #     output_layer_name="center-output-ss",
        #     returnn_config=returnn_config,
        # )

        s.set_graph_for_experiment("fh")
        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.monophone,
            crp_corpus=crp_k,
            epoch=ep,
            gpu=False,
            tensor_map=tensor_config,
            set_batch_major_for_feature_scorer=True,
            lm_gc_simple_hash=True,
        )
        recog_args = recog_args.with_lm_scale(round(recog_args.lm_scale / float(ss_factor), 2)).with_prior_scale(0.6)

        # Top 3 from monophone TDP study
        good_values = [
            (0.4, (3, 0, "infinity", 0), (3, 10, "infinity", 10)),  # 8,9%
            (0.6, (3, 0, "infinity", 3), (3, 10, "infinity", 10)),  # 8,9%
            (0.2, (3, 0, "infinity", 0), (10, 10, "infinity", 10)),  # 9,0%
        ]

        for cfg in [
            recog_args.with_tdp_scale(0.1),
            recog_args.with_tdp_scale(0.2),
            recog_args.with_tdp_scale(0.4),
            *(
                recog_args.with_tdp_scale(sc).with_tdp_speech(tdp_sp).with_tdp_silence(tdp_sil)
                for sc, tdp_sp, tdp_sil in good_values
            ),
        ]:
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=cfg,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
            )

    return s
