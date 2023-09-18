__all__ = ["run", "run_single"]

import copy
import dataclasses
import random
from dataclasses import dataclass
import itertools

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------

from i6_core import rasr, returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.nn import oclr, returnn_time_tag
from ...setups.common.nn.chunking import subsample_chunking
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.decoder.config import PriorInfo
from ...setups.fh.network import conformer
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    SubsamplingInfo,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING_10MS,
    CONF_FH_DECODING_TENSOR_CONFIG,
    CONF_FOCAL_LOSS,
    CONF_LABEL_SMOOTHING,
    CONF_SA_CONFIG,
    FROM_SCRATCH_CV_INFO,
    L2,
    RASR_ARCH,
    RASR_ROOT_TF2,
    RETURNN_PYTHON,
    TEST_EPOCH,
    ZHOU_ALLOPHONES,
    ZHOU_SUBSAMPLED_ALIGNMENT,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_TF2, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH")
RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON, hash_overwrite="RETURNN_PYTHON_EXE")

train_key = "train-other-960"


@dataclass(frozen=True)
class Experiment:
    alignment: tk.Path
    alignment_name: str
    lr: str
    multitask: bool
    dc_detection: bool
    run_performance_study: bool
    subsampling_factor: int
    tune_decoding: bool

    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    scratch_align = tk.Path(ZHOU_SUBSAMPLED_ALIGNMENT, cached=True)

    configs = [
        Experiment(
            alignment=scratch_align,
            alignment_name="scratch",
            dc_detection=True,
            lr="v13",
            multitask=True,
            run_performance_study=False,
            subsampling_factor=4,
            tune_decoding=False,
        ),
    ]
    for exp in configs:
        run_single(
            alignment=exp.alignment,
            alignment_name=exp.alignment_name,
            dc_detection=exp.dc_detection,
            focal_loss=exp.focal_loss,
            lr=exp.lr,
            multitask=exp.multitask,
            returnn_root=returnn_root,
            run_performance_study=exp.run_performance_study,
            subsampling_factor=exp.subsampling_factor,
            tune_decoding=exp.tune_decoding,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    dc_detection: bool,
    focal_loss: float,
    lr: str,
    multitask: bool,
    returnn_root: tk.Path,
    run_performance_study: bool,
    subsampling_factor: int,
    tune_decoding: bool,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-1-lr:{lr}-ss:{subsampling_factor}-fs:{subsampling_factor}"
    print(f"fh {name}")

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

    s.train_key = train_key
    if alignment_name == "scratch":
        s.cv_info = FROM_SCRATCH_CV_INFO
    s.base_allophones = ZHOU_ALLOPHONES
    s.label_info = dataclasses.replace(s.label_info, n_states_per_phone=1)

    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    s.alignments[train_key] = alignment
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=alignment_name == "scratch",
        input_key="data_preparation",
        chunk_size=CONF_CHUNKING_10MS,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
        add_base_allophones=True,
    )

    # ---------------------- returnn config---------------
    partition_epochs = {"train": 40, "dev": 1}

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
    network_builder = conformer.get_best_model_config(
        conf_model_dim,
        chunking=CONF_CHUNKING_10MS,
        focal_loss_factor=CONF_FOCAL_LOSS,
        label_smoothing=CONF_LABEL_SMOOTHING,
        num_classes=s.label_info.get_n_of_dense_classes(),
        time_tag_name=time_tag_name,
        upsample_by_transposed_conv=False,
        feature_stacking_size=subsampling_factor,
    )
    network = network_builder.network
    network = augment_net_with_label_pops(
        network,
        label_info=s.label_info,
        classes_subsampling_info=SubsamplingInfo(factor=subsampling_factor, time_tag_name=time_tag_name),
    )
    network = augment_net_with_monophone_outputs(
        network,
        add_mlps=True,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        use_multi_task=multitask,
    )
    network = aux_loss.add_intermediate_loss(
        network,
        center_state_only=True,
        context=PhoneticContext.monophone,
        encoder_output_len=conf_model_dim,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        time_tag_name=time_tag_name,
        upsampling=False,
    )

    base_config = {
        **s.initial_nn_args,
        **oclr.get_oclr_config(num_epochs=num_epochs, schedule=lr),
        **CONF_SA_CONFIG,
        "batch_size": 11000 if lr == "v7" else 6144,
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "chunking": subsample_chunking(CONF_CHUNKING_10MS, subsampling_factor),
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.0,
        "network": network,
        "extern_data": {
            "data": {"dim": 50, "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)}},
            **extern_data.get_extern_data_config(label_info=s.label_info, time_tag_name=None),
        },
    }
    keep_epochs = [TEST_EPOCH, 550, num_epochs]
    base_post_config = {
        "cleanup_old_models": {
            "keep_best_n": 3,
            "keep": keep_epochs,
        },
    }
    returnn_config = returnn.ReturnnConfig(
        config=base_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog={
            "numpy": "import numpy as np",
            "time": time_prolog,
        },
        python_epilog={
            "functions": [
                sa_mask,
                sa_random_mask,
                sa_summary,
                sa_transform,
            ],
        },
    )

    s.set_experiment_dict("fh", alignment_name, "mono", postfix_name=name)
    s.set_returnn_config_for_experiment("fh", remove_label_pops_and_losses_from_returnn_config(returnn_config))

    class FakeReturnnJob:
        def __init__(self, epoch: int, ckpt: returnn.Checkpoint):
            self.out_checkpoints = {epoch: ckpt}

    import_checkpoint = returnn.Checkpoint(
        index_path=tk.Path(
            "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling/train/nn_mono-from-scratch-conf-lr:v7-ss:4/output/models/epoch.600.index"
        )
    )
    s.experiments["fh"] = {
        **s.experiments["fh"],
        "train_job": FakeReturnnJob(600, import_checkpoint),
        "priors": PriorInfo.from_monophone_job(
            "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling/priors/nn_mono-from-scratch-conf-lr-v7-ss-4"
        ),
    }

    decoding_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)
    decoding_config.config["network"]["center-output"] = {
        **decoding_config.config["network"]["center-output"],
        "class": "linear",
        "activation": "log_softmax",
        "register_as_extern_data": "center-output",
    }
    s.set_graph_for_experiment("fh", decoding_config)

    s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.monophone)
    s._update_crp_am_setting(crp_key="dev-other", tdp_type="default", add_base_allophones=False)

    tying_cfg = rasr.RasrConfig()
    tying_cfg.type = str(RasrStateTying.monophone)

    for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-other"]):
        recog_args = dataclasses.replace(
            s.get_cart_params("fh"),
            beam=22,
            beam_limit=500_000,
            lm_scale=1.0,
            tdp_scale=0.1,
        )

        permutation = itertools.product(
            np.linspace(0.5, 1.0, 3),
            [0.0, 3.0, 10.0],
            [0.0, 3.0, 10.0],
            [10.0, 20.0, 40.0],
            [3.0, 10.0],
            [0.0],
            [0.0],
            [True],
        )
        all_cfgs = (
            *permutation,
            (0.3, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, True),  # default cfg w/ other prior scales
            (0.5, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, True),  # default cfg w/ other prior scales
            (0.5, 10.0, 10.0, 10.0, 10.0, 0.0, 3.0, False),  # try speech exit penalty to make model stick to words
            (0.5, 10.0, 10.0, 10.0, 10.0, 0.0, 10.0, False),  # try speech exit penalty to make model stick to words
            (0.5, 10.0, 15.0, 10.0, 10.0, 0.0, 3.0, False),  # speech exit penalty + even higher sil loop
            (0.5, 15.0, 10.0, 10.0, 10.0, 0.0, 0.0, False),  # even higher silence loop
        )
        for pC, sil_loop, sil_fwd, sil_exit, sp_loop, sp_fwd, sp_exit, gpu in all_cfgs:
            sil_non_w_tdp = (sil_loop, sil_fwd, "infinity", sil_exit)
            cfg = dataclasses.replace(
                recog_args,
                tdp_non_word=sil_non_w_tdp,
                tdp_silence=sil_non_w_tdp,
                tdp_speech=(sp_loop, sp_fwd, "infinity", sp_exit),
            ).with_prior_scale(pC)

            s.recognize_cart(
                cart_tree_or_tying_config=tying_cfg,
                crp_corpus=crp_k,
                encoder_output_layer="center__output",
                epoch=ep,
                gpu=gpu,
                key="fh",
                lm_gc_simple_hash=True,
                log_softmax_returnn_config=decoding_config,
                n_cart_out=s.label_info.get_n_of_dense_classes(),
                opt_lm_am_scale=False,
                parallel=20 if gpu else None,
                params=cfg,
                rtf=16,
            )

        cfg_w_phoneme_errors = [(0.5, 10.0, 10.0, 10.0, 10.0, 0.0, 10.0)]
        for pC, sil_loop, sil_fwd, sil_exit, sp_loop, sp_fwd, sp_exit in cfg_w_phoneme_errors:
            sil_non_w_tdp = (sil_loop, sil_fwd, "infinity", sil_exit)
            cfg = dataclasses.replace(
                recog_args,
                tdp_non_word=sil_non_w_tdp,
                tdp_silence=sil_non_w_tdp,
                tdp_speech=(sp_loop, sp_fwd, "infinity", sp_exit),
            ).with_prior_scale(pC)

            extra_config = rasr.RasrConfig()
            extra_config.flf_lattice_tool.network.evaluator.phoneme_errors = True

            s.recognize_cart(
                cart_tree_or_tying_config=tying_cfg,
                crp_corpus=crp_k,
                encoder_output_layer="center__output",
                epoch=ep,
                gpu=False,
                key="fh",
                lm_gc_simple_hash=True,
                log_softmax_returnn_config=decoding_config,
                n_cart_out=s.label_info.get_n_of_dense_classes(),
                opt_lm_am_scale=False,
                params=cfg,
                rtf=1,
                adv_search_extra_config=extra_config,
                alias_output_prefix="ph_",
            )

    s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.triphone, n_states_per_phone=2)
    s._update_crp_am_setting(crp_key="dev-other", tdp_type="default", add_base_allophones=False)

    recognizer, recog_args = s.get_recognizer_and_args(
        key="fh",
        context_type=PhoneticContext.monophone,
        crp_corpus="dev-other",
        epoch=max(keep_epochs),
        gpu=False,
        tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
        set_batch_major_for_feature_scorer=True,
        lm_gc_simple_hash=True,
    )
    recog_args = recog_args.with_lm_scale(1.0).with_tdp_scale(0.1)
    recognizer.recognize_count_lm(
        label_info=s.label_info,
        search_parameters=recog_args,
        num_encoder_output=conf_model_dim,
        rerun_after_opt_lm=True,
        calculate_stats=True,
        is_min_duration=True,
        pre_path="decoding-min-dur",
    )

    return s
