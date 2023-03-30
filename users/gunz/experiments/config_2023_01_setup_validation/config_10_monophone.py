__all__ = ["run", "run_single"]

import copy
import itertools
import typing

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common import oclr, returnn_time_tag
from ...setups.common.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.network import conformer
from ...setups.fh.factored import PhoneticContext, PhonemeStateClasses
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING,
    CONF_LABEL_SMOOTHING,
    CONF_NUM_TRAIN_EPOCHS,
    CONF_SA_CONFIG,
    CONF_SIZES,
    FH_DECODING_TENSOR_CONFIG,
    FH_LOSS_VARIANTS_MONO,
    L2,
    RAISSI_ALIGNMENT,
    RASR_ROOT_FH_GUNZ,
    RASR_ROOT_RS_RASR_GUNZ,
    RETURNN_PYTHON_TF15,
)
from .loss_scale import get_int_loss_scale

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_FH_GUNZ, "arch", gs.RASR_ARCH))
RASR_BINARY_PATH.hash_override = "FH_RASR_PATH"

RS_RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_RS_RASR_GUNZ, "arch", gs.RASR_ARCH))
RASR_BINARY_PATH.hash_override = "RS_RASR_PATH"

RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON_TF15)
RETURNN_PYTHON_EXE.hash_override = "FH_RETURNN_PYTHON_EXE"

train_key = "train-other-960"


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    align = tk.Path(RAISSI_ALIGNMENT)

    cfgs = itertools.product(
        CONF_SIZES,
        CONF_NUM_TRAIN_EPOCHS,
        FH_LOSS_VARIANTS_MONO,
        [PhonemeStateClasses.word_end],
    )
    for conf_size, num_epochs, losses, web_cls in cfgs:
        run_single(
            alignment=align,
            alignment_name="GMMtri",
            returnn_root=returnn_root,
            conf_size=conf_size,
            num_epochs=num_epochs,
            web_cls=web_cls,
            int_losses=losses,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    returnn_root: tk.Path,
    conf_size: int,
    num_epochs: int,
    web_cls: PhonemeStateClasses,
    int_losses: typing.List[typing.Tuple[int, PhoneticContext, bool]],
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    loss_str = "+".join([f"{i}{ctx.short_name()}{'_c' if c_only else ''}" for i, ctx, c_only in int_losses])
    name = f"conf-ph:1-ep:{num_epochs}-cls:{web_cls.value}-loss:{loss_str}"
    print(f"fh {name}")

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()
    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True)
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
    s.label_info.state_tying = "no-tying-dense"  # Multitasking
    s.label_info.use_boundary_classes = web_cls == PhonemeStateClasses.boundary
    s.label_info.use_word_end_classes = web_cls == PhonemeStateClasses.word_end
    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    s.alignments[train_key] = alignment
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=False,
        input_key="data_preparation",
        chunk_size=CONF_CHUNKING,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
    )

    # ---------------------- returnn config---------------
    partition_epochs = {"train": 40, "dev": 1}

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
    int_loss_scale = get_int_loss_scale(3, int_losses)
    network_builder = conformer.get_best_model_config(
        conf_size,
        num_classes=s.label_info.get_n_of_dense_classes(),
        label_smoothing=CONF_LABEL_SMOOTHING,
        time_tag_name=time_tag_name,
        int_loss_at_layer=None,
        int_loss_scale=int_loss_scale,
    )
    for loss_idx, *_ in int_losses:
        network_builder.add_auxiliary_loss(idx=loss_idx)
    network = network_builder.network
    network = augment_net_with_label_pops(network, label_info=s.label_info)
    network = augment_net_with_monophone_outputs(
        network,
        label_info=s.label_info,
        add_mlps=True,
        encoder_output_len=conf_size,
        final_ctx_type=PhoneticContext.triphone_forward,
        label_smoothing=CONF_LABEL_SMOOTHING,
        l2=L2,
        use_multi_task=True,
    )
    for loss_idx, ctx, center_only in int_losses:
        network = aux_loss.add_intermediate_loss(
            network,
            time_tag_name=time_tag_name,
            encoder_output_len=conf_size,
            label_info=s.label_info,
            label_smoothing=CONF_LABEL_SMOOTHING,
            l2=L2,
            at_layer=loss_idx,
            scale=int_loss_scale,
            context=ctx,
            center_state_only=center_only,
        )

    base_config = {
        **s.initial_nn_args,
        **oclr.get_oclr_config(num_epochs=num_epochs),
        **CONF_SA_CONFIG,
        "batch_size": 6144,
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "chunking": CONF_CHUNKING,
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.0,
        "network": network,
        "extern_data": {
            "data": {"dim": 50},
            **extern_data.get_extern_data_config(label_info=s.label_info, time_tag_name=time_tag_name),
        },
    }
    keep_epochs = [550, num_epochs]
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
    s.set_returnn_config_for_experiment("fh", copy.deepcopy(returnn_config))

    train_args = {
        **s.initial_train_args,
        "returnn_config": returnn_config,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
    }

    s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
        on_2080=False,
    )
    s.set_mono_priors_returnn_rasr(
        key="fh",
        epoch=keep_epochs[-2],
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
    )

    s.set_binaries_for_crp("dev-other", RS_RASR_BINARY_PATH)

    for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-other"]):
        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.monophone,
            crp_corpus=crp_k,
            epoch=ep,
            gpu=False,
            tensor_map={**FH_DECODING_TENSOR_CONFIG},
            recompile_graph_for_feature_scorer=True,
        )
        recognizer.recognize_count_lm(
            label_info=s.label_info,
            search_parameters=recog_args,
            num_encoder_output=conf_size,
            rerun_after_opt_lm=True,
            calculate_stats=True,
        )

    return s
