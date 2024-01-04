__all__ = ["run", "run_single"]

import copy
import dataclasses
import math
import typing
from dataclasses import dataclass
import itertools
import textwrap

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import delayed_ops, gs, tk

# -------------------- Recipes --------------------

from i6_core import corpus, lexicon, mm, rasr, returnn
from i6_core.util import instanciate_delayed

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.analysis import PlotViterbiAlignmentsJob
from ...setups.common.nn import baum_welch, oclr, returnn_time_tag
from ...setups.common.nn.sum_learning_rates_scores import SumScoresInLearningRatesFileJob
from ...setups.common.util.delayed import GetWrapper
from ...setups.fh import system as fh_system
from ...setups.fh.decoder.config import SearchParameters
from ...setups.fh.decoder.search import DecodingTensorMap
from ...setups.fh.factored import LabelInfo, PhoneticContext, RasrStateTying
from ...setups.fh.network import aux_loss, diphone_joint_output, extern_data
from ...setups.fh.network.augment import (
    augment_net_with_diphone_outputs,
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    augment_net_with_triphone_outputs,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups
from ..config_2023_05_baselines_thesis_tf2.config import SCRATCH_ALIGNMENT

from .config import (
    CONF_FH_DECODING_TENSOR_CONFIG,
    CONF_FOCAL_LOSS,
    CONF_LABEL_SMOOTHING,
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
    alignment: tk.Path
    alignment_name: str
    decode_all_corpora: bool
    run_performance_study: bool
    tune_decoding: bool

    filter_segments: typing.Optional[typing.List[str]] = None


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    exp = Experiment(
        alignment=tk.Path(SCRATCH_ALIGNMENT, cached=True),
        alignment_name="10ms-B",
        decode_all_corpora=False,
        run_performance_study=False,
        tune_decoding=True,
    )
    sys = run_single(returnn_root=returnn_root, exp=exp)

    return exp, sys


def run_single(returnn_root: tk.Path, exp: Experiment):
    # ******************** HY Init ********************

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()
    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True, dc_detection=False)
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
    if exp.filter_segments is not None:
        s.filter_segments = exp.filter_segments
    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    s.alignments[train_key] = exp.alignment
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=False,
        input_key="data_preparation",
        chunk_size="256:64",
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
    )

    # ---------------------- returnn config---------------
    CONF_MODEL_DIM = 512
    PARTITION_EPOCHS = {"train": 20, "dev": 1}
    SS_FACTOR = 4
    SUM_SCORE_KEYS_DI = ["dev_score_center-output", "dev_score_left-output"]
    TENSOR_CONFIG = dataclasses.replace(
        CONF_FH_DECODING_TENSOR_CONFIG,
        in_encoder_output="conformer_12_output/add",
        in_seq_length="extern_data/placeholders/classes/classes_dim0_size",
    )
    ZHOU_L2 = 5e-6

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
    returnn_config = get_conformer_config(
        conf_model_dim=CONF_MODEL_DIM, label_info=s.label_info, time_tag_name=time_tag_name, ss_factor=SS_FACTOR
    )
    viterbi_keep_epochs = [100, 200, 300, 350, 400]
    base_post_config = {
        "cleanup_old_models": {
            "keep_best_n": 3,
            "keep_last_n": 5,
            "keep": viterbi_keep_epochs,
        },
    }
    update_config = returnn.ReturnnConfig(
        config=s.initial_nn_args,
        post_config=base_post_config,
        python_prolog={
            "recursion": ["import sys", "sys.setrecursionlimit(4000)"],
            "numpy": "import numpy as np",
            "time": time_prolog,
        },
    )
    returnn_config.update(update_config)

    peak_lr_7e4_config = returnn.ReturnnConfig(
        config={},
        python_epilog=[
            dynamic_learning_rate_7e4,
            "dynamic_learning_rate = dynamic_learning_rate_7e4",
        ],
    )
    peak_lr_9e4_config = returnn.ReturnnConfig(
        config={},
        python_epilog=[
            dynamic_learning_rate_9e4,
            "dynamic_learning_rate = dynamic_learning_rate_9e4",
        ],
    )

    returnn_cfg_mo = get_monophone_network(
        returnn_config=returnn_config, conf_model_dim=CONF_MODEL_DIM, l2=ZHOU_L2, label_info=s.label_info
    )
    returnn_cfg_mo_sil_p5 = add_ce_silence_penalization(returnn_cfg_mo, loss_scale=5.0)
    returnn_cfg_mo_sil_p5_7e4 = copy.deepcopy(returnn_cfg_mo_sil_p5)
    returnn_cfg_mo_sil_p5_7e4.update(peak_lr_7e4_config)
    returnn_cfg_mo_sil_p5_9e4 = copy.deepcopy(returnn_cfg_mo_sil_p5)
    returnn_cfg_mo_sil_p5_9e4.update(peak_lr_9e4_config)
    returnn_cfg_di = get_diphone_network(
        returnn_config=returnn_config, conf_model_dim=CONF_MODEL_DIM, l2=ZHOU_L2, label_info=s.label_info
    )
    returnn_cfg_di_sil_p5 = add_ce_silence_penalization(returnn_cfg_di, loss_scale=5.0)
    returnn_cfg_di_sil_p5_7e4 = copy.deepcopy(returnn_cfg_di_sil_p5)
    returnn_cfg_di_sil_p5_7e4.update(peak_lr_7e4_config)
    returnn_cfg_di_sil_p5_9e4 = copy.deepcopy(returnn_cfg_di_sil_p5)
    returnn_cfg_di_sil_p5_9e4.update(peak_lr_9e4_config)
    # returnn_cfg_di_add = get_diphone_network(
    #     returnn_config=returnn_config, additive=True, conf_model_dim=CONF_MODEL_DIM, l2=ZHOU_L2, label_info=s.label_info
    # )
    returnn_cfg_tri = get_triphone_network(
        returnn_config=returnn_config, conf_model_dim=CONF_MODEL_DIM, l2=ZHOU_L2, label_info=s.label_info
    )
    # returnn_cfg_tri_add = get_triphone_network(
    #     returnn_config=returnn_config, additive=True, conf_model_dim=CONF_MODEL_DIM, l2=ZHOU_L2, label_info=s.label_info
    # )
    configs = [
        returnn_cfg_mo,
        returnn_cfg_mo_sil_p5_7e4,
        returnn_cfg_mo_sil_p5_9e4,
        returnn_cfg_di,
        returnn_cfg_di_sil_p5_7e4,
        returnn_cfg_di_sil_p5_9e4,
        returnn_cfg_tri,
    ]
    names = [
        "mono",
        "mono-sp5-7e4",
        "mono-sp5-9e4",
        "di",
        "di-sp5-7e4",
        "di-sp5-9e4",
        "tri",
    ]
    keys = [f"fh-{name}" for name in names]

    for cfg, name, key in zip(configs, names, keys):
        post_name = f"conf-{name}-zhou"
        print(f"fh {post_name}")

        s.set_experiment_dict(key, exp.alignment_name, name, postfix_name=post_name)
        s.set_returnn_config_for_experiment(key, copy.deepcopy(cfg))

        train_args = {
            **s.initial_train_args,
            "num_epochs": viterbi_keep_epochs[-1],
            "partition_epochs": PARTITION_EPOCHS,
            "returnn_config": copy.deepcopy(cfg),
        }
        s.returnn_rasr_training(
            experiment_key=key,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            nn_train_args=train_args,
            on_2080=True,
        )

    for (key, returnn_config), ep, crp_k in itertools.product(zip(keys, configs), viterbi_keep_epochs, ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        if key.startswith("fh-mono"):
            decode_monophone(
                s,
                key=key,
                crp_k=crp_k,
                returnn_config=returnn_config,
                epoch=ep,
                prior_epoch=min(ep, viterbi_keep_epochs[-2]),
                tune=ep == viterbi_keep_epochs[-1],
            )
        elif key.startswith("fh-di"):
            decode_diphone(
                s,
                key=key,
                crp_k=crp_k,
                returnn_config=returnn_config,
                epoch=ep,
                prior_epoch=min(ep, viterbi_keep_epochs[-2]),
                tune=ep == viterbi_keep_epochs[-1],
            )
        elif key.startswith("fh-tri"):
            if ep >= viterbi_keep_epochs[-2]:
                decode_triphone(
                    s,
                    key=key,
                    crp_k=crp_k,
                    returnn_config=returnn_config,
                    epoch=ep,
                    prior_epoch_or_key=min(ep, viterbi_keep_epochs[-2]),
                    tensor_config=TENSOR_CONFIG,
                    tune=ep == viterbi_keep_epochs[-1],
                )
        else:
            raise ValueError(f"unknown name {key}")

    # #############
    # FURTHER STEPS
    # #############
    fine_tune_keep_epochs = [25, 50, 100, 150, 200, 250, 275, 300]

    lrates = oclr.get_learning_rates(
        lrate=5e-5,
        increase=0,
        constLR=math.floor(fine_tune_keep_epochs[-1] * 0.45),
        decay=math.floor(fine_tune_keep_epochs[-1] * 0.45),
        decMinRatio=0.1,
        decMaxRatio=1,
    )
    constant_linear_decrease_lr_config = returnn.ReturnnConfig(
        config={
            "learning_rates": list(
                np.concatenate([lrates, np.linspace(min(lrates), 1e-6, fine_tune_keep_epochs[-1] - len(lrates))])
            )
        },
        post_config={
            "cleanup_old_models": {
                "keep_best_n": 3,
                "keep_last_n": 5,
                "keep": fine_tune_keep_epochs,
            },
        },
        python_epilog={"dynamic_lr_reset": "dynamic_learning_rate = None"},
    )
    newbob_lr_config = returnn.ReturnnConfig(
        config={
            "learning_rate": 1e-3,
        },
        post_config={
            "cleanup_old_models": {
                "keep_best_n": 3,
                "keep_last_n": 5,
                "keep": fine_tune_keep_epochs,
            },
        },
        python_epilog={"dynamic_lr_reset": "dynamic_learning_rate = None"},
    )

    mono_train_job = s.experiments["fh-mono"]["train_job"]
    import_mono_config = import_config(mono_train_job.out_checkpoints[viterbi_keep_epochs[-1]])

    di_train_job = s.experiments["fh-di"]["train_job"]
    import_di_config = import_config(di_train_job.out_checkpoints[viterbi_keep_epochs[-1]])

    # di_sp2_train_job = s.experiments["fh-di-sp2"]["train_job"]
    # import_di_sp2_config = import_config(di_sp2_train_job.out_checkpoints[viterbi_keep_epochs[-1]])

    tri_train_job = s.experiments["fh-tri"]["train_job"]
    import_tri_config = import_config(tri_train_job.out_checkpoints[viterbi_keep_epochs[-1]])

    # #####################
    # FORCE-ALIGNED DIPHONE
    # #####################

    alignment_name = "40ms-di-fa"
    di_forced_alignment_j = force_align_diphone(
        s,
        key="fh-di",
        alignment_name=alignment_name,
        epoch=viterbi_keep_epochs[-1],
        prior_epoch=viterbi_keep_epochs[-2],
        returnn_config=returnn_cfg_di,
    )
    allophones = lexicon.StoreAllophonesJob(s.crp[s.crp_names["train"]])
    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=di_forced_alignment_j.out_alignment_bundle,
        allophones_path=allophones.out_allophone_file,
        segments=[
            "train-other-960/2920-156224-0013/2920-156224-0013",
            "train-other-960/2498-134786-0003/2498-134786-0003",
            "train-other-960/6178-86034-0008/6178-86034-0008",
            "train-other-960/5983-39669-0034/5983-39669-0034",
        ],
        show_labels=False,
        show_title=False,
        monophone=True,
    )
    tk.register_output(f"alignments/{alignment_name}/plots", plots.out_plot_folder)

    di_fa_config_newbob = copy.deepcopy(returnn_cfg_di)
    fa_update_config = returnn.ReturnnConfig(
        config={
            "chunking": ({"classes": 64, "data": 256}, {"classes": 32, "data": 128}),
            "network": {
                **returnn_cfg_di.config["network"],
                "reinterpret_classes": {
                    "class": "reinterpret_data",
                    "from": "data:classes",
                    "set_dim_tags": {"T": returnn.CodeWrapper(f"{time_tag_name}.ceildiv_right(2).ceildiv_right(2)")},
                },
                "futureLabel": {
                    **returnn_cfg_di.config["network"]["futureLabel"],
                    "from": "reinterpret_classes",
                },
                "popFutureLabel": {
                    **returnn_cfg_di.config["network"]["popFutureLabel"],
                    "from": "reinterpret_classes",
                },
            },
            "dev": {"reduce_target_factor": 4},
            "train": {"reduce_target_factor": 4},
        },
    )
    di_fa_config_newbob.update(fa_update_config)
    di_fa_config_newbob.update(import_di_config)
    di_fa_config_newbob.update(newbob_lr_config)

    configs = [di_fa_config_newbob]
    names = ["di-fa-newbob"]
    keys = [f"fh-{name}" for name in names]

    for config, name, key in zip(configs, names, keys):
        post_name = f"conf-{name}-zhou"

        print(f"fa {post_name}")

        s.set_experiment_dict(key, alignment_name, name, postfix_name=post_name)
        s.set_returnn_config_for_experiment(key, copy.deepcopy(config))

        train_args = {
            **s.initial_train_args,
            "num_epochs": viterbi_keep_epochs[-1],
            "partition_epochs": PARTITION_EPOCHS,
            "returnn_config": copy.deepcopy(config),
        }
        s.returnn_rasr_training(
            experiment_key=key,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            nn_train_args=train_args,
            include_alignment=di_forced_alignment_j.out_alignment_bundle,
        )

    for key, crp_k, ep in itertools.product(keys, ["dev-other"], viterbi_keep_epochs):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        decode_diphone(
            s,
            key=key,
            crp_k=crp_k,
            returnn_config=returnn_cfg_di,
            epoch=ep,
            prior_epoch=min(ep, viterbi_keep_epochs[-2]),
            tune=ep == viterbi_keep_epochs[-1],
        )

    di_fa_newbob_train_job = s.experiments["fh-di-fa-newbob"]["train_job"]
    import_di_fa_newbob_config = import_config(di_fa_newbob_train_job.out_checkpoints[viterbi_keep_epochs[-1]])

    # ###############################
    # TWO STAGE MULTI STAGE TRAININGS
    # ###############################

    di_from_mono_staged_net_cfg = returnn.ReturnnConfig(
        config={"copy_param_mode": "subset"},
        staged_network_dict={
            1: {
                **returnn_cfg_mo.config["network"],
                "linear1-diphone": {
                    **returnn_cfg_mo.config["network"]["linear1-diphone"],
                    "from": [returnn_cfg_mo.config["network"]["linear1-diphone"]["from"]],
                },
                "#copy_param_mode": "subset",
            },
            2: {**returnn_cfg_di.config["network"], "#copy_param_mode": "subset"},
        },
    )
    di_from_mono_cfg = copy.deepcopy(returnn_cfg_di)
    di_from_mono_cfg.config.pop("network", None)
    di_from_mono_cfg.update(di_from_mono_staged_net_cfg)
    di_from_mono_cfg.update(newbob_lr_config)
    di_from_mono_cfg.update(import_mono_config)
    zero_lr_first_epoch_cfg = returnn.ReturnnConfig(config={"learning_rates": [1e-8, 1e-3]})
    di_from_mono_fixed_lr_cfg = copy.deepcopy(di_from_mono_cfg)
    di_from_mono_fixed_lr_cfg.update(zero_lr_first_epoch_cfg)
    tri_from_mono_staged_net_cfg = returnn.ReturnnConfig(
        config={"copy_param_mode": "subset"},
        staged_network_dict={
            1: {
                **returnn_cfg_mo.config["network"],
                "linear1-diphone": {
                    **returnn_cfg_mo.config["network"]["linear1-diphone"],
                    "from": [returnn_cfg_mo.config["network"]["linear1-diphone"]["from"]],
                },
                "#copy_param_mode": "subset",
            },
            2: {**returnn_cfg_tri.config["network"], "#copy_param_mode": "subset"},
        },
    )
    tri_from_mono_cfg = copy.deepcopy(returnn_cfg_tri)
    tri_from_mono_cfg.config.pop("network", None)
    tri_from_mono_cfg.update(tri_from_mono_staged_net_cfg)
    tri_from_mono_cfg.update(newbob_lr_config)
    tri_from_mono_cfg.update(import_mono_config)
    tri_from_di_staged_net_cfg = returnn.ReturnnConfig(
        config={"copy_param_mode": "subset"},
        staged_network_dict={
            1: {**returnn_cfg_di.config["network"], "#copy_param_mode": "subset"},
            2: {**returnn_cfg_tri.config["network"], "#copy_param_mode": "subset"},
        },
    )
    tri_from_di_cfg = copy.deepcopy(returnn_cfg_tri)
    tri_from_di_cfg.config.pop("network", None)
    tri_from_di_cfg.update(tri_from_di_staged_net_cfg)
    tri_from_di_cfg.update(newbob_lr_config)
    tri_from_di_cfg.update(import_di_config)
    tri_from_di_fixed_lr_cfg = copy.deepcopy(tri_from_di_cfg)
    tri_from_di_fixed_lr_cfg.update(zero_lr_first_epoch_cfg)
    tri_from_di_fa_cfg = copy.deepcopy(returnn_cfg_tri)
    tri_from_di_fa_cfg.config.pop("network", None)
    tri_from_di_fa_cfg.update(tri_from_di_staged_net_cfg)
    tri_from_di_fa_cfg.update(newbob_lr_config)
    tri_from_di_fa_cfg.update(import_di_fa_newbob_config)

    configs = [
        (di_from_mono_cfg, returnn_cfg_di, "di-from-mono"),
        (di_from_mono_fixed_lr_cfg, returnn_cfg_di, "di-from-mono-fixedlr"),
        (tri_from_mono_cfg, returnn_cfg_tri, "tri-from-mono"),
        (tri_from_di_cfg, returnn_cfg_tri, "tri-from-di"),
        (tri_from_di_fixed_lr_cfg, returnn_cfg_tri, "tri-from-di-fixedlr"),
        (tri_from_di_fa_cfg, returnn_cfg_tri, "tri-from-di-fa-newbob"),
    ]
    keys = [f"fh-{name}" for _, _, name in configs]
    for (returnn_config, original_returnn_config, name), key in zip(configs, keys):
        post_name = f"conf-{name}-zhou"
        print(f"ms {post_name}")

        s.set_experiment_dict(key, exp.alignment_name, name, postfix_name=post_name)
        s.set_returnn_config_for_experiment(key, copy.deepcopy(original_returnn_config))

        train_args = {
            **s.initial_train_args,
            "num_epochs": fine_tune_keep_epochs[-1],
            "partition_epochs": PARTITION_EPOCHS,
            "returnn_config": copy.deepcopy(returnn_config),
        }
        s.returnn_rasr_training(
            experiment_key=key,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            nn_train_args=train_args,
        )

    for ((_, returnn_config, _), key), crp_k, ep in itertools.product(
        zip(configs, keys), ["dev-other"], fine_tune_keep_epochs
    ):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        if key.startswith("fh-di"):
            decode_diphone(
                s,
                key=key,
                crp_k=crp_k,
                returnn_config=returnn_config,
                epoch=ep,
                prior_epoch=min(ep, fine_tune_keep_epochs[-2]),
                tune=ep == fine_tune_keep_epochs[-1],
            )
        elif key.startswith("fh-tri"):
            decode_triphone(
                s,
                key=key,
                crp_k=crp_k,
                returnn_config=returnn_config,
                epoch=ep,
                prior_epoch_or_key="fh-tri",
                tensor_config=TENSOR_CONFIG,
                tune=ep == fine_tune_keep_epochs[-1],
            )
        else:
            raise NotImplementedError("Cannot decode multistage monophones")

    tri_from_di_train_job = s.experiments["fh-tri-from-di"]["train_job"]
    import_tri_from_di_config = import_config(tri_from_di_train_job.out_checkpoints[fine_tune_keep_epochs[-1]])
    tri_from_di_newbob_best_epoch_job = returnn.GetBestTFCheckpointJob(
        model_dir=tri_from_di_train_job.out_model_dir,
        learning_rates=SumScoresInLearningRatesFileJob(
            tri_from_di_train_job.out_learning_rates, keys=SUM_SCORE_KEYS_DI
        ).out_learning_rates,
        key="sum_score",
    )
    import_tri_from_di_sel_config = import_config(tri_from_di_newbob_best_epoch_job.out_checkpoint)

    # #################################
    # THREE STAGE MULTI STAGE TRAININGS
    # #################################

    di_from_mono_newbob_train_job = s.experiments["fh-di-from-mono"]["train_job"]
    import_di_from_mono_newbob_config = import_config(
        di_from_mono_newbob_train_job.out_checkpoints[fine_tune_keep_epochs[-1]]
    )
    di_from_mono_fixedlr_newbob_train_job = s.experiments["fh-di-from-mono-fixedlr"]["train_job"]
    import_di_from_mono_fixedlr_newbob_config = import_config(
        di_from_mono_fixedlr_newbob_train_job.out_checkpoints[fine_tune_keep_epochs[-1]]
    )
    di_from_mono_newbob_best_epoch_job = returnn.GetBestTFCheckpointJob(
        model_dir=di_from_mono_newbob_train_job.out_model_dir,
        learning_rates=SumScoresInLearningRatesFileJob(
            di_from_mono_newbob_train_job.out_learning_rates, keys=SUM_SCORE_KEYS_DI
        ).out_learning_rates,
        key="sum_score",
    )
    import_di_from_mono_newbob_sel_epoch_config = import_config(di_from_mono_newbob_best_epoch_job.out_checkpoint)

    tri_from_di_from_mono_cfg = copy.deepcopy(returnn_cfg_tri)
    tri_from_di_from_mono_cfg.config.pop("network", None)
    tri_from_di_from_mono_cfg.update(tri_from_di_staged_net_cfg)
    tri_from_di_from_mono_cfg.update(newbob_lr_config)
    tri_from_di_from_mono_cfg.update(import_di_from_mono_newbob_config)
    tri_from_di_from_mono_sel_epoch_cfg = copy.deepcopy(returnn_cfg_tri)
    tri_from_di_from_mono_sel_epoch_cfg.config.pop("network", None)
    tri_from_di_from_mono_sel_epoch_cfg.update(tri_from_di_staged_net_cfg)
    tri_from_di_from_mono_sel_epoch_cfg.update(newbob_lr_config)
    tri_from_di_from_mono_sel_epoch_cfg.update(import_di_from_mono_newbob_sel_epoch_config)

    returnn_cfg_tri_safe = copy.deepcopy(returnn_cfg_tri)
    returnn_cfg_tri_safe.config["network"] = {
        k: v for k, v in returnn_cfg_tri_safe.config["network"].items() if not k.lower().startswith("aux")
    }

    configs = [
        (tri_from_di_from_mono_cfg, returnn_cfg_tri_safe, "tri-from-di-from-mono"),
        (tri_from_di_from_mono_sel_epoch_cfg, returnn_cfg_tri_safe, "tri-from-di-from-mono-sel"),
    ]
    keys = [f"fh-{name}" for _, _, name in configs]
    for (returnn_config, original_returnn_config, name), key in zip(configs, keys):
        post_name = f"conf-{name}-zhou"
        print(f"ms {post_name}")

        s.set_experiment_dict(key, exp.alignment_name, name, postfix_name=post_name)
        s.set_returnn_config_for_experiment(key, copy.deepcopy(original_returnn_config))

        train_args = {
            **s.initial_train_args,
            "num_epochs": fine_tune_keep_epochs[-1],
            "partition_epochs": PARTITION_EPOCHS,
            "returnn_config": copy.deepcopy(returnn_config),
        }
        s.returnn_rasr_training(
            experiment_key=key,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            nn_train_args=train_args,
        )

    for ((_, returnn_config, _), key), crp_k, ep in itertools.product(
        zip(configs, keys), ["dev-other"], fine_tune_keep_epochs
    ):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        if key.startswith("fh-di"):
            decode_diphone(
                s,
                key=key,
                crp_k=crp_k,
                returnn_config=returnn_config,
                epoch=ep,
                prior_epoch=min(ep, fine_tune_keep_epochs[-2]),
                tune=ep == fine_tune_keep_epochs[-1],
            )
        elif key.startswith("fh-tri"):
            decode_triphone(
                s,
                key=key,
                crp_k=crp_k,
                returnn_config=returnn_config,
                epoch=ep,
                prior_epoch_or_key="fh-tri",
                tensor_config=TENSOR_CONFIG,
                tune=ep == fine_tune_keep_epochs[-1],
            )
        else:
            raise NotImplementedError("Cannot decode multistage monophones")

    # ####################
    # FULL-SUM FINE TUNING
    # ####################

    batch_size_config = returnn.ReturnnConfig(config={"batch_size": 10_000})
    smooth_fs_constlr_config = returnn.ReturnnConfig(
        config={},
        python_epilog=[
            dynamic_learning_rate_fs,
            "dynamic_learning_rate = dynamic_learning_rate_fs",
        ],
    )

    mo_ft_sys = copy.deepcopy(s)
    mo_ft_sys.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.monophone)
    mo_ft_sys.lexicon_args["norm_pronunciation"] = False
    mo_ft_sys._update_am_setting_for_all_crps(
        train_tdp_type="heuristic",
        eval_tdp_type="heuristic",
        add_base_allophones=False,
    )
    returnn_cfg_mo_ft = remove_label_pops_and_losses_from_returnn_config(returnn_cfg_mo)
    returnn_cfg_mo_ft = baum_welch.augment_for_fast_bw(
        crp=mo_ft_sys.crp[s.crp_names["train"]],
        from_output_layer="center-output",
        returnn_config=returnn_cfg_mo_ft,
        log_linear_scales=baum_welch.BwScales(label_posterior_scale=1.0, transition_scale=0.3),
    )
    returnn_cfg_mo_from_scratch = copy.deepcopy(returnn_cfg_mo_ft)
    returnn_cfg_mo_from_scratch.update(batch_size_config)
    returnn_cfg_mo_ft_constlr = copy.deepcopy(returnn_cfg_mo_ft)
    returnn_cfg_mo_ft_constlr.update(batch_size_config)
    returnn_cfg_mo_ft_constlr.update(constant_linear_decrease_lr_config)
    returnn_cfg_mo_ft_constlr.update(import_mono_config)
    returnn_cfg_mo_ft_newbob = copy.deepcopy(returnn_cfg_mo_ft)
    returnn_cfg_mo_ft_newbob.update(batch_size_config)
    returnn_cfg_mo_ft_newbob.update(newbob_lr_config)
    returnn_cfg_mo_ft_newbob.update(import_mono_config)

    di_ft_sys = copy.deepcopy(s)
    di_ft_sys.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
    di_ft_sys.lexicon_args["norm_pronunciation"] = False
    di_ft_sys._update_am_setting_for_all_crps(
        train_tdp_type="heuristic",
        eval_tdp_type="heuristic",
        add_base_allophones=False,
    )
    returnn_cfg_di_ft = remove_label_pops_and_losses_from_returnn_config(returnn_cfg_di)
    returnn_cfg_di_ft = diphone_joint_output.augment_to_joint_diphone_softmax(
        returnn_config=returnn_cfg_di_ft,
        label_info=di_ft_sys.label_info,
        out_joint_score_layer="output",
        log_softmax=True,
        prepare_for_train=True,
    )
    returnn_cfg_di_ft = baum_welch.augment_for_fast_bw(
        crp=di_ft_sys.crp[s.crp_names["train"]],
        from_output_layer="output",
        returnn_config=returnn_cfg_di_ft,
        log_linear_scales=baum_welch.BwScales(label_posterior_scale=1.0, transition_scale=0.3),
    )
    returnn_cfg_di_ft_constlr = copy.deepcopy(returnn_cfg_di_ft)
    returnn_cfg_di_ft_constlr.update(batch_size_config)
    returnn_cfg_di_ft_constlr.update(constant_linear_decrease_lr_config)
    returnn_cfg_di_ft_constlr.update(import_di_config)
    returnn_cfg_di_ft_newbob = copy.deepcopy(returnn_cfg_di_ft)
    returnn_cfg_di_ft_newbob.update(batch_size_config)
    returnn_cfg_di_ft_newbob.update(newbob_lr_config)
    returnn_cfg_di_ft_newbob.update(import_di_config)
    # returnn_cfg_di_sp_ft = copy.deepcopy(returnn_cfg_di_ft_constlr)
    # returnn_cfg_di_sp_ft.update(import_di_sp2_config)
    returnn_cfg_di_fa_ft = copy.deepcopy(returnn_cfg_di_ft_constlr)
    returnn_cfg_di_fa_ft.update(import_di_fa_newbob_config)
    returnn_cfg_di_from_mono_ft_constlr = copy.deepcopy(returnn_cfg_di_ft)
    returnn_cfg_di_from_mono_ft_constlr.update(batch_size_config)
    returnn_cfg_di_from_mono_ft_constlr.update(constant_linear_decrease_lr_config)
    returnn_cfg_di_from_mono_ft_constlr.update(import_di_from_mono_newbob_config)
    returnn_cfg_di_from_mono_fixedlr_ft_constlr = copy.deepcopy(returnn_cfg_di_from_mono_ft_constlr)
    returnn_cfg_di_from_mono_fixedlr_ft_constlr.update(import_di_from_mono_fixedlr_newbob_config)

    tri_ft_sys = copy.deepcopy(s)
    tri_ft_sys.lexicon_args["norm_pronunciation"] = False
    tri_ft_sys._update_am_setting_for_all_crps(
        train_tdp_type="heuristic",
        eval_tdp_type="heuristic",
        add_base_allophones=False,
    )
    returnn_cfg_tri_ft = copy.deepcopy(returnn_cfg_tri)
    returnn_cfg_tri_ft = diphone_joint_output.augment_to_joint_diphone_softmax(
        returnn_config=returnn_cfg_tri_ft,
        label_info=di_ft_sys.label_info,  # use diphone label info
        out_joint_score_layer="output",
        log_softmax=True,
        prepare_for_train=True,
        remove_triphone_outputs=False,
    )
    returnn_cfg_tri_ft = baum_welch.augment_for_fast_bw(
        crp=di_ft_sys.crp[s.crp_names["train"]],  # use diphone label info
        from_output_layer="output",
        returnn_config=returnn_cfg_tri_ft,
        log_linear_scales=baum_welch.BwScales(label_posterior_scale=1.0, transition_scale=0.3),
    )
    returnn_cfg_tri_ft.config["network"]["right-output"] = {
        **returnn_cfg_tri_ft.config["network"]["right-output"],
        "register_as_extern_data": "right-output",
    }
    returnn_cfg_tri_ft_constlr = copy.deepcopy(returnn_cfg_tri_ft)
    returnn_cfg_tri_ft_constlr.update(batch_size_config)
    returnn_cfg_tri_ft_constlr.update(smooth_fs_constlr_config)
    returnn_cfg_tri_ft_constlr.update(import_tri_config)
    returnn_cfg_tri_from_di_ft_constlr = copy.deepcopy(returnn_cfg_tri_ft)
    returnn_cfg_tri_from_di_ft_constlr.update(batch_size_config)
    returnn_cfg_tri_from_di_ft_constlr.update(smooth_fs_constlr_config)
    returnn_cfg_tri_from_di_ft_constlr.update(import_tri_from_di_config)
    returnn_cfg_tri_from_di_sel_ft_constlr = copy.deepcopy(returnn_cfg_tri_ft)
    returnn_cfg_tri_from_di_sel_ft_constlr.update(batch_size_config)
    returnn_cfg_tri_from_di_sel_ft_constlr.update(smooth_fs_constlr_config)
    returnn_cfg_tri_from_di_sel_ft_constlr.update(import_tri_from_di_sel_config)

    configs = [
        (returnn_cfg_mo_from_scratch, returnn_cfg_mo, mo_ft_sys, "mono-scratch-oclr"),
    ]
    keys = [f"fh-{name}" for _, _, _, name in configs]
    for (returnn_config, original_returnn_config, sys, name), key in zip(configs, keys):
        post_name = f"conf-{name}-zhou"
        print(f"bw {post_name}")

        sys.set_experiment_dict(key, "bw", name, postfix_name=post_name)
        sys.set_returnn_config_for_experiment(key, copy.deepcopy(original_returnn_config))

        train_args = {
            **s.initial_train_args,
            "num_epochs": viterbi_keep_epochs[-1],
            "partition_epochs": PARTITION_EPOCHS,
            "returnn_config": copy.deepcopy(returnn_config),
        }
        sys.returnn_rasr_training(
            experiment_key=key,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            nn_train_args=train_args,
        )

    for ((_, orig_returnn_config, sys, name), key), crp_k, ep in itertools.product(
        zip(configs, keys), ["dev-other"], viterbi_keep_epochs
    ):
        sys.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        if key.startswith("fh-mono"):
            decode_monophone(
                sys,
                key=key,
                crp_k=crp_k,
                returnn_config=orig_returnn_config,
                epoch=ep,
                prior_epoch=min(ep, viterbi_keep_epochs[-2]),
                tune=ep == viterbi_keep_epochs[-1],
            )
        elif key.startswith("fh-di"):
            decode_diphone(
                sys,
                key=key,
                crp_k=crp_k,
                returnn_config=orig_returnn_config,
                epoch=ep,
                prior_epoch=min(ep, viterbi_keep_epochs[-2]),
                tune=ep == viterbi_keep_epochs[-1],
            )
        elif key.startswith("fh-tri"):
            decode_triphone(
                sys,
                key=key,
                crp_k=crp_k,
                returnn_config=orig_returnn_config,
                graph_config=orig_returnn_config,
                epoch=ep,
                prior_epoch_or_key="fh-tri",
                tensor_config=TENSOR_CONFIG,
                tune=ep == viterbi_keep_epochs[-1],
            )
        else:
            raise NotImplementedError("Cannot bw-fine-tune triphones")

    configs = [
        (returnn_cfg_mo_ft_constlr, returnn_cfg_mo, mo_ft_sys, "mono-fs-constlr"),
        (returnn_cfg_mo_ft_newbob, returnn_cfg_mo, mo_ft_sys, "mono-fs-newbob"),
        (returnn_cfg_di_ft_constlr, returnn_cfg_di, di_ft_sys, "di-fs-constlr"),
        (returnn_cfg_di_ft_newbob, returnn_cfg_di, di_ft_sys, "di-fs-newbob"),
        # (returnn_cfg_di_sp_ft, returnn_cfg_di, di_ft_sys, "di-fs-sp-constlr"),
        (returnn_cfg_di_fa_ft, returnn_cfg_di, di_ft_sys, "di-fs-fa-constlr"),
        (returnn_cfg_di_from_mono_ft_constlr, returnn_cfg_di, di_ft_sys, "di-fs-constlr-from-mono"),
        (returnn_cfg_di_from_mono_fixedlr_ft_constlr, returnn_cfg_di, di_ft_sys, "di-fs-constlr-from-mono-fixedlr"),
        (returnn_cfg_tri_ft_constlr, returnn_cfg_tri_safe, tri_ft_sys, "tri-fs-constlr"),
        (returnn_cfg_tri_from_di_ft_constlr, returnn_cfg_tri_safe, tri_ft_sys, "tri-fs-constlr-from-di"),
        (returnn_cfg_tri_from_di_sel_ft_constlr, returnn_cfg_tri_safe, tri_ft_sys, "tri-sel-fs-constlr-from-di"),
    ]
    keys = [f"fh-{name}" for _, _, _, name in configs]
    for (returnn_config, original_returnn_config, sys, name), key in zip(configs, keys):
        post_name = f"conf-{name}-zhou"
        print(f"bw {post_name}")

        sys.set_experiment_dict(key, "bw", name, postfix_name=post_name)
        sys.set_returnn_config_for_experiment(key, copy.deepcopy(original_returnn_config))

        train_args = {
            **s.initial_train_args,
            "num_epochs": fine_tune_keep_epochs[-1],
            "partition_epochs": PARTITION_EPOCHS,
            "returnn_config": copy.deepcopy(returnn_config),
        }
        sys.returnn_rasr_training(
            experiment_key=key,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            nn_train_args=train_args,
            on_2080=True,  # broken GPUs
        )

    for ((_, orig_returnn_config, sys, name), key), crp_k, ep in itertools.product(
        zip(configs, keys), ["dev-other"], fine_tune_keep_epochs
    ):
        sys.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        if key.startswith("fh-mono"):
            decode_monophone(
                sys,
                key=key,
                crp_k=crp_k,
                returnn_config=orig_returnn_config,
                epoch=ep,
                prior_epoch=min(ep, fine_tune_keep_epochs[-2]),
                tune=ep == fine_tune_keep_epochs[-1],
            )
        elif key.startswith("fh-di"):
            decode_diphone(
                sys,
                key=key,
                crp_k=crp_k,
                returnn_config=orig_returnn_config,
                epoch=ep,
                prior_epoch=min(ep, fine_tune_keep_epochs[-2]),
                tune=ep == fine_tune_keep_epochs[-1],
                tune_extremely=ep == fine_tune_keep_epochs[-1]
                and name in ["di-fs-constlr-from-mono", "di-fs-constlr-from-mono-fixedlr"],
            )
        elif key.startswith("fh-tri"):
            decode_triphone(
                sys,
                key=key,
                crp_k=crp_k,
                returnn_config=orig_returnn_config,
                graph_config=orig_returnn_config,
                epoch=ep,
                prior_epoch_or_key="fh-tri",
                tensor_config=TENSOR_CONFIG,
                tune=ep == fine_tune_keep_epochs[-1],
            )
        else:
            raise NotImplementedError("Cannot bw-fine-tune triphones")

    # ######################################
    # MULTI-STAGE AFTER FULL-SUM FINE TUNING
    # ######################################

    mono_scratch_oclr_train_job = mo_ft_sys.experiments["fh-mono-scratch-oclr"]["train_job"]
    import_mono_scratch_oclr_config = import_config(
        mono_scratch_oclr_train_job.out_checkpoints[viterbi_keep_epochs[-1]]
    )
    mono_fs_train_job = mo_ft_sys.experiments["fh-mono-fs-constlr"]["train_job"]
    import_mono_fs_constlr_config = import_config(mono_fs_train_job.out_checkpoints[fine_tune_keep_epochs[-1]])

    di_vit_from_mono_ft_config = copy.deepcopy(di_from_mono_cfg)
    di_vit_from_mono_ft_config.update(import_mono_fs_constlr_config)
    # classic staged network dicts do not work due to code-wrapped time tags
    net_dict = copy.deepcopy(  # instanciate-delayed
        {
            1: {"#copy_param_mode": "subset", **returnn_cfg_mo_ft_constlr.config["network"]},
            2: {"#copy_param_mode": "subset", **returnn_cfg_di_ft_constlr.config["network"]},
        }
    )
    net_dict_formatted = delayed_ops.DelayedFormat(
        textwrap.dedent(
            """
            networks_dict = {!r}
            def get_network(epoch, **kwargs):
              for epoch_ in sorted(networks_dict.keys(), reverse=True):
                if epoch_ <= epoch:
                  return networks_dict[epoch_]
              assert False, \"Error, no networks found\"
            """
        ),
        GetWrapper(net_dict).function(instanciate_delayed),
    )
    di_ft_from_mono_ft_staged_net_config = returnn.ReturnnConfig(config={}, python_epilog=net_dict_formatted)
    di_ft_from_mono_ft_config = copy.deepcopy(returnn_cfg_di_ft_constlr)
    di_ft_from_mono_ft_config.config.pop("network", None)
    di_ft_from_mono_ft_config.hash_full_python_code = False
    di_ft_from_mono_ft_config.staged_network_dict = None
    di_ft_from_mono_ft_config.update(di_ft_from_mono_ft_staged_net_config)
    di_ft_from_mono_ft_config.update(import_mono_fs_constlr_config)
    di_ft_from_mono_scratch_config = copy.deepcopy(di_ft_from_mono_ft_config)
    di_ft_from_mono_scratch_config.update(import_mono_scratch_oclr_config)
    configs = [
        (di_vit_from_mono_ft_config, returnn_cfg_di, s, "di-from-mono-fs-constlr"),
        (di_ft_from_mono_ft_config, returnn_cfg_di, di_ft_sys, "di-fs-constlr-from-mono-fs-constlr"),
        (di_ft_from_mono_scratch_config, returnn_cfg_di, di_ft_sys, "di-fs-constlr-from-mono-scratch-oclr"),
    ]
    keys = [f"fh-{name}" for _, _, _, name in configs]
    for (returnn_config, orig_returnn_cfg, sys, name), key in zip(configs, keys):
        post_name = f"conf-{name}-zhou"
        print(f"bw+ {post_name}")

        sys.set_experiment_dict(key, "bw", name, postfix_name=post_name)
        sys.set_returnn_config_for_experiment(key, copy.deepcopy(orig_returnn_cfg))

        train_args = {
            **s.initial_train_args,
            "num_epochs": fine_tune_keep_epochs[-1],
            "partition_epochs": PARTITION_EPOCHS,
            "returnn_config": copy.deepcopy(returnn_config),
        }
        sys.returnn_rasr_training(
            experiment_key=key,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            nn_train_args=train_args,
        )

    for ((_, orig_returnn_config, sys, name), key), crp_k, ep in itertools.product(
        zip(configs, keys), ["dev-other"], fine_tune_keep_epochs
    ):
        sys.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        if key.startswith("fh-mono"):
            decode_monophone(
                sys,
                key=key,
                crp_k=crp_k,
                returnn_config=orig_returnn_config,
                epoch=ep,
                prior_epoch=min(ep, fine_tune_keep_epochs[-2]),
                tune=ep == fine_tune_keep_epochs[-1],
            )
        elif key.startswith("fh-di"):
            decode_diphone(
                sys,
                key=key,
                crp_k=crp_k,
                returnn_config=orig_returnn_config,
                epoch=ep,
                prior_epoch=min(ep, fine_tune_keep_epochs[-2]),
                tune=ep == fine_tune_keep_epochs[-1],
                # tune_extremely=ep == fine_tune_keep_epochs[-1] and name in ["di-fs-constlr-from-mono-fs-constlr"],
            )
        elif key.startswith("fh-tri"):
            decode_triphone(
                sys,
                key=key,
                crp_k=crp_k,
                returnn_config=orig_returnn_config,
                graph_config=orig_returnn_config,
                epoch=ep,
                prior_epoch_or_key="fh-tri",
                tensor_config=TENSOR_CONFIG,
                tune=ep == fine_tune_keep_epochs[-1],
            )
        else:
            raise NotImplementedError("Cannot bw-fine-tune triphones")

    ##############
    # 6.0 DI MODEL
    ##############

    # best config for diphone-fine-tune
    key = "fh-di-fs-constlr"
    tdp_sil = (10, 10, "infinity", 20)
    params_4gram = dataclasses.replace(
        di_ft_sys.get_cart_params(key),
        beam=20,
        lm_scale=2.1,
        tdp_scale=0.4,
        tdp_silence=tdp_sil,
        tdp_non_word=tdp_sil,
    ).with_prior_scale(0.6)
    for ep in [262, 275, 282, 294, 297, 298, 299]:
        decode_diphone(
            di_ft_sys,
            key=key,
            epoch=ep,
            crp_k="dev-other",
            params=params_4gram,
            prior_epoch=fine_tune_keep_epochs[-2],
            returnn_config=returnn_cfg_di,
            tune=False,
        )

    # best config for diphone-fine-tune
    tdp_sil = (10, 10, "infinity", 20)
    params_4gram = dataclasses.replace(
        di_ft_sys.get_cart_params(key),
        beam=20,
        lm_scale=2.126,
        tdp_scale=0.4,
        tdp_silence=tdp_sil,
        tdp_non_word=tdp_sil,
    ).with_prior_scale(0.6)
    for crp_k, (neural, decoding_params) in itertools.product(
        ["dev-clean", "dev-other", "test-clean", "test-other"],
        [(False, params_4gram), (True, params_4gram.with_lm_scale(3.6))],
    ):
        decode_diphone(
            di_ft_sys,
            key=key,
            epoch=275,
            crp_k=crp_k,
            params=decoding_params,
            prior_epoch=fine_tune_keep_epochs[-2],
            returnn_config=returnn_cfg_di,
            tune=False,
            neural_lm=neural,
        )

    ##############
    # 5.9 DI MODEL
    ##############

    # /u/mgunz/setups/2023-08--subsampling-new/output/50_best_ss_config_40ms/recog/di-fs-constlr-from-mono-from-bw-conf-di-fs-constlr-from-mono-zhou/ep300/lm-4gram/rp275-opt/dev-other/am01.00_lm2.233403010221315_prior00.60_tdp0.4_tdpspeechloop3.0_forward0.0_skipinfinity_exit0.0_tdpsilenceloop10_forward10_skipinfinity_exit20_tdpnonspeechloop10_forward10_skipinfinity_exit20_ps2.0_altas00.00_beam20_beamlimit100000.reports/sclite.dtl
    # 15:Percent Total Error       =    5.9%   (3030)
    #
    # /u/mgunz/setups/2023-08--subsampling-new/output/50_best_ss_config_40ms/recog/di-fs-constlr-from-mono-from-bw-conf-di-fs-constlr-from-mono-zhou/ep300/lm-4gram/rp275-opt/dev-other/am01.00_lm1.881966025156_prior00.40_tdp0.4_tdpspeechloop3.0_forward0.0_skipinfinity_exit0.0_tdpsilenceloop10_forward10_skipinfinity_exit20_tdpnonspeechloop10_forward10_skipinfinity_exit20_ps2.0_altas00.00_beam20_beamlimit100000.reports/sclite.dtl
    # 15:Percent Total Error       =    5.9%   (3031)
    #
    # /work/asr4/raissi/setups/librispeech/960-ls/2023-12--LFR/output/tune_decode/tune-lm/diphone-from-full-sum/e300/dev-other/Beam24.0-Lm2.0-Pron1.5-prJ-C0.6-allAllos-tdpScale-0.1-spTdp-13.0,0.0,inf,0.0-silTdp-13.0,0.0,inf,20.0.wer/sclite.dtl
    # 15:Percent Total Error       =    5.9%   (2995)

    key = "fh-di-fs-constlr-from-mono"
    tuned_sil_tdp_tina = (13.0, 0.0, "infinity", 20.0)
    tuned_sp_tdp_tina = (13.0, 0.0, "infinity", 0.0)
    shared_params = dataclasses.replace(
        di_ft_sys.get_cart_params(key),
        beam=24,
        lm_scale=2.0,
        normalize_pronunciation=True,
        tdp_scale=0.1,
        tdp_speech=tuned_sp_tdp_tina,
        tdp_silence=tuned_sil_tdp_tina,
        tdp_non_word=tuned_sil_tdp_tina,
    ).with_prior_scale(0.6)
    params_4gram = dataclasses.replace(shared_params, pron_scale=1.5)
    params_neural = dataclasses.replace(
        shared_params,
        am_scale=0.7,
        beam=22,
        beam_limit=500_000,
        lm_lookahead_scale=shared_params.lm_scale / 2,
        pron_scale=0.3,
        we_pruning=0.8,
        we_pruning_limit=2000,
    )
    for crp_k, (params, neural) in itertools.product(
        ["dev-other", "test-other"],
        [(params_4gram, False), (params_neural, True)],
    ):
        decode_diphone(
            di_ft_sys,
            key=key,
            epoch=fine_tune_keep_epochs[-1],
            crp_k=crp_k,
            params=params,
            prior_epoch=fine_tune_keep_epochs[-1],
            returnn_config=returnn_cfg_di,
            fix_respect_add_all_allophones=True,
            neural_lm=neural,
            tune=False,
            use_full_prior_share=True,
        )


def decode_monophone(
    s: fh_system.FactoredHybridSystem,
    key: str,
    crp_k: str,
    returnn_config: returnn.ReturnnConfig,
    epoch: int,
    prior_epoch: int,
    tune: bool,
):
    clean_returnn_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)

    s.set_mono_priors_returnn_rasr(
        key=key,
        epoch=prior_epoch,
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        smoothen=True,
        returnn_config=clean_returnn_config,
        output_layer_name="center-output",
    )

    monophone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.monophone)
    nn_pch_config = copy.deepcopy(clean_returnn_config)
    nn_pch_config.config["network"]["center-output"] = {
        **nn_pch_config.config["network"]["center-output"],
        "class": "linear",
        "activation": "log_softmax",
    }
    nn_pch_config.config["network"]["output"] = {
        "class": "copy",
        "from": "center-output",
        "register_as_extern_data": "output",
    }
    tying_cfg = rasr.RasrConfig()
    tying_cfg.type = "monophone-dense"

    search_params = [
        dataclasses.replace(s.get_cart_params(key), beam=22, lm_scale=2.0, tdp_scale=0.4).with_prior_scale(0.6)
    ]
    if tune:
        base_cfg = search_params[-1]
        other_cfgs = [
            base_cfg.with_prior_scale(round(p_c, 1))
            .with_tdp_scale(round(tdp_s, 1))
            .with_tdp_silence(tdp_silence)
            .with_tdp_non_word(tdp_silence)
            for p_c, tdp_s, tdp_silence in itertools.product(
                np.linspace(0.2, 0.8, 4),
                [0.1, *np.linspace(0.2, 0.8, 4)],
                [base_cfg.tdp_silence, (10, 10, "infinity", 20)],
            )
        ]
        search_params.extend(other_cfgs)

    for cfg in search_params:
        s.recognize_cart(
            key=key,
            epoch=epoch,
            crp_corpus=crp_k,
            n_cart_out=monophone_li.get_n_of_dense_classes(),
            cart_tree_or_tying_config=tying_cfg,
            params=cfg,
            log_softmax_returnn_config=nn_pch_config,
            calculate_statistics=True,
            opt_lm_am_scale=True,
            fix_tdp_non_word_tying=True,
            prior_epoch=prior_epoch,
            rtf=8,
            cpu_rqmt=2,
        )


def decode_diphone(
    s: fh_system.FactoredHybridSystem,
    key: str,
    crp_k: str,
    returnn_config: returnn.ReturnnConfig,
    epoch: int,
    prior_epoch: int,
    tune: bool,
    fix_respect_add_all_allophones: bool = False,
    neural_lm: bool = False,
    params: typing.Optional[SearchParameters] = None,
    tune_extremely: bool = False,
    use_full_prior_share: bool = False,
):
    assert not ((tune or tune_extremely) and neural_lm), "neural LM decodings should be done with tuned parameters"

    clean_returnn_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)

    prior_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
        returnn_config=clean_returnn_config,
        label_info=s.label_info,
        out_joint_score_layer="output",
        log_softmax=False,
    )
    s.set_mono_priors_returnn_rasr(
        key=key,
        epoch=epoch if tune_extremely else prior_epoch,
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        smoothen=True,
        returnn_config=prior_returnn_config,
        output_layer_name="output",
        data_share=1.0 if tune_extremely or use_full_prior_share else 1.0 / 3.0,
    )

    diphone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
    nn_pch_config = diphone_joint_output.augment_to_joint_diphone_softmax(
        returnn_config=clean_returnn_config,
        label_info=s.label_info,
        out_joint_score_layer="output",
        log_softmax=True,
    )
    tying_cfg = rasr.RasrConfig()
    tying_cfg.type = "diphone-dense"
    search_params = [
        dataclasses.replace(s.get_cart_params(key), beam=20, lm_scale=2.5, tdp_scale=0.4).with_prior_scale(0.6)
        if params is None
        else params.with_prior_files(s.get_cart_params(key))  # ensure priors are set correctly
    ]

    if tune or tune_extremely:
        base_cfg = search_params[-1]
        other_cfgs = [
            base_cfg.with_prior_scale(round(p_c, 1))
            .with_tdp_scale(round(tdp_s, 1))
            .with_tdp_silence(tdp_silence)
            .with_tdp_non_word(tdp_silence)
            for p_c, tdp_s, tdp_silence in itertools.product(
                np.linspace(0.2, 0.8, 4),
                [0.1, *np.linspace(0.2, 0.8, 4)],
                [base_cfg.tdp_silence, (10, 10, "infinity", 20)],
            )
        ]
        search_params.extend(other_cfgs)

    for cfg in search_params:
        s.recognize_cart(
            key=key,
            epoch=epoch,
            crp_corpus=crp_k,
            n_cart_out=diphone_li.get_n_of_dense_classes(),
            cart_tree_or_tying_config=tying_cfg,
            params=cfg,
            log_softmax_returnn_config=nn_pch_config,
            calculate_statistics=True,
            opt_lm_am_scale=True,
            fix_tdp_non_word_tying=True,
            prior_epoch=epoch if tune_extremely else prior_epoch,
            decode_trafo_lm=neural_lm,
            recognize_only_trafo=neural_lm,
            cpu_rqmt=2,
            mem_rqmt=4 if not neural_lm else 8,
            gpu=neural_lm,
            rtf=2 if not neural_lm else 20,
            remove_or_set_concurrency=False,
            fix_respect_add_all_allophones=fix_respect_add_all_allophones,
        )

    if not tune_extremely:
        return

    base_cfg = search_params[0]
    tdp_geom_values = [round(v, 1) for v in [0.0, *np.geomspace(3.0, 20.0, num=4)]]
    additional_silence_tdps = [
        (p_l, p_f, "infinity", p_e)
        for p_l, p_f, p_e in itertools.product(tdp_geom_values, tdp_geom_values, tdp_geom_values)
        if p_e > p_l >= p_f and p_e > 0
    ]
    # silence_tdps = [base_cfg.tdp_silence, (10, 10, "infinity", 20), *additional_silence_tdps]
    silence_tdps = [(10, 10, "infinity", 20)]
    extreme_configs = [
        dataclasses.replace(
            base_cfg.with_prior_scale(round(p_c, 1)),
            am_scale=am,
            normalize_pronunciation=bool(am != 1.0),  # suppress np boolean
            lm_scale=1.9 if p_c <= 0.4 else 2.3,
            pron_scale=pron,
            tdp_scale=round(tdp_s, 1),
            tdp_silence=tdp_silence,
            tdp_non_word=tdp_silence,
        )
        for am, pron, p_c, tdp_s, tdp_silence in itertools.product(
            [0.1, 0.3, 0.7, 1.0],
            sorted([round(v, 1) for v in [1.2, *np.geomspace(0.1, 2.0, num=4)]]),
            [round(v, 1) for v in np.linspace(0.2, 0.8, 4)],
            [0.1, 0.2, 0.4, 0.6],
            silence_tdps,
        )
    ]

    print(f"tuning {key}/{epoch} extremely with {len(extreme_configs)} distinct configurations")

    for cfg in extreme_configs:
        s.recognize_cart(
            key=key,
            epoch=epoch,
            crp_corpus=crp_k,
            n_cart_out=diphone_li.get_n_of_dense_classes(),
            cart_tree_or_tying_config=tying_cfg,
            params=cfg,
            log_softmax_returnn_config=nn_pch_config,
            calculate_statistics=True,
            opt_lm_am_scale=False,
            fix_tdp_non_word_tying=True,
            prior_epoch=epoch if tune_extremely else prior_epoch,
            decode_trafo_lm=neural_lm,
            recognize_only_trafo=neural_lm,
            cpu_rqmt=2,
            mem_rqmt=4 if not neural_lm else 8,
            gpu=neural_lm,
            rtf=2 if not neural_lm else 20,
            remove_or_set_concurrency=1,
            fix_respect_add_all_allophones=fix_respect_add_all_allophones,
        )


def decode_triphone(
    s: fh_system.FactoredHybridSystem,
    key: str,
    crp_k: str,
    returnn_config: returnn.ReturnnConfig,
    epoch: int,
    prior_epoch_or_key: typing.Union[int, str],
    tensor_config: DecodingTensorMap,
    tune: bool,
    graph_config: typing.Optional[returnn.ReturnnConfig] = None,
):
    s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

    if isinstance(prior_epoch_or_key, int):
        s.set_triphone_priors_returnn_rasr(
            key=key,
            epoch=prior_epoch_or_key,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
            data_share=0.1,
        )
    else:
        s.experiments[key]["priors"] = s.experiments[prior_epoch_or_key]["priors"]

    if graph_config:
        s.set_graph_for_experiment(key=key, override_cfg=graph_config)

    recognizer, recog_args_base = s.get_recognizer_and_args(
        key=key,
        context_type=PhoneticContext.triphone_forward,
        crp_corpus=crp_k,
        epoch=epoch,
        gpu=False,
        tensor_map=tensor_config,
        set_batch_major_for_feature_scorer=True,
        lm_gc_simple_hash=True,
    )

    tdp_sil = (10, 10, "infinity", 20)
    recog_args = dataclasses.replace(
        recog_args_base,
        lm_scale=2.5,
        tdp_scale=0.2,
        tdp_silence=tdp_sil,
        tdp_non_word=tdp_sil,
    ).with_prior_scale(0.4, 0.2, 0.2)
    recognizer.recognize_count_lm(
        label_info=s.label_info,
        search_parameters=recog_args,
        num_encoder_output=512,
        rerun_after_opt_lm=True,
        calculate_stats=True,
        rtf_cpu=35,
    )

    if tune:
        tune_args = recog_args_base.with_lm_scale(2.5).with_tdp_scale(0.2)
        best_config = recognizer.recognize_optimize_scales(
            label_info=s.label_info,
            search_parameters=tune_args,
            num_encoder_output=512,
            altas_value=8,
            tdp_sil=[tune_args.tdp_silence, (10, 10, "infinity", 20)],
            prior_scales=list(
                itertools.product(
                    np.linspace(0.4, 0.8, 3),
                    [0.1, 0.2, 0.4],
                    np.linspace(0.2, 0.8, 4),
                )
            ),
            tdp_scales=[0.1, 0.2, 0.4],
        )
        recognizer.recognize_count_lm(
            label_info=s.label_info,
            search_parameters=dataclasses.replace(best_config, we_pruning=0.7),
            num_encoder_output=512,
            rerun_after_opt_lm=True,
            calculate_stats=True,
            name_override="best/4gram",
            rtf_cpu=35,
        )


def force_align(
    sys: fh_system.FactoredHybridSystem,
    alignment_name: str,
    key: str,
    epoch: int,
    prior_epoch: int,
    nn_pch_config: returnn.ReturnnConfig,
    prior_config: returnn.ReturnnConfig,
    tying: RasrStateTying,
    tdp_scale: float = 1.0,
    sil_e: float = 0.0,
    prior_scale: float = 0.6,
) -> mm.AlignmentJob:
    assert tying in [RasrStateTying.diphone], "rest is unimplemented (prior generation)"

    import i6_core.returnn.flow as flow

    (_, dev_data_inputs, _) = lbs_data_setups.get_data_inputs()

    sys = copy.deepcopy(sys)
    crp_k = sys.crp_names["train"]

    sys.label_info = dataclasses.replace(sys.label_info, state_tying=tying)
    sys.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)
    sys.create_stm_from_corpus(crp_k)
    sys._set_scorer_for_corpus(crp_k)
    sys._init_lm(crp_k, **next(iter(dev_data_inputs.values())).lm)
    sys._update_crp_am_setting(crp_k, tdp_type="default", add_base_allophones=False)

    sys.set_mono_priors_returnn_rasr(
        key=key,
        epoch=prior_epoch,
        train_corpus_key=sys.crp_names["train"],
        dev_corpus_key=sys.crp_names["cvtrain"],
        smoothen=True,
        returnn_config=prior_config,
        output_layer_name="output",
    )

    crp = copy.deepcopy(sys.crp[crp_k])
    crp.acoustic_model_config.allophones.add_all = False
    crp.acoustic_model_config.allophones.add_from_lexicon = True
    crp.acoustic_model_config.state_tying.type = "diphone-dense"
    crp.acoustic_model_config.tdp.applicator_type = "corrected"
    crp.acoustic_model_config.tdp.scale = tdp_scale

    v = (3.0, 0.0, "infinity", 0.0)
    sv = (0.0, 3.0, "infinity", sil_e)
    keys = ["loop", "forward", "skip", "exit"]
    for i, k in enumerate(keys):
        crp.acoustic_model_config.tdp["*"][k] = v[i]
        crp.acoustic_model_config.tdp["silence"][k] = sv[i]

    crp.concurrent = 300
    crp.segment_path = corpus.SegmentCorpusJob(sys.corpora[sys.train_key].corpus_file, crp.concurrent).out_segment_path

    p_mixtures = mm.CreateDummyMixturesJob(sys.label_info.get_n_of_dense_classes(), sys.initial_nn_args["num_input"])
    priors = sys.experiments[key]["priors"].center_state_prior
    feature_scorer = rasr.PrecomputedHybridFeatureScorer(
        prior_mixtures=p_mixtures.out_mixtures,
        prior_file=priors.file,
        priori_scale=prior_scale,
    )
    sys.set_graph_for_experiment(key=key, override_cfg=nn_pch_config)
    tf_flow = flow.make_precomputed_hybrid_tf_feature_flow(
        tf_graph=sys.experiments[key]["graph"]["inference"],
        tf_checkpoint=sys.experiments[key]["train_job"].out_checkpoints[epoch],
        output_layer_name="output",
        tf_fwd_input_name="tf-fwd-input",
    )
    feature_flow = flow.add_tf_flow_to_base_flow(
        base_flow=sys.feature_flows[crp_k],
        tf_flow=tf_flow,
        tf_fwd_input_name="tf-fwd-input",
    )

    recognizer, _ = sys.get_recognizer_and_args(
        key=key,
        context_type=PhoneticContext.diphone,
        crp_corpus=crp_k,
        epoch=prior_epoch,
        gpu=False,
        tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
        set_batch_major_for_feature_scorer=True,
        lm_gc_simple_hash=True,
    )
    a_job = recognizer.align(
        f"{alignment_name}/pC{prior_scale}-tdp{tdp_scale}-silE{sil_e}",
        crp=crp,
        feature_scorer=feature_scorer,
        feature_flow=feature_flow,
        default_tdp=False,
        set_do_not_normalize_lemma_sequence_scores=False,
        set_no_tying_dense=False,
        rtf=2,
    )

    return a_job


def force_align_diphone(
    sys: fh_system.FactoredHybridSystem,
    returnn_config: returnn.ReturnnConfig,
    **kwargs,
) -> mm.AlignmentJob:
    clean_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)
    prior_config = diphone_joint_output.augment_to_joint_diphone_softmax(
        label_info=sys.label_info,
        log_softmax=False,
        out_joint_score_layer="output",
        returnn_config=clean_config,
    )
    nn_pch_config = diphone_joint_output.augment_to_joint_diphone_softmax(
        label_info=sys.label_info,
        log_softmax=True,
        out_joint_score_layer="output",
        returnn_config=clean_config,
    )

    return force_align(
        sys=sys, prior_config=prior_config, nn_pch_config=nn_pch_config, tying=RasrStateTying.diphone, **kwargs
    )


def import_config(ckpt: returnn.Checkpoint) -> returnn.ReturnnConfig:
    return returnn.ReturnnConfig(
        config={
            "preload_from_files": {
                "existing-model": {
                    "init_for_train": True,
                    "ignore_missing": True,
                    "filename": ckpt,
                },
            },
        }
    )


def get_monophone_network(
    returnn_config: returnn.ReturnnConfig,
    conf_model_dim: int,
    l2: float,
    label_info: LabelInfo,
    out_layer_name: str = "encoder-output",
) -> returnn.ReturnnConfig:
    network = augment_net_with_monophone_outputs(
        returnn_config.config["network"],
        add_mlps=True,
        encoder_output_layer=out_layer_name,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=CONF_FOCAL_LOSS,
        l2=l2,
        label_info=label_info,
        label_smoothing=0.1,
        use_multi_task=True,
    )
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["network"] = network
    return returnn_config


def get_diphone_network(
    returnn_config: returnn.ReturnnConfig,
    conf_model_dim: int,
    l2: float,
    label_info: LabelInfo,
    additive: bool = False,
    out_layer_name: str = "encoder-output",
) -> returnn.ReturnnConfig:
    network = augment_net_with_monophone_outputs(
        returnn_config.config["network"],
        add_mlps=True,
        encoder_output_layer=out_layer_name,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=CONF_FOCAL_LOSS,
        l2=l2,
        label_info=label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        use_multi_task=True,
    )
    network = augment_net_with_diphone_outputs(
        network,
        encoder_output_layer=out_layer_name,
        encoder_output_len=conf_model_dim,
        l2=l2,
        label_smoothing=CONF_LABEL_SMOOTHING,
        ph_emb_size=label_info.ph_emb_size,
        st_emb_size=label_info.st_emb_size,
        use_multi_task=True,
    )

    if additive:
        for name, l in network.items():
            if name.startswith("linear") and "n_out" in l:
                l["n_out"] = conf_model_dim

        network["pastEmbed"]["n_out"] = conf_model_dim
        network["linear1-diphone-linear"] = {
            **network["linear1-diphone"],
            "from": "linear1-diphone",
        }
        network["linear2-diphone"]["from"] = "linear1-diphone-linear"
        network["linear1-diphone"] = {
            "class": "combine",
            "kind": "add",
            "from": network["linear1-diphone"]["from"],
        }

    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["network"] = network
    return returnn_config


def get_triphone_network(
    returnn_config: returnn.ReturnnConfig,
    conf_model_dim: int,
    l2: float,
    label_info: LabelInfo,
    additive: bool = False,
    out_layer_name: str = "encoder-output",
) -> returnn.ReturnnConfig:
    network = augment_net_with_monophone_outputs(
        returnn_config.config["network"],
        add_mlps=True,
        encoder_output_layer=out_layer_name,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=CONF_FOCAL_LOSS,
        l2=l2,
        label_info=label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        use_multi_task=True,
    )
    network = augment_net_with_triphone_outputs(
        network,
        encoder_output_layer=out_layer_name,
        l2=l2,
        ph_emb_size=label_info.ph_emb_size,
        st_emb_size=label_info.st_emb_size,
        variant=PhoneticContext.triphone_forward,
    )

    if additive:
        for name, l in network.items():
            if name.startswith("linear") and "n_out" in l:
                l["n_out"] = conf_model_dim

        network["pastEmbed"]["n_out"] = conf_model_dim
        network["currentState"]["n_out"] = conf_model_dim

        network["linear1-diphone-linear"] = {
            **network["linear1-diphone"],
            "from": "linear1-diphone",
        }
        network["linear2-diphone"]["from"] = "linear1-diphone-linear"
        network["linear1-diphone"] = {
            "class": "combine",
            "kind": "add",
            "from": network["linear1-diphone"]["from"],
        }

        network["linear1-triphone-linear"] = {
            **network["linear1-triphone"],
            "from": "linear1-triphone",
        }
        network["linear2-triphone"]["from"] = "linear1-triphone-linear"
        network["linear1-triphone"] = {
            "class": "combine",
            "kind": "add",
            "from": network["linear1-triphone"]["from"],
        }

    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["network"] = network
    return returnn_config


def add_ce_silence_penalization(
    config: returnn.ReturnnConfig, loss_scale: float = 5.0, silence_id: int = 81
) -> returnn.ReturnnConfig:
    config = copy.deepcopy(config)

    network = config.config["network"]
    network["segmental_loss"] = {
        "class": "eval",
        "eval": "self.network.get_config().typed_value('segmental_loss')(source)",
        "from": ["ce_loss", "mask_silence"],
        "loss": "as_is",
        "loss_opts": {"scale": loss_scale},
    }
    network["ce_loss"] = {
        "class": "loss",
        "from": "center-output",
        "loss_": "ce",
        "target_": "centerState",
    }
    network["mask_silence"] = {
        "class": "compare",
        "from": ["centerState"],
        "kind": "not_equal",
        "value": silence_id,
    }
    additional_epilog = textwrap.dedent(
        """
        def segmental_loss(source):
            import tensorflow.compat.v1 as tf
            loss = source(0, enforce_batch_major=True)
            mask = source(1, enforce_batch_major=True)
            return tf.where(mask, loss, tf.zeros_like(loss))
        """
    )

    update_cfg = returnn.ReturnnConfig(config={}, python_epilog=additional_epilog)
    config.update(update_cfg)

    return config


def get_conformer_config(
    conf_model_dim: int,
    label_info: LabelInfo,
    time_tag_name: str,
    ss_factor: int = 4,
    out_layer_name: str = "encoder-output",
) -> returnn.ReturnnConfig:
    assert ss_factor == 4, "unimplemented"

    ZHOU_L2 = 5e-6
    network = {
        "input_dropout": {"class": "copy", "dropout": 0.1, "from": "input_linear"},
        "input_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conv_merged",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "source": {
            "class": "eval",
            "from": "data",
            "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)",
        },
        "conv_1": {
            "L2": 0.01,
            "activation": "swish",
            "class": "conv",
            "filter_size": (3, 3),
            "from": "conv_source",
            "n_out": 32,
            "padding": "same",
            "with_bias": True,
        },
        "conv_1_pool": {
            "class": "pool",
            "from": "conv_1",
            "mode": "max",
            "padding": "same",
            "pool_size": (1, 2),
            "trainable": False,
        },
        "conv_2": {
            "L2": 0.01,
            "activation": "swish",
            "class": "conv",
            "filter_size": (3, 3),
            "from": "conv_1_pool",
            "n_out": 64,
            "padding": "same",
            "strides": (2, 1),
            "with_bias": True,
        },
        "conv_3": {
            "L2": 0.01,
            "activation": "swish",
            "class": "conv",
            "filter_size": (3, 3),
            "from": "conv_2",
            "n_out": 64,
            "padding": "same",
            "strides": (2, 1),
            "with_bias": True,
        },
        "conv_merged": {"axes": "static", "class": "merge_dims", "from": "conv_3"},
        "conv_source": {"axis": "F", "class": "split_dims", "dims": (-1, 1), "from": "source"},
        "conformer_01_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_01_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_01_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_01_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_01_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_01_conv_mod_pointwise_conv_2",
        },
        "conformer_01_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_01_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_01_conv_mod_ln": {"class": "layer_norm", "from": "conformer_01_ffmod_1_half_res_add"},
        "conformer_01_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_01_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_01_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_01_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_01_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_01_conv_mod_dropout", "conformer_01_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_01_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_01_conv_mod_bn",
        },
        "conformer_01_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_1_dropout_linear",
        },
        "conformer_01_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_01_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_01_ffmod_1_dropout", "input_dropout"],
        },
        "conformer_01_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_01_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_01_ffmod_1_ln": {"class": "layer_norm", "from": "input_dropout"},
        "conformer_01_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_2_dropout_linear",
        },
        "conformer_01_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_01_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_01_ffmod_2_dropout", "conformer_01_mhsa_mod_res_add"],
        },
        "conformer_01_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_01_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_01_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_01_mhsa_mod_res_add"},
        "conformer_01_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_01_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_01_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_01_mhsa_mod_att_linear"},
        "conformer_01_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_01_conv_mod_res_add"},
        "conformer_01_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_01_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_01_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_01_mhsa_mod_dropout", "conformer_01_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_01_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_01_mhsa_mod_ln",
            "key_shift": "conformer_01_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_01_output": {"class": "layer_norm", "from": "conformer_01_ffmod_2_half_res_add"},
        "conformer_02_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_02_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_02_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_02_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_02_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_02_conv_mod_pointwise_conv_2",
        },
        "conformer_02_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_02_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_02_conv_mod_ln": {"class": "layer_norm", "from": "conformer_02_ffmod_1_half_res_add"},
        "conformer_02_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_02_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_02_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_02_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_02_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_02_conv_mod_dropout", "conformer_02_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_02_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_02_conv_mod_bn",
        },
        "conformer_02_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_1_dropout_linear",
        },
        "conformer_02_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_02_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_02_ffmod_1_dropout", "conformer_01_output"],
        },
        "conformer_02_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_02_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_02_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_01_output"},
        "conformer_02_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_2_dropout_linear",
        },
        "conformer_02_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_02_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_02_ffmod_2_dropout", "conformer_02_mhsa_mod_res_add"],
        },
        "conformer_02_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_02_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_02_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_02_mhsa_mod_res_add"},
        "conformer_02_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_02_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_02_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_02_mhsa_mod_att_linear"},
        "conformer_02_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_02_conv_mod_res_add"},
        "conformer_02_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_02_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_02_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_02_mhsa_mod_dropout", "conformer_02_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_02_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_02_mhsa_mod_ln",
            "key_shift": "conformer_02_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_02_output": {"class": "layer_norm", "from": "conformer_02_ffmod_2_half_res_add"},
        "conformer_03_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_03_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_03_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_03_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_03_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_03_conv_mod_pointwise_conv_2",
        },
        "conformer_03_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_03_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_03_conv_mod_ln": {"class": "layer_norm", "from": "conformer_03_ffmod_1_half_res_add"},
        "conformer_03_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_03_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_03_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_03_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_03_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_03_conv_mod_dropout", "conformer_03_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_03_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_03_conv_mod_bn",
        },
        "conformer_03_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_1_dropout_linear",
        },
        "conformer_03_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_03_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_03_ffmod_1_dropout", "conformer_02_output"],
        },
        "conformer_03_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_03_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_03_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_02_output"},
        "conformer_03_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_2_dropout_linear",
        },
        "conformer_03_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_03_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_03_ffmod_2_dropout", "conformer_03_mhsa_mod_res_add"],
        },
        "conformer_03_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_03_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_03_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_03_mhsa_mod_res_add"},
        "conformer_03_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_03_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_03_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_03_mhsa_mod_att_linear"},
        "conformer_03_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_03_conv_mod_res_add"},
        "conformer_03_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_03_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_03_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_03_mhsa_mod_dropout", "conformer_03_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_03_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_03_mhsa_mod_ln",
            "key_shift": "conformer_03_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_03_output": {"class": "layer_norm", "from": "conformer_03_ffmod_2_half_res_add"},
        "conformer_04_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_04_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_04_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_04_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_04_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_04_conv_mod_pointwise_conv_2",
        },
        "conformer_04_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_04_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_04_conv_mod_ln": {"class": "layer_norm", "from": "conformer_04_ffmod_1_half_res_add"},
        "conformer_04_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_04_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_04_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_04_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_04_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_04_conv_mod_dropout", "conformer_04_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_04_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_04_conv_mod_bn",
        },
        "conformer_04_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_1_dropout_linear",
        },
        "conformer_04_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_04_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_04_ffmod_1_dropout", "conformer_03_output"],
        },
        "conformer_04_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_04_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_04_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_03_output"},
        "conformer_04_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_2_dropout_linear",
        },
        "conformer_04_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_04_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_04_ffmod_2_dropout", "conformer_04_mhsa_mod_res_add"],
        },
        "conformer_04_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_04_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_04_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_04_mhsa_mod_res_add"},
        "conformer_04_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_04_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_04_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_04_mhsa_mod_att_linear"},
        "conformer_04_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_04_conv_mod_res_add"},
        "conformer_04_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_04_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_04_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_04_mhsa_mod_dropout", "conformer_04_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_04_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_04_mhsa_mod_ln",
            "key_shift": "conformer_04_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_04_output": {"class": "layer_norm", "from": "conformer_04_ffmod_2_half_res_add"},
        "conformer_05_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_05_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_05_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_05_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_05_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_05_conv_mod_pointwise_conv_2",
        },
        "conformer_05_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_05_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_05_conv_mod_ln": {"class": "layer_norm", "from": "conformer_05_ffmod_1_half_res_add"},
        "conformer_05_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_05_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_05_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_05_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_05_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_05_conv_mod_dropout", "conformer_05_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_05_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_05_conv_mod_bn",
        },
        "conformer_05_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_1_dropout_linear",
        },
        "conformer_05_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_05_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_05_ffmod_1_dropout", "conformer_04_output"],
        },
        "conformer_05_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_05_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_05_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_04_output"},
        "conformer_05_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_2_dropout_linear",
        },
        "conformer_05_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_05_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_05_ffmod_2_dropout", "conformer_05_mhsa_mod_res_add"],
        },
        "conformer_05_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_05_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_05_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_05_mhsa_mod_res_add"},
        "conformer_05_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_05_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_05_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_05_mhsa_mod_att_linear"},
        "conformer_05_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_05_conv_mod_res_add"},
        "conformer_05_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_05_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_05_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_05_mhsa_mod_dropout", "conformer_05_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_05_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_05_mhsa_mod_ln",
            "key_shift": "conformer_05_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_05_output": {"class": "layer_norm", "from": "conformer_05_ffmod_2_half_res_add"},
        "conformer_06_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_06_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_06_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_06_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_06_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_06_conv_mod_pointwise_conv_2",
        },
        "conformer_06_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_06_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_06_conv_mod_ln": {"class": "layer_norm", "from": "conformer_06_ffmod_1_half_res_add"},
        "conformer_06_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_06_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_06_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_06_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_06_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_06_conv_mod_dropout", "conformer_06_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_06_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_06_conv_mod_bn",
        },
        "conformer_06_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_1_dropout_linear",
        },
        "conformer_06_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_06_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_06_ffmod_1_dropout", "conformer_05_output"],
        },
        "conformer_06_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_06_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_06_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_05_output"},
        "conformer_06_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_2_dropout_linear",
        },
        "conformer_06_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_06_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_06_ffmod_2_dropout", "conformer_06_mhsa_mod_res_add"],
        },
        "conformer_06_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_06_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_06_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_06_mhsa_mod_res_add"},
        "conformer_06_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_06_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_06_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_06_mhsa_mod_att_linear"},
        "conformer_06_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_06_conv_mod_res_add"},
        "conformer_06_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_06_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_06_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_06_mhsa_mod_dropout", "conformer_06_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_06_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_06_mhsa_mod_ln",
            "key_shift": "conformer_06_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_06_output": {"class": "layer_norm", "from": "conformer_06_ffmod_2_half_res_add"},
        "conformer_07_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_07_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_07_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_07_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_07_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_07_conv_mod_pointwise_conv_2",
        },
        "conformer_07_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_07_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_07_conv_mod_ln": {"class": "layer_norm", "from": "conformer_07_ffmod_1_half_res_add"},
        "conformer_07_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_07_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_07_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_07_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_07_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_07_conv_mod_dropout", "conformer_07_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_07_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_07_conv_mod_bn",
        },
        "conformer_07_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_1_dropout_linear",
        },
        "conformer_07_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_07_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_07_ffmod_1_dropout", "conformer_06_output"],
        },
        "conformer_07_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_07_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_07_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_06_output"},
        "conformer_07_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_2_dropout_linear",
        },
        "conformer_07_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_07_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_07_ffmod_2_dropout", "conformer_07_mhsa_mod_res_add"],
        },
        "conformer_07_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_07_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_07_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_07_mhsa_mod_res_add"},
        "conformer_07_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_07_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_07_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_07_mhsa_mod_att_linear"},
        "conformer_07_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_07_conv_mod_res_add"},
        "conformer_07_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_07_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_07_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_07_mhsa_mod_dropout", "conformer_07_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_07_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_07_mhsa_mod_ln",
            "key_shift": "conformer_07_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_07_output": {"class": "layer_norm", "from": "conformer_07_ffmod_2_half_res_add"},
        "conformer_08_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_08_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_08_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_08_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_08_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_08_conv_mod_pointwise_conv_2",
        },
        "conformer_08_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_08_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_08_conv_mod_ln": {"class": "layer_norm", "from": "conformer_08_ffmod_1_half_res_add"},
        "conformer_08_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_08_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_08_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_08_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_08_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_08_conv_mod_dropout", "conformer_08_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_08_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_08_conv_mod_bn",
        },
        "conformer_08_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_1_dropout_linear",
        },
        "conformer_08_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_08_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_08_ffmod_1_dropout", "conformer_07_output"],
        },
        "conformer_08_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_08_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_08_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_07_output"},
        "conformer_08_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_2_dropout_linear",
        },
        "conformer_08_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_08_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_08_ffmod_2_dropout", "conformer_08_mhsa_mod_res_add"],
        },
        "conformer_08_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_08_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_08_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_08_mhsa_mod_res_add"},
        "conformer_08_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_08_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_08_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_08_mhsa_mod_att_linear"},
        "conformer_08_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_08_conv_mod_res_add"},
        "conformer_08_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_08_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_08_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_08_mhsa_mod_dropout", "conformer_08_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_08_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_08_mhsa_mod_ln",
            "key_shift": "conformer_08_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_08_output": {"class": "layer_norm", "from": "conformer_08_ffmod_2_half_res_add"},
        "conformer_09_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_09_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_09_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_09_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_09_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_09_conv_mod_pointwise_conv_2",
        },
        "conformer_09_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_09_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_09_conv_mod_ln": {"class": "layer_norm", "from": "conformer_09_ffmod_1_half_res_add"},
        "conformer_09_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_09_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_09_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_09_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_09_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_09_conv_mod_dropout", "conformer_09_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_09_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_09_conv_mod_bn",
        },
        "conformer_09_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_1_dropout_linear",
        },
        "conformer_09_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_09_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_09_ffmod_1_dropout", "conformer_08_output"],
        },
        "conformer_09_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_09_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_09_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_08_output"},
        "conformer_09_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_2_dropout_linear",
        },
        "conformer_09_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_09_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_09_ffmod_2_dropout", "conformer_09_mhsa_mod_res_add"],
        },
        "conformer_09_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_09_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_09_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_09_mhsa_mod_res_add"},
        "conformer_09_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_09_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_09_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_09_mhsa_mod_att_linear"},
        "conformer_09_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_09_conv_mod_res_add"},
        "conformer_09_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_09_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_09_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_09_mhsa_mod_dropout", "conformer_09_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_09_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_09_mhsa_mod_ln",
            "key_shift": "conformer_09_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_09_output": {"class": "layer_norm", "from": "conformer_09_ffmod_2_half_res_add"},
        "conformer_10_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_10_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_10_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_10_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_10_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_10_conv_mod_pointwise_conv_2",
        },
        "conformer_10_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_10_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_10_conv_mod_ln": {"class": "layer_norm", "from": "conformer_10_ffmod_1_half_res_add"},
        "conformer_10_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_10_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_10_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_10_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_10_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_10_conv_mod_dropout", "conformer_10_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_10_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_10_conv_mod_bn",
        },
        "conformer_10_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_1_dropout_linear",
        },
        "conformer_10_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_10_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_10_ffmod_1_dropout", "conformer_09_output"],
        },
        "conformer_10_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_10_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_10_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_09_output"},
        "conformer_10_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_2_dropout_linear",
        },
        "conformer_10_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_10_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_10_ffmod_2_dropout", "conformer_10_mhsa_mod_res_add"],
        },
        "conformer_10_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_10_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_10_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_10_mhsa_mod_res_add"},
        "conformer_10_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_10_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_10_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_10_mhsa_mod_att_linear"},
        "conformer_10_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_10_conv_mod_res_add"},
        "conformer_10_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_10_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_10_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_10_mhsa_mod_dropout", "conformer_10_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_10_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_10_mhsa_mod_ln",
            "key_shift": "conformer_10_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_10_output": {"class": "layer_norm", "from": "conformer_10_ffmod_2_half_res_add"},
        "conformer_11_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_11_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_11_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_11_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_11_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_11_conv_mod_pointwise_conv_2",
        },
        "conformer_11_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_11_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_11_conv_mod_ln": {"class": "layer_norm", "from": "conformer_11_ffmod_1_half_res_add"},
        "conformer_11_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_11_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_11_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_11_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_11_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_11_conv_mod_dropout", "conformer_11_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_11_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_11_conv_mod_bn",
        },
        "conformer_11_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_1_dropout_linear",
        },
        "conformer_11_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_11_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_11_ffmod_1_dropout", "conformer_10_output"],
        },
        "conformer_11_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_11_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_11_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_10_output"},
        "conformer_11_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_2_dropout_linear",
        },
        "conformer_11_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_11_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_11_ffmod_2_dropout", "conformer_11_mhsa_mod_res_add"],
        },
        "conformer_11_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_11_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_11_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_11_mhsa_mod_res_add"},
        "conformer_11_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_11_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_11_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_11_mhsa_mod_att_linear"},
        "conformer_11_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_11_conv_mod_res_add"},
        "conformer_11_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_11_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_11_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_11_mhsa_mod_dropout", "conformer_11_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_11_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_11_mhsa_mod_ln",
            "key_shift": "conformer_11_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_11_output": {"class": "layer_norm", "from": "conformer_11_ffmod_2_half_res_add"},
        "conformer_12_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_12_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_12_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_12_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_12_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_12_conv_mod_pointwise_conv_2",
        },
        "conformer_12_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_12_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_12_conv_mod_ln": {"class": "layer_norm", "from": "conformer_12_ffmod_1_half_res_add"},
        "conformer_12_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_12_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_12_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_12_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_12_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_12_conv_mod_dropout", "conformer_12_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_12_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_12_conv_mod_bn",
        },
        "conformer_12_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_1_dropout_linear",
        },
        "conformer_12_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_12_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_12_ffmod_1_dropout", "conformer_11_output"],
        },
        "conformer_12_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_12_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_12_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_11_output"},
        "conformer_12_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_2_dropout_linear",
        },
        "conformer_12_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_12_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_12_ffmod_2_dropout", "conformer_12_mhsa_mod_res_add"],
        },
        "conformer_12_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_12_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_12_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_12_mhsa_mod_res_add"},
        "conformer_12_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_12_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_12_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_12_mhsa_mod_att_linear"},
        "conformer_12_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_12_conv_mod_res_add"},
        "conformer_12_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_12_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_12_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_12_mhsa_mod_dropout", "conformer_12_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_12_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_12_mhsa_mod_ln",
            "key_shift": "conformer_12_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_12_output": {"class": "layer_norm", "from": "conformer_12_ffmod_2_half_res_add"},
        "enc_006": {  # for aux loss
            "class": "copy",
            "from": "conformer_06_output",
            "n_out": conf_model_dim,
        },
        out_layer_name: {
            "class": "copy",
            "from": "conformer_12_output",
            "n_out": conf_model_dim,
        },
    }
    network = augment_net_with_label_pops(network, label_info=label_info, labeling_input="slice_classes1")
    network = {
        **network,
        "slice_classes0": {
            "axis": "T",
            "class": "slice",
            "from": "data:classes",
            "slice_step": ss_factor // 2,
        },
        "slice_classes1": {
            "axis": "T",
            "class": "slice",
            "from": "slice_classes0",
            "slice_step": ss_factor // 2,
        },
    }
    network = aux_loss.add_intermediate_loss(
        network,
        center_state_only=True,
        context=PhoneticContext.monophone,
        encoder_output_len=conf_model_dim,
        focal_loss_factor=CONF_FOCAL_LOSS,
        l2=ZHOU_L2,
        label_info=label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        time_tag_name=time_tag_name,
        upsampling=False,
    )
    config = returnn.ReturnnConfig(
        config={
            "batching": "random",
            "batch_size": 15_000,
            "cache_size": "0",
            "chunking": "256:128",
            "debug_print_layer_output_template": True,
            "extern_data": {
                "data": {"dim": 50, "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)}},
                **extern_data.get_extern_data_config(label_info=label_info, time_tag_name=None),
            },
            "gradient_clip": 20,
            "gradient_noise": 0.0,
            "learning_rate": 0.001,
            "min_learning_rate": 1e-6,
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_error_measure": "sum_dev_score",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "log_batch_size": True,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 20,
            "newbob_multi_update_interval": 1,
            "optimizer": {"class": "nadam"},
            "optimizer_epsilon": 1e-8,
            "max_seqs": 128,
            "network": network,
            "tf_log_memory_usage": True,
            "use_tensorflow": True,
            "update_on_device": True,
            "window": 1,
        },
        hash_full_python_code=True,
        python_epilog=[
            _mask,
            random_mask,
            summary,
            transform,
            dynamic_learning_rate,
        ],
    )
    # Only apply dim tag to classes, not to popped labels as they are popped from
    # the sliced/sampled classes.
    config.config["extern_data"]["classes"]["same_dim_tags_as"] = {"T": returnn.CodeWrapper(time_tag_name)}
    return config


# for debug only
def summary(name, x):
    """
    :param str name:
    :param tf.Tensor x: (batch,time,feature)
    """
    import tensorflow as tf

    # tf.summary.image wants [batch_size, height,  width, channels],
    # we have (batch, time, feature).
    img = tf.expand_dims(x, axis=3)  # (batch,time,feature,1)
    img = tf.transpose(img, [0, 2, 1, 3])  # (batch,feature,time,1)
    tf.summary.image(name, img, max_outputs=10)
    tf.summary.scalar("%s_max_abs" % name, tf.reduce_max(tf.abs(x)))
    mean = tf.reduce_mean(x)
    tf.summary.scalar("%s_mean" % name, mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
    tf.summary.scalar("%s_stddev" % name, stddev)
    tf.summary.histogram("%s_hist" % name, tf.reduce_max(tf.abs(x), axis=2))


def _mask(x, batch_axis, axis, pos, max_amount):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
    """
    import tensorflow as tf

    ndim = x.get_shape().ndims
    n_batch = tf.shape(x)[batch_axis]
    dim = tf.shape(x)[axis]
    amount = tf.random.uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
    pos2 = tf.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
    if batch_axis > axis:
        cond = tf.transpose(cond)  # (dim,batch)
    cond = tf.reshape(cond, [tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)])
    from TFUtil import where_bc

    x = where_bc(cond, 0.0, x)
    return x


def random_mask(x, batch_axis, axis, min_num, max_num, max_dims):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
    """
    import tensorflow as tf

    n_batch = tf.shape(x)[batch_axis]
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
    else:
        num = tf.random.uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.math.log(-tf.math.log(tf.random.uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
    _, indices = tf.nn.top_k(z, num if isinstance(num, int) else tf.reduce_max(num))
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
    if isinstance(num, int):
        for i in range(num):
            x = _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims)
    else:
        _, x = tf.compat.v1.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
            body=lambda i, x: (
                i + 1,
                tf.compat.v1.where(
                    tf.less(i, num),
                    _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims),
                    x,
                ),
            ),
            loop_vars=(0, x),
        )
    return x


def transform(data, network):
    # to be adjusted (20-50%)
    max_time_num = 1
    max_time = 15

    max_feature_num = 5
    max_feature = 5

    # halved before this step
    conservatvie_step = 2000

    x = data.placeholder
    import tensorflow as tf

    # summary("features", x)
    step = network.global_train_step
    increase_flag = tf.compat.v1.where(tf.greater_equal(step, conservatvie_step), 0, 1)

    def get_masked():
        x_masked = x
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=0,
            max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // int(1 / 0.70 * max_time), max_time_num)
            // (1 + increase_flag),
            max_dims=max_time,
        )
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=0,
            max_num=max_feature_num // (1 + increase_flag),
            max_dims=max_feature,
        )
        # summary("features_mask", x_masked)
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


# one cycle LR: triangular linear w.r.t. iterations(steps)
def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    # -- need to be adjusted w.r.t. training -- #
    initialLR = 8e-5
    peakLR = 8e-4
    finalLR = 1e-6
    cycleEpoch = 180
    totalEpoch = 400
    nStep = 2420  # steps/epoch depending on batch_size

    # -- derived -- #
    steps = cycleEpoch * nStep
    stepSize = (peakLR - initialLR) / steps
    steps2 = (totalEpoch - 2 * cycleEpoch) * nStep
    stepSize2 = (initialLR - finalLR) / steps2

    import tensorflow as tf

    n = tf.cast(global_train_step, tf.float32)
    return tf.where(
        global_train_step <= steps,
        initialLR + stepSize * n,
        tf.where(
            global_train_step <= 2 * steps,
            peakLR - stepSize * (n - steps),
            tf.maximum(initialLR - stepSize2 * (n - 2 * steps), finalLR),
        ),
    )


# one cycle LR: triangular linear w.r.t. iterations(steps)
def dynamic_learning_rate_7e4(*, network, global_train_step, learning_rate, **kwargs):
    # -- need to be adjusted w.r.t. training -- #
    initialLR = 8e-5
    peakLR = 7e-4
    finalLR = 1e-6
    cycleEpoch = 180
    totalEpoch = 400
    nStep = 2420  # steps/epoch depending on batch_size

    # -- derived -- #
    steps = cycleEpoch * nStep
    stepSize = (peakLR - initialLR) / steps
    steps2 = (totalEpoch - 2 * cycleEpoch) * nStep
    stepSize2 = (initialLR - finalLR) / steps2

    import tensorflow as tf

    n = tf.cast(global_train_step, tf.float32)
    return tf.where(
        global_train_step <= steps,
        initialLR + stepSize * n,
        tf.where(
            global_train_step <= 2 * steps,
            peakLR - stepSize * (n - steps),
            tf.maximum(initialLR - stepSize2 * (n - 2 * steps), finalLR),
        ),
    )


# one cycle LR: triangular linear w.r.t. iterations(steps)
def dynamic_learning_rate_9e4(*, network, global_train_step, learning_rate, **kwargs):
    # -- need to be adjusted w.r.t. training -- #
    initialLR = 8e-5
    peakLR = 9e-4
    finalLR = 1e-6
    cycleEpoch = 180
    totalEpoch = 400
    nStep = 2420  # steps/epoch depending on batch_size

    # -- derived -- #
    steps = cycleEpoch * nStep
    stepSize = (peakLR - initialLR) / steps
    steps2 = (totalEpoch - 2 * cycleEpoch) * nStep
    stepSize2 = (initialLR - finalLR) / steps2

    import tensorflow as tf

    n = tf.cast(global_train_step, tf.float32)
    return tf.where(
        global_train_step <= steps,
        initialLR + stepSize * n,
        tf.where(
            global_train_step <= 2 * steps,
            peakLR - stepSize * (n - steps),
            tf.maximum(initialLR - stepSize2 * (n - 2 * steps), finalLR),
        ),
    )


# adapt one cycle LR for fine-tuning: half cycle constLR + half cycle decay + final decay
def dynamic_learning_rate_fs(*, network, global_train_step, learning_rate, **kwargs):
    # -- need to be adjusted w.r.t. training -- #
    constLR = 5e-5
    decayLR = 1e-5
    finalLR = 1e-6
    cycleEpoch = 135
    totalEpoch = 300
    nStep = 2230  # steps/epoch depending on batch_size

    # -- derived -- #
    steps = cycleEpoch * nStep
    stepSize = (constLR - decayLR) / steps
    steps2 = (totalEpoch - 2 * cycleEpoch) * nStep
    stepSize2 = (decayLR - finalLR) / steps2

    import tensorflow as tf

    n = tf.cast(global_train_step, tf.float32)
    return tf.where(
        global_train_step <= steps,
        constLR,
        tf.where(
            global_train_step <= 2 * steps,
            constLR - stepSize * (n - steps),
            tf.maximum(decayLR - stepSize2 * (n - 2 * steps), finalLR),
        ),
    )
