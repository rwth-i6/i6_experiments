import copy
import os

from sisyphus import gs, tk

import i6_core.rasr as rasr
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.baselines.librispeech.ls960.gmm.baseline_config import (
    run_librispeech_960_common_baseline,
)
from i6_experiments.users.berger.args.experiments import hybrid as exp_args
from i6_experiments.users.berger.args.returnn.config import Backend, get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.corpus.libri_css.hybrid_data import get_hybrid_data
from i6_experiments.users.berger.network.models import conformer_hybrid_dualchannel_v2 as conformer_hybrid_dualchannel
from i6_experiments.users.berger.recipe.converse.scoring import MeetEvalJob
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import (
    ConfigVariant,
    CustomStepKwargs,
    FeatureType,
    ReturnnConfigs,
    SummaryKey,
)
from i6_experiments.users.berger.systems.functors.rasr_base import LatticeProcessingType
from i6_experiments.users.berger.systems.returnn_legacy_system import (
    ReturnnLegacySystem,
)
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.args.jobs.recognition_args import get_atr_search_parameters


# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 12001

tools = copy.deepcopy(default_tools_v2)


# ********** Returnn config generators **********


def returnn_config_generator(
    variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, **kwargs
) -> ReturnnConfig:
    extra_config: dict = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant != ConfigVariant.RECOG:
        extra_config["chunking"] = "400:200"

    if variant == ConfigVariant.RECOG:
        extra_config["extern_data"] = {
            "data": {"dim": 3 * 50},
            "classes": {"dim": num_outputs, "sparse": True},
        }
    else:
        extra_config["__time_tag__"] = CodeWrapper('Dim(dimension=None, kind=Dim.Types.Spatial, description="time")')
        extra_config["extern_data"] = {
            "features_primary": {"dim": 50, "same_dim_tags_as": {"T": CodeWrapper("__time_tag__")}},
            "features_secondary": {"dim": 50, "same_dim_tags_as": {"T": CodeWrapper("__time_tag__")}},
            "features_mix": {"dim": 50, "same_dim_tags_as": {"T": CodeWrapper("__time_tag__")}},
            "classes": {"dim": 1, "dtype": "int32", "same_dim_tags_as": {"T": CodeWrapper("__time_tag__")}},
        }

    if kwargs.get("model_init", False):
        extra_config["preload_from_files"] = {}
        extra_config["preload_from_files"]["prim_encoder"] = {
            "filename": "/work/asr4/vieting/setups/converse/dependencies/librispeech_hybrid_conformer_training_job/output/models/epoch.600",
            "prefix": "prim_",
            "ignore_missing": True,
            "init_for_train": True,
        }
        extra_config["preload_from_files"]["mas_encoder"] = {
            "filename": "/work/asr4/vieting/setups/converse/dependencies/librispeech_hybrid_conformer_training_job/output/models/epoch.600",
            "prefix": "mas_",
            "ignore_missing": True,
            "init_for_train": True,
        }
        extra_config["preload_from_files"]["outputs"] = {
            "filename": "/work/asr4/vieting/setups/converse/dependencies/librispeech_hybrid_conformer_training_job/output/models/epoch.600",
            "ignore_missing": True,
            "init_for_train": True,
        }
        if not kwargs.get("emulate_single_speaker", False):
            extra_config["preload_from_files"]["sec_encoder"] = {
                "filename": "/work/asr4/vieting/setups/converse/dependencies/librispeech_hybrid_conformer_training_job/output/models/epoch.600",
                "prefix": "sec_",
                "ignore_missing": True,
                "init_for_train": True,
            }

    if variant == ConfigVariant.TRAIN or variant == ConfigVariant.PRIOR:
        net_dict, extra_python = conformer_hybrid_dualchannel.make_conformer_hybrid_dualchannel_model(
            specaug_args={
                "max_len_feature": 15,
                "max_len_time": 20,
                "max_reps_feature": 1,
                "max_reps_time": 20,
                "min_reps_feature": 0,
                "min_reps_time": 0,
            },
            use_secondary_audio=kwargs.get("sec_audio", False),
            use_comb_init=kwargs.get("use_comb_init", False),
            sep_comb_diag=kwargs.get("sep_comb_diag", 0.9),
            mix_comb_diag=kwargs.get("mix_comb_diag", 0.1),
            comb_noise=kwargs.get("comb_noise", 0.01),
            emulate_single_speaker=kwargs.get("emulate_single_speaker", False),
        )

    else:
        net_dict, extra_python = conformer_hybrid_dualchannel.make_conformer_hybrid_dualchannel_recog_model(
            use_secondary_audio=kwargs.get("sec_audio", False),
            emulate_single_speaker=kwargs.get("emulate_single_speaker", False),
        )

    return get_returnn_config(
        network=net_dict,
        num_epochs=kwargs.get("num_subepochs", 600),
        num_inputs=50,
        num_outputs=num_outputs,
        target="classes",
        python_prolog=[
            "import sys",
            "sys.setrecursionlimit(10 ** 6)",
            "from returnn.tf.util.data import Dim",
            "import numpy as np",
        ],
        extra_python=extra_python,
        extern_data_config=False,
        backend=Backend.TENSORFLOW,
        grad_noise=0.0,
        grad_clip=0.0,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=kwargs.get("lr", 1.1074e-5),
        peak_lr=kwargs.get("lr", 3e-04),
        final_lr=kwargs.get("lr", 1e-06),
        batch_size=12000 if variant == ConfigVariant.TRAIN else 4000,
        use_chunking=False,
        extra_config=extra_config,
    )


def get_returnn_config_collection(
    train_data_config: dict, dev_data_config: dict, **kwargs
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, **kwargs}
    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **generator_kwargs),
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **generator_kwargs)},
    )


# ********** Main exp function **********


def run_exp() -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path

    gmm_system = run_librispeech_960_common_baseline(recognition=False)

    data_per_lm = {}

    for lm_name in ["4gram", "kazuki_transformer"]:
        data_per_lm[lm_name] = get_hybrid_data(
            train_key="enhanced_tfgridnet_v1",
            dev_keys=["segmented_libri_css_tfgridnet_dev_v1", "segmented_libri_css_tfgridnet_eval_v1"]
            if lm_name == "4gram"
            else [],
            test_keys=["segmented_libri_css_tfgridnet_eval_v1"] if lm_name == "kazuki_transformer" else [],
            gmm_system=gmm_system,
            returnn_root=tools.returnn_root,
            returnn_python_exe=tools.returnn_python_exe,
            rasr_binary_path=tools.rasr_binary_path,
            add_unknown_phoneme_and_mapping=True,
            lm_name=lm_name,
        )

    data = copy.deepcopy(next(iter(data_per_lm.values())))
    data.dev_keys = []
    data.test_keys = []
    data.data_inputs = {}

    for lm_name in data_per_lm:
        data.dev_keys += [f"{key}_{lm_name}" for key in data_per_lm[lm_name].dev_keys]
        data.test_keys += [f"{key}_{lm_name}" for key in data_per_lm[lm_name].test_keys]
        data.data_inputs.update(
            {f"{key}_{lm_name}": data_input for key, data_input in data_per_lm[lm_name].data_inputs.items()}
        )

    for data_input in data.data_inputs.values():
        data_input.lexicon.filename = tk.Path(
            "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.JOqKFQpjp04H/output/oov.lexicon.gz"
        )
    for key in data.test_keys:
        data.data_inputs[key].lexicon.filename = tk.Path(
            "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/lexicon/modification/MergeLexiconJob.z54fVoMlr0md/output/lexicon.xml.gz"
        )

    # ********** Step args **********

    train_args = exp_args.get_hybrid_train_step_args(num_epochs=600, gpu_mem_rqmt=24)
    dev_recog_args = [
        exp_args.get_hybrid_recog_step_args(
            num_classes=num_outputs,
            epochs=[600],  # [20, 40, 80, 160, 240, 320, 400, 480, 560, 600],
            prior_scales=[0.3],
            pronunciation_scales=[6.0],
            lm_scales=[10.0],
            feature_type=FeatureType.CONCAT_GAMMATONE,
            lattice_processing_type=LatticeProcessingType.MultiChannelMultiSegment,
            search_parameters={"word-end-pruning-limit": 15000},
            mem=16,
            rtf=50,
        ),
        exp_args.get_hybrid_recog_step_args(
            num_classes=num_outputs,
            epochs=[600],  # [20, 40, 80, 160, 240, 320, 400, 480, 560, 600],
            prior_scales=[0.8],
            pronunciation_scales=[6.0],
            lm_scales=[11.0],
            feature_type=FeatureType.CONCAT_GAMMATONE,
            lattice_processing_type=LatticeProcessingType.MultiChannelMultiSegment,
            search_parameters={"word-end-pruning-limit": 15000},
            mem=16,
            rtf=50,
        ),
    ]
    test_recog_args = exp_args.get_hybrid_recog_step_args(
        num_classes=num_outputs,
        epochs=[600],  # [20, 40, 80, 160, 240, 320, 400, 480, 560, 600],
        prior_scales=[0.2],
        pronunciation_scales=[2.0],
        lm_scales=[9.3],
        search_parameters=get_atr_search_parameters(bp=16.0, bpl=100_000, wep=0.5, wepl=25_000),
        feature_type=FeatureType.CONCAT_GAMMATONE,
        lattice_processing_type=LatticeProcessingType.MultiChannelMultiSegment,
        use_gpu=True,
        mem=16,
        rtf=50,
    )

    # ********** System **********

    system = ReturnnLegacySystem(
        tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.CORPUS,
            SummaryKey.EPOCH,
            SummaryKey.PRIOR,
            SummaryKey.LM,
            SummaryKey.WER,
            SummaryKey.SUB,
            SummaryKey.INS,
            SummaryKey.DEL,
        ],
    )

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.get_hybrid_am_args(
            # cart_file=gmm_system.outputs["train-other-960"]["final"].crp.acoustic_model_config.state_tying.file
            cart_file=tk.Path("/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.tree.xml.gz"),
        ),
    )
    system.setup_scoring(
        scorer_type=MeetEvalJob,
        score_kwargs={
            "meet_eval_exe": tk.Path(
                "/work/asr4/vieting/programs/conda/20230126/anaconda3/envs/py310_upb/bin/python3",
                hash_overwrite="MEET_EVAL_EXE",
            )
        },
        stm_paths={
            **{
                key: tk.Path(
                    "/work/asr4/vieting/setups/converse/data/ref_libri_css_dev.stm",
                    hash_overwrite="libri_css_stm_dev",
                )
                for key in data.dev_keys + data.test_keys
                if "dev" in key
            },
            **{
                key: tk.Path(
                    "/work/asr4/vieting/setups/converse/data/ref_libri_css_test.stm",
                    hash_overwrite="libri_css_stm_test",
                )
                for key in data.dev_keys + data.test_keys
                if "eval" in key
            },
        },
    )

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        # f"tfgridnet_6prim_6mix_6mas_init_sep-{sep_diag:.1f}_mix-{mix_diag:.1f}_noise-{noise}",
        "tfgridnet_6prim_6mix_6mas_init",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
            sec_audio=False,
            model_init=True,
            num_subepochs=300,
            use_comb_init=True,
            sep_comb_diag=0.9,
            mix_comb_diag=0.1,
            comb_noise=0.01,
        ),
        custom_step_kwargs=CustomStepKwargs(
            train_step_kwargs=exp_args.get_hybrid_train_step_args(num_epochs=300, gpu_mem_rqmt=24),
            dev_recog_step_kwargs={
                "epochs": [20],
            },
            test_recog_step_kwargs={
                "epochs": [20],
                "prior_scales": [0.2, 0.3],
                "lm_scales": [9.3, 13.0],
            },
        ),
    )

    system.add_experiment_configs(
        "tfgridnet_6prim_6mix_6mas_init_notrain",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
            sec_audio=False,
            model_init=True,
            num_subepochs=1,
            use_comb_init=True,
            sep_comb_diag=0.9,
            mix_comb_diag=0.1,
            lr=0.0,
            comb_noise=0.01,
        ),
        custom_step_kwargs=CustomStepKwargs(
            train_step_kwargs=exp_args.get_hybrid_train_step_args(num_epochs=1, gpu_mem_rqmt=24),
            dev_recog_step_kwargs={
                "epochs": [1],
            },
            test_recog_step_kwargs={
                "epochs": [],
            },
        ),
    )

    # system.add_experiment_configs(
    #     "tfgridnet_6prim_6sec_6mas_init",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         sec_audio=True,
    #         model_init=True,
    #         use_prim_identity_init=True,
    #     ),
    # )

    system.add_experiment_configs(
        "tfgridnet_6prim_6mix_6mas_scratch",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
            sec_audio=False,
            model_init=False,
            custom_step_kwargs=CustomStepKwargs(
                train_step_kwargs=exp_args.get_hybrid_train_step_args(num_epochs=1, gpu_mem_rqmt=24),
                dev_recog_step_kwargs={
                    "epochs": [480, 560, 600],
                },
                test_recog_step_kwargs={
                    "epochs": [],
                },
            ),
        ),
    )

    # system.add_experiment_configs(
    #     "tfgridnet_6prim_6sec_6mas_scratch",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         sec_audio=True,
    #         model_init=False,
    #         use_prim_identity_init=False,
    #     ),
    # )

    # system.add_experiment_configs(
    #     "tfgridnet_12prim_init",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         sec_audio=False,
    #         model_init=True,
    #         emulate_single_speaker=True,
    #     ),
    #     custom_step_kwargs=CustomStepKwargs(train_step_kwargs={"gpu_mem_rqmt": 11}),
    # )

    system.add_experiment_configs(
        "tfgridnet_12prim_init_short",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
            sec_audio=False,
            model_init=True,
            emulate_single_speaker=True,
            num_subepochs=300,
        ),
        custom_step_kwargs=CustomStepKwargs(
            train_step_kwargs=exp_args.get_hybrid_train_step_args(num_epochs=300, gpu_mem_rqmt=11),
            dev_recog_step_kwargs={"epochs": [20]},
            test_recog_step_kwargs={
                "epochs": [20],
                "prior_scales": [0.3],
                "lm_scales": [13.0],
            },
        ),
    )

    # system.add_experiment_configs(
    #     "tfgridnet_12prim_scratch",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         sec_audio=False,
    #         model_init=False,
    #         emulate_single_speaker=True,
    #     ),
    #     custom_step_kwargs=CustomStepKwargs(train_step_kwargs={"gpu_mem_rqmt": 11}),
    # )

    system.run_train_step(**train_args)
    for recog_args in dev_recog_args:
        system.run_dev_recog_step(**recog_args)

    system.run_test_recog_step(**test_recog_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
