import copy
import os
import torch
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.baselines.librispeech.ls960.gmm.baseline_config import (
    run_librispeech_960_common_baseline,
)
from i6_experiments.users.berger.systems.returnn_legacy_system import (
    ReturnnLegacySystem,
)
from i6_experiments.users.berger.systems.functors.rasr_base import (
    LatticeProcessingType,
)

from sisyphus import gs, tk

import i6_core.rasr as rasr
from i6_experiments.users.berger.args.experiments import hybrid as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.corpus.libri_css.hybrid_data import get_hybrid_data
from i6_experiments.users.berger.pytorch.models import conformer_hybrid_dualspeaker
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.recipe.converse.scoring import MeetEvalJob
from i6_experiments.users.berger.systems.dataclasses import (
    ConfigVariant,
    CustomStepKwargs,
    FeatureType,
    ReturnnConfigs,
)
from i6_experiments.users.berger.util import default_tools_v2

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_inputs = 50
num_outputs = 12001
num_subepochs = 400

tools = copy.deepcopy(default_tools_v2)

tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")


def worker_wrapper(job, task_name, call):
    torch_jobs = {
        "ReturnnTrainingJob",
        "ReturnnRasrTrainingJob",
        "OptunaReturnnTrainingJob",
        "CompileTFGraphJob",
        "OptunaCompileTFGraphJob",
        "ReturnnRasrComputePriorJob",
        "ReturnnComputePriorJob",
        "ReturnnComputePriorJobV2",
        "OptunaReturnnComputePriorJob",
        "ReturnnForwardJob",
        "ReturnnForwardComputePriorJob",
        "OptunaReturnnForwardComputePriorJob",
        "TorchOnnxExportJob",
        "ExportPyTorchModelToOnnxJob",
        "OptunaExportPyTorchModelToOnnxJob",
    }
    rasr_jobs = {
        "MakeJob",
        "CompileNativeOpJob",
        "AdvancedTreeSearchJob",
        "AdvancedTreeSearchLmImageAndGlobalCacheJob",
        "GenericSeq2SeqSearchJob",
        "GenericSeq2SeqLmImageAndGlobalCacheJob",
        "AlignmentJob",
        "Seq2SeqAlignmentJob",
        "EstimateMixturesJob",
        "EstimateCMLLRJob",
        "OptimizeAMandLMScaleJob",
        "LatticeToCtmJob",
    }

    if type(job).__name__ in torch_jobs:
        image = "/work/asr4/berger/apptainer/images/i6_u22_pytorch1.13_onnx.sif"
    elif type(job).__name__ in rasr_jobs:
        image = "/work/asr4/berger/apptainer/images/i6_tensorflow-2.8_onnx-1.15.sif"
    else:
        return call
    binds = ["/work/asr4", "/work/asr3", "/work/common", "/work/tools/"]
    ts = {t.name(): t for t in job.tasks()}
    t = ts[task_name]

    app_call = [
        "apptainer",
        "exec",
    ]
    if t._rqmt.get("gpu", 0) > 0:
        app_call += ["--nv"]

    for path in binds:
        app_call += ["-B", path]

    app_call += [
        image,
        "python3",
    ]

    app_call += call[1:]

    return app_call


gs.worker_wrapper = worker_wrapper

# ********** Return Config generators **********


def returnn_config_generator(
    variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, **kwargs
) -> ReturnnConfig:
    model_config = conformer_hybrid_dualspeaker.get_default_config_v1(num_inputs=num_inputs, num_outputs=num_outputs)

    model_config.primary_frontend.cfg.out_features = 384
    model_config.conformer_block_cfg.ff_cfg.input_dim = 384
    model_config.conformer_block_cfg.ff_cfg.hidden_dim = 384
    model_config.conformer_block_cfg.conv_cfg.channels = 384
    model_config.conformer_block_cfg.conv_cfg.norm = torch.nn.BatchNorm1d(num_features=384, affine=True)
    model_config.conformer_block_cfg.mhsa_cfg.input_dim = 384
    model_config.conformer_block_cfg.mhsa_cfg.num_att_heads = 6
    model_config.num_primary_layers = kwargs.get("prim_blocks", 6)
    model_config.num_secondary_layers = kwargs.get("sec_blocks", 6)
    model_config.num_mixture_aware_speaker_layers = kwargs.get("mas_blocks", 6)
    model_config.use_secondary_audio = kwargs.get("sec_audio", True)

    extra_config: dict = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.RECOG:
        extern_data_config = {
            "features_primary": {"dim": num_inputs * 2 if kwargs.get("sec_audio", True) else num_inputs},
            "classes": {"dim": 1},
        }
        extra_config["model_outputs"] = {"log_probs": {"dim": num_outputs}}
    else:
        extern_data_config = {
            "features_primary": {"dim": num_inputs},
            "classes": {"dim": 1},
        }
        if kwargs.get("sec_audio", True):
            extern_data_config["features_mix"] = {"dim": num_inputs}
        extra_config["chunking"] = "400:200"

    extra_config["extern_data"] = extern_data_config

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        target="classes",
        extra_python=[conformer_hybrid_dualspeaker.get_serializer(model_config, variant=variant)],
        extern_data_config=False,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=0.0,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=1e-05,
        peak_lr=3e-04,
        final_lr=1e-06,
        # batch_size=6144,
        batch_size=6000 if variant == ConfigVariant.TRAIN else 1000,
        accum_grad=2,
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


def run_exp() -> SummaryReport:
    gmm_system = run_librispeech_960_common_baseline(recognition=False)

    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path

    data_per_lm = {}

    for lm_name in ["4gram", "kazuki_transformer"]:
        data_per_lm[lm_name] = get_hybrid_data(
            train_key="enhanced_tfgridnet_v1",
            dev_keys=(
                ["segmented_libri_css_tfgridnet_dev_v1", "segmented_libri_css_tfgridnet_eval_v1"]
                if lm_name == "4gram"
                else []
            ),
            test_keys=["segmented_libri_css_tfgridnet_eval_v1"] if lm_name == "kazuki_transformer" else [],
            gmm_system=gmm_system,
            returnn_root=tools.returnn_root,
            returnn_python_exe=tools.returnn_python_exe,
            rasr_binary_path=tools.rasr_binary_path,
            add_unknown_phoneme_and_mapping=True,
            lm_name=lm_name,
            add_sec_audio=False,
            add_mix_audio=True,
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

    train_args = exp_args.get_hybrid_train_step_args(num_epochs=num_subepochs)  # , gpu_mem_rqmt=24, mem_rqmt=24)
    recog_args = exp_args.get_hybrid_recog_step_args(
        num_classes=num_outputs,
        epochs=[240, 320, 400, "best"],
        prior_scales=[0.8],
        pronunciation_scales=[6.0],
        lm_scales=[11.0],
        feature_type=FeatureType.CONCAT_MIX_GAMMATONE_16K,
        lattice_processing_type=LatticeProcessingType.MultiChannelMultiSegment,
        mem=16,
        rtf=50,
        model_flow_args={"features_name": "features_primary", "features_size_name": "features_primary:size1"},
    )

    # ********** System **********

    system = ReturnnLegacySystem(tools)

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
        "pt_tfgridnet_conformer_6prim_6mix_6mas",
        get_returnn_config_collection(
            data.train_data_config, data.cv_data_config, prim_blocks=6, sec_blocks=6, mas_blocks=6, sec_audio=True
        ),
    )

    system.add_experiment_configs(
        "pt_tfgridnet_conformer_12prim",
        get_returnn_config_collection(
            data.train_data_config, data.cv_data_config, prim_blocks=12, sec_blocks=0, mas_blocks=0, sec_audio=False
        ),
        custom_step_kwargs=CustomStepKwargs(
            dev_recog_step_kwargs={"feature_type": FeatureType.GAMMATONE_16K},
            test_recog_step_kwargs={"feature_type": FeatureType.GAMMATONE_16K},
        ),
    )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
