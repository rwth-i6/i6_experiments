import dataclasses
import os

from typing import Optional, Dict

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.users.berger.args.experiments import hybrid as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.corpus import sms_librispeech as ls_corpus
from i6_experiments.users.berger.network.models import (
    blstm_hybrid_dual_output as net_construct,
)
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import (
    DualSpeakerReturnnConfig,
    ReturnnConfigs,
)
from i6_experiments.users.berger.systems.dual_speaker_returnn_legacy_system import (
    DualSpeakerReturnnLegacySystem,
)
from i6_experiments.users.berger.util import default_tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from returnn.tf.util.data import batch_dim
from .config_01_blstm_hybrid_mixed_inputs import py as py_pretrain

# ********** Settings **********

# rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

# rasr_binary_path = tk.Path("/u/berger/rasr_tf2/arch/linux-x86_64-standard")
returnn_root = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/returnn",
    commit="f02a6b7cdf4417e66cc1a9307d24462f3cfaee6f",
    checkout_folder_name="returnn",
).out_repository
tools = dataclasses.replace(default_tools, returnn_root=returnn_root)
pytorch_to_returnn_root = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/pytorch-to-returnn",
    commit="6790829498373bbff4d9503038e2ef617bbe8ef6",
    checkout_folder_name="pytorch-to-returnn",
).out_repository
padertorch_root = tk.Path("/work/asr4/vieting/programs/padertorch/20230201/padertorch/")

num_inputs = 50
num_classes = 12001

train_key = "train_960"
cv_key = "dev_clean"
dev_key = "dev_clean"
test_key = "test_clean"

recog_dev_keys = "sms_cv_dev93_mixed"
recog_test_keys = "sms_test_eval92_mixed"

scoring_dev_keys = "sms_cv_dev93"
scoring_test_keys = "sms_test_eval92"

am_args = {
    "state_tying": "cart",
    "state_tying_file": tk.Path("/work/asr4/berger/dependencies/librispeech/cart/cart-12001.xml.gz"),
}

assert tools.returnn_python_exe is not None
assert tools.returnn_root is not None
speech_sep_net_dict, speech_sep_checkpoint = ls_corpus.get_separator(
    python_exe=tools.returnn_python_exe,
    pytorch_to_returnn_root=pytorch_to_returnn_root,
    returnn_root=tools.returnn_root,
    padertorch_root=padertorch_root,
)


def speaker_returnn_config_generator(
    train: bool,
    speaker_idx: int,
    *,
    train_data_config: dict,
    cv_data_config: dict,
    python_prolog: dict,
    am_checkpoint: Optional[tk.Path] = None,
    **kwargs,
) -> ReturnnConfig:
    num_01_layers = kwargs.get("num_01_layers", 4)
    num_mix_layers = kwargs.get("num_mix_layers", 4)
    num_01_mix_layers = kwargs.get("num_01_mix_layers", 2)
    num_combine_layers = kwargs.get("num_combine_layers", 1)
    num_context_layers = kwargs.get("num_context_layers", 0)
    if num_combine_layers:
        if train:
            (net, python_code, dim_tags,) = net_construct.make_blstm_hybrid_dual_output_combine_enc_model(
                num_outputs=num_classes,
                gt_args={
                    "sample_rate": 16000,
                    "specaug_after_dct": True,
                },
                blstm_01_args={"num_layers": num_01_layers, "size": 400},
                blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
                blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
                blstm_combine_args={"num_layers": num_combine_layers, "size": 800},
                aux_loss_01_layers=[(num_01_layers, 0.3)] if num_mix_layers else [],
                aux_loss_01_mix_layers=[(num_01_mix_layers, 0.3)] if num_01_mix_layers else [],
                freeze_separator=False,
            )
        else:
            (net, python_code, dim_tags,) = net_construct.make_blstm_hybrid_dual_output_combine_enc_recog_model(
                num_outputs=num_classes,
                speaker_idx=speaker_idx,
                gt_args={"sample_rate": 16000},
                blstm_01_args={"num_layers": num_01_layers, "size": 400},
                blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
                blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
                blstm_combine_args={"num_layers": num_combine_layers, "size": 800},
            )
    elif num_context_layers:
        if train:
            (net, python_code, dim_tags,) = net_construct.make_blstm_hybrid_dual_output_soft_context_model(
                num_outputs=num_classes,
                gt_args={
                    "sample_rate": 16000,
                    "specaug_after_dct": True,
                },
                blstm_01_args={"num_layers": num_01_layers, "size": 400},
                blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
                blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
                blstm_context_args={"num_layers": num_context_layers, "size": 400},
                aux_loss_01_layers=[(num_01_layers, 0.3)] if num_mix_layers else [],
                aux_loss_01_mix_layers=[],
                pre_context_loss_scale=0.3,
                use_logits=True,
                freeze_separator=False,
            )
        else:
            (net, python_code, dim_tags,) = net_construct.make_blstm_hybrid_dual_output_soft_context_recog_model(
                num_outputs=num_classes,
                speaker_idx=speaker_idx,
                gt_args={"sample_rate": 16000},
                blstm_01_args={"num_layers": num_01_layers, "size": 400},
                blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
                blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
                blstm_context_args={"num_layers": num_context_layers, "size": 400},
                use_logits=kwargs.get("use_logits", True),
            )
    else:
        if train:
            (net, python_code, dim_tags,) = net_construct.make_blstm_hybrid_dual_output_model(
                num_outputs=num_classes,
                gt_args={
                    "sample_rate": 16000,
                    "specaug_after_dct": True,
                },
                blstm_01_args={"num_layers": num_01_layers, "size": 400},
                blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
                blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
                aux_loss_01_layers=[(num_01_layers, 0.3)] if num_01_layers and num_01_mix_layers else [],
                aux_loss_01_mix_layers=[],
                freeze_separator=False,
            )
        else:
            (net, python_code, dim_tags,) = net_construct.make_blstm_hybrid_dual_output_recog_model(
                num_outputs=num_classes,
                speaker_idx=speaker_idx,
                gt_args={"sample_rate": 16000},
                blstm_01_args={"num_layers": num_01_layers, "size": 400},
                blstm_mix_args={"num_layers": num_mix_layers, "size": 400},
                blstm_01_mix_args={"num_layers": num_01_mix_layers, "size": 400},
            )

    extern_data_config = {
        "data": {
            "dim": 1,
            "dim_tags": [
                batch_dim,
                dim_tags["waveform_time"],
                dim_tags["waveform_feature"],
            ],
        },
    }
    if train:
        extern_data_config.update(
            {
                "target_signals": {
                    "dim": 2,
                    "shape": (None, 2),
                    "dim_tags": [
                        batch_dim,
                        dim_tags["waveform_time"],
                        dim_tags["speaker"],
                    ],
                },
                "target_classes": {
                    "dim": num_classes,
                    "sparse": True,
                    "shape": (None, 2),
                    "dim_tags": [
                        batch_dim,
                        dim_tags["target_time"],
                        dim_tags["speaker"],
                    ],
                },
            }
        )

    model_preload = {
        "mask_estimator": {
            "filename": speech_sep_checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
            "prefix": "speech_separator/",
        },
    }

    if am_checkpoint is None:
        model_preload = {
            "mask_estimator": {
                "filename": speech_sep_checkpoint,
                "init_for_train": True,
                "ignore_missing": True,
                "prefix": "speech_separator/",
            },
        }
    else:
        model_preload = {
            "am": {
                "filename": am_checkpoint,
                "init_for_train": True,
                "ignore_missing": True,
            },
        }

    extra_config = {
        "train": train_data_config,
        "dev": cv_data_config,
        "preload_from_files": model_preload,
        "extern_data": extern_data_config,
    }
    if kwargs.get("chunking", True):
        extra_config["chunking"] = (
            {
                "data": 128 * 160,
                "target_signals": 128 * 160,
                "target_classes": 128,
            },
            {
                "data": 64 * 160,
                "target_signals": 64 * 160,
                "target_classes": 64,
            },
        )

    returnn_config = get_returnn_config(
        net,
        target="target_classes" if train else None,
        num_inputs=num_inputs,
        num_outputs=num_classes,
        num_epochs=80,
        batch_size=kwargs.get("batch_size", 1_600_000),
        accum_grad=kwargs.get("accum_grad", 1),
        schedule=kwargs.get("schedule", LearningRateSchedules.Newbob),
        learning_rate=3e-05,
        min_learning_rate=1e-06,
        use_chunking=kwargs.get("chunking", True),
        python_prolog={**python_prolog, "net_python": python_code},
        extra_config=extra_config,
    )

    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def returnn_config_generator(
    train: bool,
    **kwargs,
) -> DualSpeakerReturnnConfig:
    if train:
        return DualSpeakerReturnnConfig(speaker_returnn_config_generator(train, 0, **kwargs))
    return DualSpeakerReturnnConfig(
        speaker_returnn_config_generator(train, 0, **kwargs),
        speaker_returnn_config_generator(train, 1, **kwargs),
    )


def run_exp(am_checkpoints: Dict[str, tk.Path]) -> SummaryReport:
    sms_data = ls_corpus.get_sms_data()

    system = DualSpeakerReturnnLegacySystem(tool_paths=tools)
    system.init_corpora(
        dev_keys=sms_data.dev_keys,
        test_keys=sms_data.test_keys,
        corpus_data=sms_data.data_inputs,
        am_args=am_args,
    )
    system.setup_scoring(scoring_corpora=sms_data.scoring_corpora, score_kwargs={"sort_files": True})

    base_config_kwargs = {
        "train_data_config": sms_data.train_data_config,
        "cv_data_config": sms_data.cv_data_config,
        "python_prolog": sms_data.python_prolog,
    }

    for exp_name, exp_config in [
        (
            "modular",
            {
                "num_01_layers": 6,
                "num_mix_layers": 0,
                "num_01_mix_layers": 0,
                "chunking": False,
            },
        ),
        (
            "full_enc_structure",
            {
                "num_01_layers": 6,
                "num_mix_layers": 4,
                "num_01_mix_layers": 1,
                "num_combine_layers": 1,
                "chunking": False,
                "batch_size": 1_600_000,
            },
        ),
    ]:
        train_returnn_config = returnn_config_generator(
            train=True,
            am_checkpoint=am_checkpoints[f"blstm_hybrid_{exp_name}"],
            **exp_config,
            **base_config_kwargs,
        )
        recog_returnn_config = returnn_config_generator(train=False, **exp_config, **base_config_kwargs)
        returnn_configs = ReturnnConfigs(
            train_config=train_returnn_config,
            recog_configs={"recog": recog_returnn_config},
        )
        system.add_experiment_configs(f"blstm_hybrid_{exp_name}_joint", returnn_configs)

    train_args = exp_args.get_hybrid_train_step_args(
        num_epochs=80,
        log_verbosity=5,
        gpu_mem_rqmt=24,
        mem_rqmt=16,
    )
    recog_args = exp_args.get_hybrid_recog_step_args(num_classes=sms_data.num_classes, epochs=[40, 80, "best"])

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    return system.summary_report


def py() -> SummaryReport:
    _, model_checkpoints = py_pretrain()

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = run_exp(model_checkpoints)

    tk.register_report(
        f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report",
        summary_report,
    )
    return summary_report
