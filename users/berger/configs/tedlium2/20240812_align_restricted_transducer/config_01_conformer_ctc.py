import copy
import os
from typing import Dict, Tuple

import i6_core.rasr as rasr
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import Backend, get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.tedlium2.ctc_data import get_tedlium2_data_dumped_labels
from i6_experiments.users.berger.pytorch.models import conformer_ctc_minireturnn as conformer_ctc
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import (
    AlignmentData,
    ConfigVariant,
    FeatureType,
    ReturnnConfigs,
    SummaryKey,
)
from i6_experiments.users.berger.systems.functors.recognition.returnn_search import LexiconType, LmType, VocabType
from i6_experiments.users.berger.systems.returnn_native_system import ReturnnNativeSystem
from i6_experiments.users.berger.util import default_tools_v2
from sisyphus import gs, tk

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 1000
sub_checkpoints = [100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 960, 970, 980, 990, 1000]

tools = copy.deepcopy(default_tools_v2)
tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard")
assert tools.returnn_root is not None
normal_returnn = tools.returnn_root
tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")


# ********** Return Config generators **********


def returnn_config_generator(
    variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, **kwargs
) -> ReturnnConfig:
    model_config = conformer_ctc.get_default_config_v1(num_outputs)

    if variant == ConfigVariant.TRAIN:
        extra_config: dict = {
            "train": train_data_config,
            "dev": dev_data_config,
            "extern_data": {"data": {"dim": 1}, "classes": {"dim": num_outputs, "sparse": True}},
            "torch_amp": {"dtype": "bfloat16"},
            "num_workers_per_gpu": 2,
        }
    if variant == ConfigVariant.PRIOR:
        extra_config: dict = {
            "forward": train_data_config,
            "extern_data": {"forward": {"dim": 1}},
            "torch_amp": {"dtype": "bfloat16"},
        }
    if variant == ConfigVariant.RECOG:
        extra_config: dict = {
            "extern_data": {"data": {"dim": 1}},
        }

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="classes",
        extra_python=[
            conformer_ctc.get_serializer(model_config, variant=variant, recog_type=conformer_ctc.RecogType.FLASHLIGHT)
        ],
        extern_data_config=False,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=1.0,
        optimizer=Optimizers.AdamW,
        weight_decay=0.01,
        keep_last_n=1,
        keep_best_n=1,
        keep=sub_checkpoints,
        schedule=LearningRateSchedules.OCLR_V2,
        inc_epochs=480,
        max_seqs=60,
        initial_lr=7e-06,
        peak_lr=5e-04,
        decayed_lr=5e-05,
        final_lr=1e-07,
        batch_size=36000 * 160,
        use_chunking=False,
        extra_config=extra_config,
    )


def get_returnn_config_collection(
    train_data_config: dict,
    dev_data_config: dict,
    num_subepochs: int,
    **kwargs,
) -> ReturnnConfigs[ReturnnConfig]:
    return ReturnnConfigs(
        train_config=returnn_config_generator(
            variant=ConfigVariant.TRAIN,
            train_data_config=train_data_config,
            dev_data_config=dev_data_config,
            num_subepochs=num_subepochs,
            **kwargs,
        ),
        prior_config=returnn_config_generator(
            variant=ConfigVariant.PRIOR,
            train_data_config=train_data_config,
            dev_data_config=dev_data_config,
            num_subepochs=num_subepochs,
            **kwargs,
        ),
        recog_configs={
            "recog": returnn_config_generator(
                variant=ConfigVariant.RECOG,
                train_data_config=train_data_config,
                dev_data_config=dev_data_config,
                num_subepochs=num_subepochs,
                **kwargs,
            )
        },
    )


def run_exp() -> Tuple[SummaryReport, Dict[str, AlignmentData]]:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_tedlium2_data_dumped_labels(
        num_classes=num_outputs,
        returnn_root=normal_returnn,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
        partition_epoch=10,
    )

    for data_input in data.data_inputs.values():
        data_input.create_lm_images(tools.rasr_binary_path)

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs)
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=sub_checkpoints,
        prior_scales=[0.3],
        lm_scales=[0.9],
        feature_type=FeatureType.SAMPLES,
        lexicon_type=LexiconType.FLASHLIGHT,
        lm_type=LmType.ARPA_FILE,
        vocab_type=VocabType.LEXICON_INVENTORY,
        search_stats=True,
    )
    align_args = exp_args.get_ctc_align_step_args(
        num_classes=num_outputs,
        feature_type=FeatureType.LOGMEL_16K,
        prior_scale=0.3,
        reduction_factor=4,
        label_scorer_args={"extra_args": {"reduction_subtrahend": 3}},
        epoch=num_subepochs,
        register_output=True,
    )

    # ********** System **********

    system = ReturnnNativeSystem(
        tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.RECOG_NAME,
            SummaryKey.CORPUS,
            SummaryKey.EPOCH,
            SummaryKey.LM,
            SummaryKey.PRIOR,
            SummaryKey.WER,
            SummaryKey.SUB,
            SummaryKey.DEL,
            SummaryKey.INS,
            SummaryKey.ERR,
            SummaryKey.RTF,
        ],
        summary_sort_keys=[SummaryKey.CORPUS, SummaryKey.ERR],
    )

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        "Conformer_CTC",
        get_returnn_config_collection(
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
            num_subepochs=num_subepochs,
        ),
    )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    # align_data = next(iter(system.run_align_step(**align_args).values()))
    align_data = None

    for train_job in system._train_jobs.values():
        train_job.update_rqmt(task_name="run", rqmt={"gpu_mem": 24})

    assert system.summary_report
    return system.summary_report, align_data


def py() -> Tuple[SummaryReport, Dict[str, AlignmentData]]:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    report, align_data = run_exp()
    summary_report.merge_report(report, update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, align_data
