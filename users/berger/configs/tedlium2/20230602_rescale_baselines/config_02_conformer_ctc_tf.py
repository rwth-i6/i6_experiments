import copy
import os

import i6_core.rasr as rasr
from i6_core.returnn.config import ReturnnConfig
import i6_experiments.users.berger.network.models.fullsum_ctc as ctc_model
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import Backend, get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules
from i6_experiments.users.berger.corpus.tedlium2.ctc_data import get_tedlium2_tf_data
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.systems.returnn_seq2seq_system import ReturnnSeq2SeqSystem
from i6_experiments.users.berger.util import default_tools_v2
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from sisyphus import gs, tk

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 250

tools = copy.deepcopy(default_tools_v2)

# tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/onnx/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")


# ********** Return Config generators **********


def returnn_config_generator(
    variant: ConfigVariant,
    train_data_config: dict,
    dev_data_config: dict,
    loss_corpus: tk.Path,
    loss_lexicon: tk.Path,
    am_args: dict,
) -> ReturnnConfig:
    if variant == ConfigVariant.TRAIN or variant == ConfigVariant.PRIOR:
        net_dict, extra_python = ctc_model.make_conformer_fullsum_ctc_model(
            num_outputs=num_outputs,
            specaug_args={
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 5,
            },
            conformer_args={
                "l2": 0.0001,
            },
            output_args={
                "rasr_binary_path": tools.rasr_binary_path,
                "loss_corpus_path": loss_corpus,
                "loss_lexicon_path": loss_lexicon,
                "remove_prefix": "",
                "am_args": am_args,
            },
        )
    else:
        net_dict, extra_python = ctc_model.make_conformer_ctc_recog_model(
            num_outputs=num_outputs,
        )

    returnn_config = get_returnn_config(
        network=net_dict,
        num_epochs=num_subepochs,
        num_inputs=50,
        num_outputs=num_outputs,
        target=None,
        python_prolog=[
            "import sys",
            "sys.setrecursionlimit(10 ** 6)",
        ],
        extra_python=extra_python,
        extern_data_config=True,
        # extern_data_kwargs={"dtype": "int16" if train else "float32"},
        backend=Backend.TENSORFLOW,
        grad_noise=0.0,
        grad_clip=0.0,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=1e-05,
        peak_lr=4e-04,
        cycle_epoch=100,
        final_lr=1e-05,
        batch_size=10000,
        use_chunking=False,
        extra_config={
            "train": train_data_config,
            "dev": dev_data_config,
        },
    )

    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def get_returnn_config_collection(
    train_data_config: dict,
    dev_data_config: dict,
    loss_corpus: tk.Path,
    loss_lexicon: tk.Path,
) -> ReturnnConfigs[ReturnnConfig]:
    config_generator_kwargs = {
        "train_data_config": train_data_config,
        "dev_data_config": dev_data_config,
        "loss_corpus": loss_corpus,
        "loss_lexicon": loss_lexicon,
        "am_args": exp_args.ctc_loss_am_args,
    }

    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **config_generator_kwargs),
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **config_generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **config_generator_kwargs)},
    )


def run_exp() -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_tedlium2_tf_data(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.GAMMATONE_16K,
    )

    # ********** Step args **********

    train_step_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs)
    recog_step_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[160, 240, 250],
        prior_scales=[0.3, 0.5, 0.9],
        lm_scales=[0.7, 1.1, 1.4, 2.0],
        feature_type=FeatureType.GAMMATONE_16K,
    )

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    # tools.rasr_binary_path = tk.Path(
    #     "/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    # )
    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        "Conformer_CTC_TF",
        get_returnn_config_collection(
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
            loss_corpus=data.loss_corpus,
            loss_lexicon=data.loss_lexicon,
        ),
    )

    system.run_train_step(**train_step_args)
    system.run_dev_recog_step(**recog_step_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
