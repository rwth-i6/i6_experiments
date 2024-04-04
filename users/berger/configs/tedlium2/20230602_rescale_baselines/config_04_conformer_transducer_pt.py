import copy
import os
from typing import List, Optional
from i6_core.returnn.config import ReturnnConfig

from sisyphus import gs, tk

import i6_core.rasr as rasr
from i6_experiments.users.berger.args.experiments import transducer as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.tedlium2.bpe_transducer_data import get_tedlium2_pytorch_data
from i6_experiments.users.berger.pytorch.models import conformer_transducer_torchaudio as model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.systems.returnn_native_system import (
    ReturnnNativeSystem,
)
from i6_experiments.users.berger.util import default_tools_v2

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 1069
num_subepochs = 200

tools = copy.deepcopy(default_tools_v2)
tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard")


# ********** Return Config generators **********


def returnn_config_generator(
    variant: ConfigVariant,
    train_data_config: dict,
    dev_data_config: dict,
    forward_data_config: Optional[dict] = None,
    k2: bool = True,
    pruned: bool = True,
    i6_config: bool = False,
    **kwargs,
) -> ReturnnConfig:
    if i6_config:
        model_config = model.get_i6_default_config_v1(num_inputs=50, num_outputs=num_outputs)
    else:
        model_config = model.get_torchaudio_default_config_v1(num_inputs=50, num_outputs=num_outputs)

    if variant == ConfigVariant.RECOG:
        assert forward_data_config is not None
        extra_config = {
            "forward_data": forward_data_config,
            "model_outputs": {
                "tokens": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                }
            },
        }
        serializer = model.get_beam_search_serializer(model_config, **kwargs)
    else:
        extra_config = {
            "train": train_data_config,
            "dev": dev_data_config,
        }
        if k2:
            if pruned:
                serializer = model.get_pruned_k2_train_serializer(model_config, **kwargs)
            else:
                serializer = model.get_k2_train_serializer(model_config, **kwargs)
        else:
            serializer = model.get_train_serializer(model_config, **kwargs)

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=50,
        num_outputs=num_outputs,
        target="targets" if variant != ConfigVariant.RECOG else None,
        extra_python=[serializer],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=0.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=1e-05,
        peak_lr=3e-04,
        final_lr=1e-06,
        batch_size=5000 * (1 + 1 * int(pruned)),
        accum_grad=6 // (1 + 1 * int(pruned)),
        use_chunking=False,
        extra_config=extra_config,
    )


def get_returnn_config_collection(
    train_data_config: dict,
    dev_data_config: dict,
    forward_data_config: dict,
    bpe_lexicon: tk.Path,
    k2: bool,
    pruned: bool,
    beam_sizes: List[int],
    **kwargs,
) -> ReturnnConfigs[ReturnnConfig]:
    return ReturnnConfigs(
        train_config=returnn_config_generator(
            variant=ConfigVariant.TRAIN,
            train_data_config=train_data_config,
            dev_data_config=dev_data_config,
            k2=k2,
            pruned=pruned,
            blank_id=0,
            **kwargs,
        ),
        recog_configs={
            f"recog_beam-{beam_size}": returnn_config_generator(
                variant=ConfigVariant.RECOG,
                train_data_config=train_data_config,
                dev_data_config=dev_data_config,
                forward_data_config=forward_data_config,
                lexicon_file=bpe_lexicon,
                beam_size=beam_size,
                blank_id=0,
                **kwargs,
            )
            for beam_size in beam_sizes
        },
    )


def run_exp() -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_tedlium2_pytorch_data(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.GAMMATONE_16K,
    )

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(num_epochs=num_subepochs)  # , gpu_mem_rqmt=24)
    recog_args = {
        "epochs": [20, 40, 80, 160, 200],
    }

    # ********** System **********

    system = ReturnnNativeSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.transducer_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    # system.add_experiment_configs(
    #     "Conformer_Transducer",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         data.forward_data_config["dev"],
    #         bpe_lexicon=data.bpe_lexicon,
    #         beam_sizes=[10],
    #         k2=False,
    #         pruned=False,
    #     ),
    # )

    system.add_experiment_configs(
        "Conformer_Transducer_k2",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
            data.forward_data_config["dev"],
            bpe_lexicon=data.bpe_lexicon,
            beam_sizes=[10],
            k2=True,
            pruned=False,
            rnnt_type="modified",
        ),
    )

    # system.add_experiment_configs(
    #     "Conformer_Transducer_k2-pruned",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         data.forward_data_config["dev"],
    #         bpe_lexicon=data.bpe_lexicon,
    #         beam_sizes=[10],
    #         k2=True,
    #         pruned=True,
    #         rnnt_type="modified",
    #     ),
    # )

    # system.add_experiment_configs(
    #     "Conformer_Transducer_i6-cfg",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         data.forward_data_config["dev"],
    #         bpe_lexicon=data.bpe_lexicon,
    #         beam_sizes=[10],
    #         k2=False,
    #         i6_config=True,
    #         pruned=False,
    #     ),
    # )

    # system.add_experiment_configs(
    #     "Conformer_Transducer_k2_i6-cfg",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         data.forward_data_config["dev"],
    #         bpe_lexicon=data.bpe_lexicon,
    #         beam_sizes=[10],
    #         k2=True,
    #         i6_config=True,
    #         pruned=False,
    #         rnnt_type="modified",
    #     ),
    # )

    # system.add_experiment_configs(
    #     "Conformer_Transducer_k2-pruned_i6-cfg",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         data.forward_data_config["dev"],
    #         bpe_lexicon=data.bpe_lexicon,
    #         beam_sizes=[10],
    #         k2=True,
    #         i6_config=True,
    #         pruned=True,
    #         rnnt_type="modified",
    #     ),
    # )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
