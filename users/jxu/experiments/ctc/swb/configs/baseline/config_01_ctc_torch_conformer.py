import copy
from typing import Dict, Tuple

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_core.returnn import Checkpoint
from i6_core.recognition import Hub5ScoreJob
from i6_experiments.common.tools.sctk import compile_sctk

from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType, ReturnnConfigs, ConfigVariant

from i6_experiments.users.berger.recipe.summary.report import SummaryReport
import i6_experiments.users.jxu.experiments.ctc.swb.pytorch_networks.baseline.conformer_ctc_d_model_386_num_layers_12_logmel as conformer_ctc
from i6_experiments.users.jxu.experiments.ctc.swb.data.ctc_data import get_switchboard_data
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from sisyphus import gs, tk


# ********** Settings **********
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_classes = 88
num_subepochs = 600

tools = copy.deepcopy(default_tools_v2)
tools.rasr_binary_path = tk.Path(
    "/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard",
    hash_overwrite="/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard"
)
tools.returnn_root = tk.Path("/u/jxu/setups/tedlium2/2023-07-11--ctc-tedlium2/tools/20240509_returnn/returnn",
                             hash_overwrite="/u/berger/repositories/returnn")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"

# ********** Return Config generators **********


def generate_returnn_config(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, lr: dict,
                             batch_size: int, network_args:dict) -> ReturnnConfig:
    network_args["num_outputs"] = num_classes
    model_config = conformer_ctc.get_default_config_v1(**network_args)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"classes": {"dim": num_classes}}

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_classes,
        target="targets",
        extra_python=[conformer_ctc.get_serializer(model_config, variant=variant)],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=0.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        max_seqs=60,
        initial_lr=lr["initial_lr"],
        peak_lr=lr["peak_lr"],
        final_lr=lr["final_lr"],
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
    )


def run_exp() -> Tuple[SummaryReport, Checkpoint, Dict[str, AlignmentData]]:
    assert tools.returnn_root is not None
    assert tools.returnn_python_exe is not None
    assert tools.rasr_binary_path is not None

    data = get_switchboard_data(
        returnn_root=tools.returnn_root,
        feature_type=FeatureType.SAMPLES,
        augmented_lexicon=True,
        test_keys=["hub5e01"],
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=300,
        # gpu_mem_rqmt=11,
    )

    recog_args = exp_args.get_ctc_recog_step_args(num_classes)
    align_args = exp_args.get_ctc_align_step_args(num_classes)
    recog_args["epochs"] = [160, 300, "best"]
    recog_args["feature_type"] = FeatureType.SAMPLES
    recog_args["prior_scales"] = [0.3, 0.5]
    recog_args["lm_scales"] = [0.5, 0.7, 0.9]
    align_args["feature_type"] = FeatureType.SAMPLES

    recog_am_args = copy.deepcopy(exp_args.ctc_recog_am_args)
    recog_am_args.update(
        {
            "tying_type": "global-and-nonword",
            "nonword_phones": ["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"],
        }
    )
    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
        am_args=recog_am_args,
    )
    system.setup_scoring(
        scorer_type=Hub5ScoreJob,
        # stm_kwargs={"non_speech_tokens": ["[NOISE]", "[LAUGHTER]", "[VOCALIZED-NOISE]"]},
        stm_kwargs={"stm_paths": {key: tk.Path("/u/corpora/speech/hub5e_00/xml/hub5e_00.stm") for key in data.dev_keys}},
        score_kwargs={
            "glm": tk.Path("/u/corpora/speech/hub5e_00/xml/glm"),
            # "glm": tk.Path("/u/corpora/speech/hub-5-00/raw/transcriptions/reference/en20000405_hub5.glm"),
        },
    )

    # ********** Returnn Configs **********

    network_args = {}
    lr ={"initial_lr": 7e-6, "peak_lr": 7e-4, "final_lr": 1e-7}

    for ordering in [
        # "laplace:.1000",
        "laplace:.384",
        # "laplace:.100",
        # "laplace:.50",
        # "laplace:.25",
        # "laplace:.10",
        # "random",
    ]:
        mod_train_data_config = copy.deepcopy(data.train_data_config)
        mod_train_data_config["seq_ordering"] = ordering

        train_config = generate_returnn_config(
            ConfigVariant.TRAIN, train_data_config=data.train_data_config, dev_data_config=data.cv_data_config, lr=lr, batch_size=18000*160, network_args=network_args
        )
        recog_config = generate_returnn_config(
            ConfigVariant.RECOG, train_data_config=data.train_data_config, dev_data_config=data.cv_data_config, lr=lr, batch_size=18000*160, network_args=network_args
        )

        returnn_configs = ReturnnConfigs(
            train_config=train_config,
            recog_configs={"recog": recog_config},
        )

        system.add_experiment_configs(f"Conformer_CTC_order-{ordering}", returnn_configs)

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    # system.run_test_recog_step(**recog_args)
    assert system.summary_report
    return system.summary_report
