import copy

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.wu.experiments.modality_matching.networks import duration_model as duration_model
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.corpus.librispeech.viterbi_transducer_data import get_librispeech_data
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 400

tools = copy.deepcopy(default_tools_v2)
# tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/onnx/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(train_data_config: dict, dev_data_config: dict, extra_config:dict, lr: dict,
                             batch_size: int) -> ReturnnConfig:
    model_config = duration_model.get_default_config_v1(vocab_size=num_outputs)  # default config should be enough for a small model like this

    train_data_config = train_data_config["datasets"]["classes"]
    train_data_config["partition_epoch"] = 10 # partition epoch just for better LR adaptation
    dev_data_config = dev_data_config["datasets"]["classes"]
    extra_config.update({
        "train": train_data_config,
        "dev": dev_data_config,
        "extern_data": {"data": {"dim": 79, "sparse": True}}
    })

    returnn_config = get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="classes",
        extra_python=[duration_model.get_serializer(model_config)],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=0.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        max_seqs=None,
        initial_lr=lr["initial_lr"],
        peak_lr=lr["peak_lr"],
        final_lr=lr["final_lr"],
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
    )
    return returnn_config


def get_returnn_config_collection(
        train_data_config: dict,
        dev_data_config: dict,
        extra_config: dict,
        lr: dict,
        batch_size: int = 36000
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, "lr": lr,
                        "batch_size": batch_size, "extra_config":extra_config}
    train_generator_kwargs = copy.deepcopy(generator_kwargs)
    train_config = returnn_config_generator(**train_generator_kwargs)
    return ReturnnConfigs(
        train_config=train_config,
        prior_config=None,
        recog_configs=None,
    )


def run_lbs_960_duration_model(alignments) -> SummaryReport:
    prefix = "experiments/duration_model"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )

    data = get_librispeech_data(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        alignments=alignments,
        add_unknown_phoneme_and_mapping=False,
        use_augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
    )

    # ********** Step args **********

    # just use ctc training args might be fine
    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs)

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    tools.rasr_binary_path = tk.Path(
        "/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard",
        hash_overwrite="/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    )
    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
    )

    # ********** Returnn Configs **********

    for peak_lr in [7e-4]:
        lr_dict = {
            "initial_lr": peak_lr / 100,
            "peak_lr": peak_lr,
            "final_lr": 1e-8,
        }
        str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")

        extra_config = {}
        system.add_experiment_configs(
            f"duration_model_lbs_960_lr_{str_peak_lr}",
            get_returnn_config_collection(data.train_data_config, data.cv_data_config,
                                            extra_config=extra_config, lr=lr_dict,
                                            batch_size=500000)  # should be large enough?
        )

    system.run_train_step(**train_args)
    
