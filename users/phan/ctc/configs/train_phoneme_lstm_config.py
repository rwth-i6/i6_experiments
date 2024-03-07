import copy

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
# from i6_experiments.users.berger.pytorch.models import conformer_ctc
# from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import conformer_ctc_downsample_4 as conformer_ctc
from i6_experiments.users.phan.models import lstm_lm
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.corpus.librispeech.ctc_data import get_librispeech_data_hdf
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.jxu.experiments.ctc.lbs_960.ctc_data import get_librispeech_data_hdf
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.common.setups.serialization import ExplicitHash, CodeFromFunction
from i6_core.returnn.config import CodeWrapper

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 240
num_lstm_layers = 1
embed_dim = 128
dropout = 0.1
hidden_dim = 640
init_learning_rates = [1e-2, 1e-3, 1e-4]
train_split_epoch = 20

tools = copy.deepcopy(default_tools_v2)
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********

def remove_librispeech_audio_features(data_config):
    data_config["datasets"]["targets"]["partition_epoch"] = data_config["datasets"]["features"]["partition_epoch"]
    data_config["datasets"]["targets"]["seq_ordering"] = data_config["datasets"]["features"]["seq_ordering"]
    data_config["datasets"].pop("features")
    data_config["data_map"].pop("targets")
    data_config["data_map"]["data"] = ("targets", "data")
    data_config["seq_order_control_dataset"] = "targets"
    return data_config


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, lr: dict,
                             batch_size: int, kwargs:dict) -> ReturnnConfig:
    model_config = lstm_lm.LSTMLMConfig(
        vocab_dim=num_outputs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_lstm_layers=num_lstm_layers,
        dropout=dropout,
    )

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"classes": {"dim": num_outputs}}

    kwargs.update(lr)
    extra_config.update({"target": "data"})


    return get_returnn_config(
        num_epochs=num_subepochs,
        extern_data_dict={"data": {"shape": (None,)}},
        extra_python=[
            lstm_lm.get_train_serializer(
                model_config,
                "i6_experiments.users.phan.ctc.train_steps.phoneme_lstm"
            ),
        ],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip_global_norm=2.0,
        weight_decay=0.01,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.NewbobRel,
        max_seqs=5,
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
        **kwargs
    )


def get_returnn_config_collection(
        train_data_config: dict,
        dev_data_config: dict,
        lr: dict,
        batch_size: int,
        kwargs: dict
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, "lr": lr,
                        "batch_size": batch_size, "kwargs":kwargs}
    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **generator_kwargs),
    )


def lbs960_train_phoneme_lstm() -> SummaryReport:
    prefix = "experiments/ctc/phoneme_lstm"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )

    data = get_librispeech_data_hdf(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
        blank_index_last=False,
    )
    train_data_config = remove_librispeech_audio_features(data.train_data_config)
    dev_data_config = remove_librispeech_audio_features(data.cv_data_config)

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_subepochs,
        gpu_mem_rqmt=11,
        log_verbosity=4,
    )

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    tools.rasr_binary_path = tk.Path(
        "/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    )
    tools.returnn_root = tk.Path("/u/minh-nghia.phan/tools/simon_returnn")
    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring(score_kwargs={"sctk_binary_path": SCTK_BINARY_PATH})

    # ********** Returnn Configs **********
    # Use newbob lr here
    lr = {
        "learning_rate": init_learning_rate,
        "decay": 0.9,
        "multi_num_epochs": train_split_epoch,
        "relative_error_threshold": -0.005,
        "multi_update_interval": 2,
        "error_measure": "dev_loss_ppl",
    }
    system.add_experiment_configs(
        f"phoneme_lstm_layers{num_lstm_layers}_embed{embed_dim}_hidden{hidden_dim}_dropout{dropout}_sgd_lr{init_learning_rate}",
        get_returnn_config_collection(
            train_data_config,
            dev_data_config,
            lr=lr,
            batch_size=600,
            kwargs={
                "log_verbosity": 4,
            }
        )
    )

    system.run_train_step(**train_args)

    assert system.summary_report
    return system.summary_report
