import copy
import inspect

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
from i6_experiments.common.setups.serialization import (
    ExplicitHash, CodeFromFunction, Import, Collection,
    NonhashedCode, PythonCacheManagerFunctionNonhashedCode
)
from i6_core.returnn.config import CodeWrapper
from i6_experiments.users.phan.ctc.train_steps.phoneme_lstm_old_dataloader import \
    train_step as old_dataloader_train_step
from i6_experiments.users.phan.models.lstm_lm import LSTMLM

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 120
num_lstm_layers = 1
embed_dim = 128
dropout = 0.1
hidden_dim = 640
init_learning_rates = [1e-2, 1e-3, 1e-4]
train_split_epoch = 10

tools = copy.deepcopy(default_tools_v2)
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def get_dataset(data):
    epoch_split = {"train": 10, "cv": 1, "test": 1}
    seq_order = {
        "train": "random",
        "cv": "sorted",
        "test": "default"
    }
    assert data in ["train", "cv", "test"]
    return {
        "class": "LmDataset",
        "corpus_file": lambda: cf(data_files[data]),
        "orth_symbols_map_file": lambda: cf(vocab_file),
        "orth_replace_map_file": None,
        "word_based": True,
        "seq_end_symbol": '<sb>',
        "auto_replace_unknown_symbol": False,
        "unknown_symbol": None,
        "add_delayed_seq_data": True,
        "delayed_seq_data_start_symbol": '<sb>',
        "seq_ordering": seq_order[data],
        "partition_epoch": epoch_split[data]
    }

def returnn_config_generator(
    variant: ConfigVariant,
    lr: dict,
    batch_size: int,
    kwargs:dict
) -> ReturnnConfig:
    model_config = lstm_lm.LSTMLMConfig(
        vocab_dim=num_outputs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_lstm_layers=num_lstm_layers,
        dropout=dropout,
    )

    extra_config = {
        "train": CodeWrapper("get_dataset('train')"),
        "dev": CodeWrapper("get_dataset('cv')")
    }

    kwargs.update(lr)
    extra_config.update({
        "target": "data",
        "data_files": {
            "train": "/work/asr3/zhou/hiwis/wu/data/librispeech/lm/train-merged-960.phon-corpus.txt",
            "cv": "/work/asr3/zhou/hiwis/wu/data/librispeech/lm/dev-cv.phon-corpus.txt"
        },
        "vocab_file": "/u/hwu/asr-exps/util/lmPrep/vocab",
        "orth_replace_map_file": CodeWrapper("None"),
        "num_inputs": 79,
        "train_epoch_split": 10,
    })

    return get_returnn_config(
        num_epochs=num_subepochs,
        extern_data_dict={"data": {"shape": (None,)}, "delayed": {"shape": (None,)}},
        extra_python=[
            lstm_lm.get_train_serializer(
                model_config,
                "i6_experiments.users.phan.ctc.train_steps.phoneme_lstm_old_dataloader"
            ),
        ],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip_global_norm=2.0,
        weight_decay=0.01,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.NewbobRel,
        max_seqs=32,
        max_seq_length = 1000,
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
        python_prolog=[
            Collection([
                NonhashedCode("import os\n"),
                Import("subprocess.check_output"),
                PythonCacheManagerFunctionNonhashedCode,
            ]),
            CodeFromFunction("get_dataset", func=get_dataset),
        ],
        **kwargs
    )


def get_returnn_config_collection(
        lr: dict,
        batch_size: int,
        kwargs: dict
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {
        "lr": lr,
        "batch_size": batch_size,
        "kwargs":kwargs
    }
    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **generator_kwargs),
    )


def lbs960_train_phoneme_lstm() -> SummaryReport:
    prefix = "experiments/ctc/phoneme_lstm"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )

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

    # ********** Returnn Configs **********
    # Use newbob lr here
    for init_learning_rate in init_learning_rates:
        lr = {
            "learning_rate": init_learning_rate,
            "decay": 0.9,
            "multi_num_epochs": 10,
            "relative_error_threshold": -0.005,
            "multi_update_interval": 1,
            "error_measure": "dev_loss_ppl",
        }
        system.add_experiment_configs(
            f"phoneme_lstm_layers{num_lstm_layers}_embed{embed_dim}_hidden{hidden_dim}"
            f"_adamw_lr{init_learning_rate}_old_dataloader",
            get_returnn_config_collection(
                lr=lr,
                batch_size=1000,
                kwargs={
                    "log_verbosity": 4,
                }
            )
        )

    system.run_train_step(**train_args)

    assert system.summary_report
    return system.summary_report
