import copy

from sisyphus import gs, tk
from torch import nn

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.yang.torch.returnn.config import get_returnn_config, Backend
from i6_experiments.users.yang.torch.lm.network import lstm_lm
from i6_experiments.users.yang.torch.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.yang.torch.lm.dataloader.lm_dataloader_bpe_10k import get_librispeech_lm_data_bpe_10k

from i6_experiments.common.setups.serialization import (
    ExplicitHash, CodeFromFunction, Import, Collection,
    NonhashedCode, PythonCacheManagerFunctionNonhashedCode
)



from i6_core.returnn.config import CodeWrapper



tools = copy.deepcopy(default_tools_v2)
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


def returnn_config_generator(
    num_epochs: int,
    lr: dict,
    batch_size: int,
    **kwargs
) -> ReturnnConfig:

    ############# training data ####################
    train_epoch_split = kwargs.get('train_epoch_split',4)
    train_data, dev_data = get_librispeech_lm_data_bpe_10k(train_epoch_split=train_epoch_split)
    extra_config = {
        "train": train_data,
        "dev": dev_data
    }
    #################### model config ####################

    num_outputs = kwargs.get('num_outputs', 10025)
    embed_dim = kwargs.get('embed_dim', 512)
    hidden_dim = kwargs.get('hidden_dim', 2048)
    num_lstm_layers = kwargs.get('num_lstm_layers',2)
    bottle_neck = kwargs.get('bottle_neck', False)
    bottle_neck_dim = kwargs.get('bottle_neck_dim', 512)
    dropout = kwargs.get('dropout', 0.2)
    default_init_args = {
        'init_args_w':{'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}},
        'init_args_b': {'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}}
    }
    init_args = kwargs.get('init_args', default_init_args)
    model_config = lstm_lm.LSTMLMConfig(
        vocab_dim=num_outputs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_lstm_layers=num_lstm_layers,
        init_args=init_args,
        dropout=dropout,
    )

    kwargs.update(lr)
    extra_config.update({
        "target": "data",
        "orth_replace_map_file": CodeWrapper("None"),
        "num_inputs": num_outputs,
    })
    return get_returnn_config(
        num_epochs=num_epochs,
        extern_data_dict={"data": {"shape": (None,)}, "delayed": {"shape": (None,)}},
        extra_python=[
            lstm_lm.get_train_serializer(
                model_config,
                "i6_experiments.users.yang.torch.lm.train_steps.bpe_loss_sum"
            ),
        ],
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip_global_norm=2.0,
        optimizer=Optimizers.SGD,
        schedule=LearningRateSchedules.NewbobAbs,
        max_seqs=128,
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
        python_prolog=[
            Collection([
                NonhashedCode("import os\n"),
                Import("subprocess.check_output"),
                PythonCacheManagerFunctionNonhashedCode,
            ]),
        ],
        **kwargs
    )

def get_returnn_config_collection(
        lr: dict,
        batch_size: int,
        **kwargs
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = copy.deepcopy(kwargs)
    generator_kwargs.update({
        "lr": lr,
        "batch_size": batch_size,
    })

    return ReturnnConfigs(
        train_config=returnn_config_generator(**generator_kwargs),
    )

def lbs960_train_bpe_lstm() -> SummaryReport:
    prefix = "experiments/bpe_lm"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )
    #

    # ********** Step args **********


    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    tools.rasr_binary_path = tk.Path(
        "/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    )
    tools.returnn_root = tk.Path("/u/zyang/returnn/simon_returnn")
    system = ReturnnSeq2SeqSystem(tools)

    # ********** experiments **********
    num_epochs= 30
    init_learning_rates = [1.0]
    train_epoch_split = 4
    batch_size = 1280
    training_args = {
        'num_epochs': num_epochs,
        "log_verbosity": 4,
    }
    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_epochs,
        gpu_mem_rqmt=11,
        mem_rqmt=32,
        log_verbosity=5,
    )

    for init_learning_rate in init_learning_rates:
        lr = {
            "learning_rate": init_learning_rate,
            "decay": 0.8,
            "multi_num_epochs": train_epoch_split,
            "relative_error_threshold": 0,
            "multi_update_interval": 1,
            "error_measure": "log_ppl",
        }


        system.add_experiment_configs(
            f"kazuki_lstm_bpe_SGD_lr{init_learning_rate}_b128_log_ppl",
            get_returnn_config_collection(
                lr=lr,
                batch_size=batch_size,
                **training_args
            )
        )

    system.run_train_step(**train_args)

    assert system.summary_report
    return system.summary_report



