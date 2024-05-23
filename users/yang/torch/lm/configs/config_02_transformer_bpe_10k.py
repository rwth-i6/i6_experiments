import copy

from sisyphus import gs, tk
from torch import nn

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.yang.torch.returnn.config import get_returnn_config, Backend
from i6_experiments.users.yang.torch.lm.network import transformer_lm
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

def generate_transformer_block_config(input_dim, ff_dim, output_dim, num_heads, dropout=0.0, batch_first=False):
    linear_config = transformer_lm.TransformerLinearConfig(
        input_dim=input_dim,
        ff_dim=ff_dim,
        output_dim=output_dim,
        dropout=0.0,
        batch_first=batch_first
    )
    mhsa_config = transformer_lm.TransformerMHSAConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=batch_first
    )
    block_config = transformer_lm.TransformerBlockConfig(
        linear_config=linear_config,
        mhsa_config=mhsa_config
    )

    return block_config




def returnn_config_generator(
    num_epochs: int,
    lr: dict,
    batch_size: int,
    **kwargs
) -> ReturnnConfig:

    ############# training data ####################
    train_epoch_split = kwargs.get('train_epoch_split',4)
    train_data, dev_data = get_librispeech_lm_data_bpe_10k(train_epoch_split=train_epoch_split)
    train_data.update({"seq_ordering":"random"})
    extra_config = {
        "train": train_data,
        "dev": dev_data
    }
    #################### model config ####################

    num_outputs = kwargs.get('num_outputs', 10025)
    embed_dim = kwargs.get('embed_dim', 128)
    hidden_dim = kwargs.get('hidden_dim', 1024)
    num_layers = kwargs.get('num_layers',24)
    ff_dim = kwargs.get('ff_dim', 4096)
    dropout = kwargs.get('dropout', 0.0)
    num_heads = kwargs.get('num_heads',8)
    batch_first = kwargs.get('batch_first', False)
    trafo_block_config = generate_transformer_block_config(hidden_dim,ff_dim,hidden_dim, num_heads, dropout, batch_first=batch_first)
    model_config = transformer_lm.TransformerLMConfig(
        embed_dim=embed_dim,
        hid_dim=hidden_dim,
        vocab_dim=num_outputs,
        num_layers=num_layers,
        block_config=trafo_block_config,
        batch_first=batch_first,
        dropout=dropout
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
            transformer_lm.get_train_serializer(
                model_config,
                "i6_experiments.users.yang.torch.lm.train_steps.bpe_loss_v2"
            ),
        ],
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip_global_norm=1.0,
        optimizer=Optimizers.SGD,
        schedule=LearningRateSchedules.NewbobAbs,
        max_seqs=32,
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

def lbs960_train_bpe_trafo() -> SummaryReport:
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
    num_epochs = 30
    init_learning_rates = [1.0]
    train_epoch_split = 4
    batch_size = 1350
    training_args = {
        'num_epochs': num_epochs,
        "log_verbosity": 4,
    }
    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_epochs,
        gpu_mem_rqmt=11,
        mem_rqmt=64,
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
            f"kazuki_trafo_bpe_10k_SGD_lr{init_learning_rate}_b128_log_ppl",
            get_returnn_config_collection(
                lr=lr,
                batch_size=batch_size,
                **training_args
            )
        )

    system.run_train_step(**train_args)

    assert system.summary_report
    return system.summary_report



