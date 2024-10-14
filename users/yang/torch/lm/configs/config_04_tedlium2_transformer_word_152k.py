import copy

from sisyphus import gs, tk
from torch import nn
import numpy as np

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.yang.torch.returnn.config import get_returnn_config, Backend
import i6_experiments.users.yang.torch.lm.network.transformer_lm_v3 as transformer_lm
from i6_experiments.users.yang.torch.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.yang.torch.lm.dataloader.tedlium2_lm_dataloader_word_152k import get_tedlium2_lm_data_word_152k

from i6_experiments.common.setups.serialization import (
    ExplicitHash, CodeFromFunction, Import, Collection,
    NonhashedCode, PythonCacheManagerFunctionNonhashedCode
)



from i6_core.returnn.config import CodeWrapper



tools = copy.deepcopy(default_tools_v2)
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"

def generate_transformer_block_config(input_dim, ff_dim, output_dim, num_heads, dropout=0.0, num_additional_ff=0, batch_first=False, ):
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
        mhsa_config=mhsa_config,
        num_additional_ff=num_additional_ff,
    )

    return block_config




def returnn_config_generator(
    num_epochs: int,
    lr: dict,
    batch_size: int,
    pretrain: bool=True,
    **kwargs
) -> ReturnnConfig:

    ############# training data ####################
    extra_config = kwargs.pop("extra_config", {})
    train_epoch_split = kwargs.get('train_epoch_split',10)
    train_data, dev_data = get_tedlium2_lm_data_word_152k(train_epoch_split=train_epoch_split, pretrain=pretrain)
    extra_config.update({
        "train": train_data,
        "dev": dev_data
    })
    #################### model config ####################

    num_outputs = kwargs.get('num_outputs', 152267)
    embed_dim = kwargs.get('embed_dim', 128)
    hidden_dim = kwargs.get('hidden_dim', 768)
    num_layers = kwargs.get('num_layers',8)
    ff_dim = kwargs.get('ff_dim', 4096)
    dropout = kwargs.get('dropout', 0.1)
    num_heads = kwargs.get('num_heads',12)
    batch_first = kwargs.get('batch_first', False)
    num_additional_ff = kwargs.get('num_additional_ff',3)
    max_seq_length = kwargs.get('max_seq_length', 602)
    optimizer = kwargs.get('returnn_optimizer', 'SGD')
    if optimizer == 'SGD':
        returnn_optimizer = Optimizers.SGD
    elif optimizer == 'RAdam':
        returnn_optimizer = Optimizers.RAdam
    else:
        raise NotImplementedError

    lr_schedule = kwargs.get('lr_schedule', 'newbob_rel')
    if lr_schedule == 'newbob_rel':
        returnn_schedule = LearningRateSchedules.NewbobRel
    elif lr_schedule == 'custom':
        returnn_schedule = LearningRateSchedules.Custom
    else:
        raise NotImplementedError

    trafo_block_config = generate_transformer_block_config(hidden_dim,ff_dim,hidden_dim, num_heads, dropout=dropout,
                                                           num_additional_ff=num_additional_ff,
                                                           batch_first=batch_first)
    model_config = transformer_lm.TransformerLMConfig(
        embed_dim=embed_dim,
        hid_dim=hidden_dim,
        vocab_dim=num_outputs,
        num_layers=num_layers,
        block_config=trafo_block_config,
        batch_first=batch_first,
        dropout=dropout,
        use_pos_encoding=False,
    )
    kwargs.update(lr)
    extra_config.update({
        "target": "data",
        "num_inputs": num_outputs,
        "max_seq_length": max_seq_length
    })
    # extra_config_update = kwargs.get("extra_config", None)
    # if extra_config_update is not None:
    #     extra_config.update(extra_config_update)
    avg_loss = kwargs.get('avg_loss', False)
    if avg_loss:
        loss_function_str = "i6_experiments.users.yang.torch.lm.train_steps.bpe_loss_v2_avg"
    else:
        loss_function_str = "i6_experiments.users.yang.torch.lm.train_steps.bpe_loss_v2"
    return get_returnn_config(
        num_epochs=num_epochs,
        extern_data_dict={"data": {"shape": (None,)}, "delayed": {"shape": (None,)}},
        extra_python=[
            transformer_lm.get_train_serializer(
                model_config,
                loss_function_str
            ),
        ],
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip_global_norm=1.0,
        optimizer=returnn_optimizer,
        schedule=returnn_schedule,
        max_seqs=64,
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

def tedlium2_pretrain_trafo() -> SummaryReport:
    prefix = "experiments/tedlium_word_lm"
    # two stages: pretrain and finetune


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
    num_epochs = 40
    init_learning_rates = [1.0]
    train_epoch_split = 10
    batch_size = 900
    training_args = {
        'num_epochs': num_epochs,
        "log_verbosity": 4,
        "train_epoch_split": train_epoch_split,
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
            "decay": 0.95,
            "multi_num_epochs": train_epoch_split,
            "relative_error_threshold": -0.007,
            "multi_update_interval": 1,
            "error_measure": "fake_ppl",
        }


        system.add_experiment_configs(
            f"tedlium2_kazuki_trafo_pre_train_word_152k_nopos_4ff_SGD_lr{init_learning_rate}_fake_ppl",
            get_returnn_config_collection(
                lr=lr,
                batch_size=batch_size,
                **training_args
            )
        )
    #system.run_train_step(**train_args)




    system.run_train_step(**train_args)

    system = ReturnnSeq2SeqSystem(tools)
    #### epoch split 100, RAdam optimizer
    num_epochs = 200
    train_epoch_split = 100
    # lr 1e-3 too large, performance not good
    # 1e-4 works fine, half full epoch dev ppl 108, same as SGD, but slower

    learning_rates = list(np.linspace(1e-5, 1e-4, 20)) + [1e-4] * 80 + list(np.linspace(1e-4, 1e-5, 100))
    training_args = {
        'num_epochs': num_epochs,
        "log_verbosity": 5,
        "returnn_optimizer": "RAdam",
        "learning_rates": learning_rates,
        "lr_schedule": "custom",
        "train_epoch_split": train_epoch_split,
        "avg_loss": True,
    }
    assert len(learning_rates)==num_epochs
    system.add_experiment_configs(
        f"tedlium2_kazuki_trafo_pre_train_word_152k_nopos_4ff_RAdam_lr1e-5_1e-4_1e-5_split{train_epoch_split}_wp20_fake_ppl",
        get_returnn_config_collection(
            lr={},
            batch_size=batch_size,
            **training_args
        )
    )
    learning_rates = [1e-5]*10 + list(np.linspace(1e-5,5e-4,90)) + list(np.linspace(5e-4, 1e-5, 100))
    training_args = {
        'num_epochs': num_epochs,
        "log_verbosity": 4,
        "returnn_optimizer": "RAdam",
        "learning_rates": learning_rates,
        "lr_schedule": "custom",
        "train_epoch_split": train_epoch_split,
        "avg_loss": True,
    }
    assert len(learning_rates)==num_epochs
    system.add_experiment_configs(
        f"tedlium2_kazuki_trafo_pre_train_word_152k_nopos_4ff_RAdam_lr1e-5_ep10_5e-4_ep90_1e-5_split{train_epoch_split}_fake_ppl",
        get_returnn_config_collection(
            lr={},
            batch_size=batch_size,
            **training_args
        )
    )


    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_epochs,
        gpu_mem_rqmt=11,
        mem_rqmt=64,
        log_verbosity=5,
    )




    system.run_train_step(**train_args)




    assert system.summary_report

    ##############################only the pretrain phase
    return system.summary_report


#### fine-tune

def tedlium2_finetune_trafo() -> SummaryReport:
    prefix = "experiments/tedlium_word_lm"
    # two stages: pretrain and finetune

    pre_model_path = "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.NT5CuaY8plk5/output/models/epoch.040.pt"


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
    num_epochs = 20
    init_learning_rates = [0.5]
    train_epoch_split = 10
    batch_size = 900
    training_args = {
        'num_epochs': num_epochs,
        "log_verbosity": 4,
        "train_epoch_split": train_epoch_split,
        "extra_config": {"preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename":pre_model_path,}}},
        "pretrain": False,
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
            "decay": 0.95,
            "multi_num_epochs": train_epoch_split,
            "relative_error_threshold": -0.007,
            "multi_update_interval": 1,
            "error_measure": "fake_ppl",
        }


        system.add_experiment_configs(
            f"tedlium2_kazuki_trafo_fine_tune_word_152k_nopos_4ff_SGD_lr{init_learning_rate}_fake_ppl",
            get_returnn_config_collection(
                lr=lr,
                batch_size=batch_size,
                **training_args
            )
        )

    system.run_train_step(**train_args)





    assert system.summary_report

    ##############################only the pretrain phase
    return system.summary_report

