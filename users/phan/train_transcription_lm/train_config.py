from typing import Optional, Callable
from collections import OrderedDict
import textwrap

from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.common.setups import serialization


# _cf_cache = {}

# def cf(filename):
#     """Cache manager"""
#     if filename in _cf_cache:
#         return _cf_cache[filename]
#     if check_output(["hostname"]).strip() in ["cluster-cn-211", "sulfid"]:
#         print("use local file: %s" % filename)
#         return filename  # for debugging
#     cached_fn = check_output(["cf", filename]).strip().decode("utf8")
#     assert os.path.exists(cached_fn)
#     _cf_cache[filename] = cached_fn
#     return cached_fn


base_sgd_config = OrderedDict( # base config without dataloader
    backend="torch",
    behavior_version=21,
    task="train",
    batching = "random",
    batch_size = 900,
    max_seq_length = 602,
    max_seqs = 64,
    # chunking = "0",
    gradient_clip_global_norm = 2.,
    gradient_noise = 0.,
    learning_rate = 1.,
    learning_rate_control = "newbob_rel",
    learning_rate_control_relative_error_relative_lr = False,
    newbob_multi_num_epochs = CodeWrapper("train_epoch_split"),
    newbob_relative_error_div_by_old = True,
    newbob_learning_rate_decay = 0.8,
    newbob_relative_error_threshold = -0.02,
    newbob_multi_update_interval = 1,
    optimizer = {"class": "sgd"},
    learning_rate_control_error_measure = "dev_loss_ppl",
)

base_adamw_newbob_lr_config = OrderedDict( # base config without dataloader
    backend="torch",
    behavior_version=21,
    task="train",
    batching = "random",
    batch_size = 900,
    max_seq_length = 602,
    max_seqs = 64,
    # chunking = "0",
    learning_rate = 0.001,
    learning_rate_control = "newbob_rel",
    learning_rate_control_relative_error_relative_lr = False,
    newbob_multi_num_epochs = CodeWrapper("train_epoch_split"),
    newbob_relative_error_div_by_old = True,
    newbob_learning_rate_decay = 0.8,
    newbob_relative_error_threshold = -0.02,
    newbob_multi_update_interval = 1,
    optimizer={
        "class": "adamw",
        "epsilon": 1e-8,
        "weight_decay": 1e-6,
    },
    learning_rate_control_error_measure = "dev_loss_ppl",
)

base_adamw_const_lr_config = OrderedDict( # base config without dataloader
    backend="torch",
    behavior_version=21,
    task="train",
    batching = "random",
    batch_size = 900,
    max_seq_length = 602,
    max_seqs = 64,
    # chunking = "0",
    learning_rate = 0.001,
    optimizer={
        "class": "adamw",
        "epsilon": 1e-8,
        "weight_decay": 1e-6,
    },
    learning_rate_control_error_measure = "dev_loss_ppl",
)

def get_dataset(data):
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

def get_dataset_eos_bos(data):
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
        "seq_end_symbol": '</s>',
        "auto_replace_unknown_symbol": False,
        "unknown_symbol": None,
        "add_delayed_seq_data": True,
        "delayed_seq_data_start_symbol": '<s>',
        "seq_ordering": seq_order[data],
        "partition_epoch": epoch_split[data]
    }

def get_dataset_multifiles(data):
    assert data in ["train", "cv", "test"]
    return {
        "class": "LmDataset",
        "corpus_file": lambda: list(map(cf, data_files[data])),
        "orth_symbols_map_file": lambda: cf(vocab_file),
        "orth_replace_map_file": orth_replace_map_file,
        "word_based": True,
        "seq_end_symbol": "<sb>",
        "auto_replace_unknown_symbol": True,
        "unknown_symbol": "<unk>",
        "add_delayed_seq_data": True,
        "delayed_seq_data_start_symbol": "<sb>",
        "seq_ordering": seq_order[data],
        "partition_epoch": epoch_split[data]
    }


def train_transcription_lm_bpe10k_lower_case_job(
    get_model: Callable,
    train_step: Callable,
    config: dict,
    num_epochs: int = 10,
    hashed_config: dict = {},
    non_hashed_config: dict = {},
):
    """
    Get a Returnn Config to train a transcription LM

    :param hashed_config: Extra config to be hashed
    :param non_hashed_config: Extra unhashed config
    """

    returnn_train_config = ReturnnConfig(
        config=config,
        python_prolog=[
            serialization.Collection(
                [   
                    serialization.NonhashedCode(
                        textwrap.dedent(
                            """
                            import os
                            """
                        )
                    ),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                    serialization.NonhashedCode(
                        textwrap.dedent(
                            """
                            sys.path.insert(0, "/u/minh-nghia.phan/setups/rf_ctc/recipe")
                            sys.path.insert(1, "/u/zeyer/setups/combined/2021-05-31/tools/sisyphus")
                            """
                        )
                    ),
                    
                    serialization.NonhashedCode(
                        textwrap.dedent(
                            """
                            from returnn.tensor import Dim, batch_dim
                            target_spatial_dim = Dim(description="target-spatial", dimension=None, kind=Dim.Types.Spatial)
                            vocab_dim = Dim(description="vocab", dimension=10025, kind=Dim.Types.Spatial)

                            extern_data = {
                                "data": {
                                    "dim_tags": [batch_dim, target_spatial_dim],
                                    "sparse_dim": vocab_dim,
                                    "vocab": {
                                        "bpe_file": "/u/zyang/setups/vocab/librispeech/bpe10k/lower_case.bpe.codes",
                                        "vocab_file": "/u/zyang/setups/vocab/librispeech/bpe10k/lower_case.bpe.vocab",
                                        "unknown_label": None,
                                        "bos_label": 0,
                                        "eos_label": 0,
                                    },
                                },
                            }
                            extern_data["delayed"] = extern_data["data"]
                            """
                        )
                    ),
                    serialization.Import(get_model, import_as="get_model"),
                    serialization.Import(train_step, import_as="train_step"),
                    
                ]
            )
        ],
        python_epilog=[
            serialization.Collection(
                [
                    serialization.CodeFromFunction("get_dataset", func=get_dataset_eos_bos),
                    serialization.PythonModelineNonhashedCode,
                    serialization.NonhashedCode(
                        textwrap.dedent(
                            """
                            # dev = get_dataset("cv")
                            eval_datasets = {
                                "dev": get_dataset("cv"),
                                "test": get_dataset("test"),    
                            }
                            train = get_dataset("train")
                            """
                        )
                    ),
                ]
            )
        ],
        post_config=dict(  # not hashed
            log_batch_size=True,
            cleanup_old_models=True,
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # stop_on_nonfinite_train_score = False,
            torch_log_memory_usage=True,
            watch_memory=True,
            use_lovely_tensors=True,
            use_train_proc_manager=True,
        ),
        sort_config=False,
    )

    returnn_train_config.config.update(hashed_config)
    returnn_train_config.post_config.update(non_hashed_config)

    returnn_train_job = ReturnnTrainingJob(
        returnn_config=returnn_train_config,
        log_verbosity=5,
        num_epochs=num_epochs,
    )
    returnn_train_job.rqmt["gpu_mem"] = 11

    return returnn_train_job
