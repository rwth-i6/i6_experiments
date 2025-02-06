from typing import Optional, Callable
from collections import OrderedDict
import textwrap

from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.common.setups import serialization

base_dataloader_config = OrderedDict(
    train_epoch_split = 1,
    data_files = {
        "train": "/work/asr3/irie/data/librispeech/lm_bpe/transcriptions/librispeech.trans.train.10M.bpe.txt.gz",
        "cv": "/work/asr3/irie/data/librispeech/lm_bpe/dev.clean.other.bpe.txt",
        "test": "/u/michel/setups/language_modelling/librispeech/data/test.clean.other.unk.txt.gz",
    },
    vocab_file = "/work/asr3/irie/data/librispeech/lm_bpe/trans.bpe.vocab.lm.txt", # compatible
    epoch_split = {"train": CodeWrapper("train_epoch_split"), "cv": 1, "test": 1},
    target = "data",
    update_on_device = True,
    num_inputs=10025,
)

lbs_bpe10k_trans_lm_base_config = OrderedDict(
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
    newbob_multi_num_epochs = base_dataloader_config["train_epoch_split"],
    newbob_relative_error_div_by_old = True,
    newbob_learning_rate_decay = 0.8,
    newbob_relative_error_threshold = -0.02,
    newbob_multi_update_interval = 1,
    optimizer = {"class": "sgd"},
    learning_rate_control_error_measure = "dev_loss_ppl",
    **base_dataloader_config,
)

lbs_bpe10k_trans_lm_same_lr_as_kldiv_ilm = OrderedDict(
    backend="torch",
    behavior_version=21,
    task="train",
    batching = "random",
    batch_size = 900,
    max_seq_length = 602,
    max_seqs = 200,
    # chunking = "0",
    accum_grad_multiple_step = 4,
    learning_rate = 1e-5,
    learning_rates = [1e-5]*2,
    learning_rate_control = "newbob_rel",
    learning_rate_control_relative_error_relative_lr = False,
    optimizer = {"class": "adamw", "epsilon": 1e-08, "weight_decay": 1e-06},
    learning_rate_control_error_measure = "dev_loss_ppl",
    **base_dataloader_config,
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


def train_lbs_bpe10k_transcription_lm(
    get_model: Callable,
    train_step: Callable,
    config: dict = lbs_bpe10k_trans_lm_base_config,
    num_epochs: int = 10,
    hashed_config: dict = {},
    non_hashed_config: dict = {},
    post_config_update: dict = {},
):
    """
    Get a Returnn Config to train a transcription LM

    :param hashed_config: Extra config to be hashed
    :param non_hashed_config: Extra unhashed config
    """
    post_config = dict(  # not hashed
        log_batch_size=True,
        cleanup_old_models=True,
        # debug_add_check_numerics_ops = True
        # debug_add_check_numerics_on_output = True
        # stop_on_nonfinite_train_score = False,
        torch_log_memory_usage=True,
        watch_memory=True,
        use_lovely_tensors=True,
        use_train_proc_manager=True,
    )
    post_config.update(post_config_update)
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
                                        "bpe_file": "/u/minh-nghia.phan/setups/rf_ctc/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes",
                                        "vocab_file": "/u/minh-nghia.phan/setups/rf_ctc/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab",
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
                    serialization.CodeFromFunction("get_dataset", func=get_dataset),
                    serialization.PythonModelineNonhashedCode,
                    serialization.NonhashedCode(
                        textwrap.dedent(
                            """
                            dev = get_dataset("cv")
                            train = get_dataset("train")
                            """
                        )
                    ),
                ]
            )
        ],
        post_config=post_config,
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
