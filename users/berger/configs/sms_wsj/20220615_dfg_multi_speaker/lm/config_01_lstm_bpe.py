import os
from i6_core.returnn.training import Checkpoint

from sisyphus import gs, tk

import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr
from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnTrainingJob

from i6_experiments.users.berger.corpus.sms_wsj.data import (
    get_bpe,
    get_data_inputs,
    PreprocessWSJLexiconJob,
    get_lm_corpus,
)
from i6_experiments.users.berger.network.models.lstm_lm import make_lstm_lm_model
from i6_experiments.users.berger.args.returnn.dataset import get_lm_dataset_config
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.recipe.lexicon.bpe_lexicon import CreateBPELexiconJob
from i6_experiments.users.berger.recipe.text import BpeVocabToVocabFileJob
from i6_experiments.users.berger.recipe.corpus.transform import ReplaceUnknownWordsJob
from i6_experiments.users.berger.args.returnn.learning_rates import (
    Optimizers,
    LearningRateSchedules,
)


# ********** Settings **********

dir_handle = os.path.dirname(__file__).split("config/")[1]
filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

dev_key = "cv_dev93"
test_key = "test_eval92"

bpe_size = 100


def run_exp(**kwargs) -> Checkpoint:

    lm_cleaning = kwargs.get("lm_cleaning", True)

    # ********** Init args **********

    _, dev_data_inputs, test_data_inputs, _ = get_data_inputs(
        train_keys=[],
        dev_keys=[dev_key],
        test_keys=[test_key],
        recog_lex_name="nab-64k",
        lm_cleaning=lm_cleaning,
    )

    # ********** Data inputs **********

    bpe_job = get_bpe(size=bpe_size, lm_cleaning=lm_cleaning)

    bpe_codes = bpe_job.out_bpe_codes
    bpe_vocab = bpe_job.out_bpe_vocab

    num_classes = bpe_job.out_vocab_size  # bpe count

    subword_nmt_repo = CloneGitRepositoryJob("https://github.com/albertz/subword-nmt.git").out_repository

    train_corpus = ApplyBPEToTextJob(
        get_lm_corpus(lm_cleaning=lm_cleaning),
        bpe_codes,
        subword_nmt_repo=subword_nmt_repo,
    ).out_bpe_text

    # Convert corpora to text files containing the BPE symbols
    base_lexicon = dev_data_inputs[dev_key].lexicon["filename"]
    base_lexicon = PreprocessWSJLexiconJob(base_lexicon, lm_cleaning=lm_cleaning).out_lexicon_file
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=base_lexicon,
        bpe_codes=bpe_codes,
        bpe_vocab=bpe_vocab,
        subword_nmt_repo=subword_nmt_repo,
    ).out_lexicon

    for data_input in [dev_data_inputs[dev_key], test_data_inputs[test_key]]:
        corpus_file = data_input.corpus_object.corpus_file
        corpus_file = ReplaceUnknownWordsJob(corpus_file, base_lexicon, unknown_token="[UNKNOWN]").out_corpus_file
        corpus_file = corpus_recipe.ApplyLexiconToCorpusJob(corpus_file, bpe_lexicon).out_corpus
        corpus_file = corpus_recipe.CorpusToTxtJob(corpus_file, gzip=True).out_txt
        data_input.corpus_object.corpus_file = corpus_file

    corpora = {
        "train": train_corpus,
        "dev": dev_data_inputs[dev_key].corpus_object.corpus_file,
        "test": test_data_inputs[test_key].corpus_object.corpus_file,
    }

    vocab_file = BpeVocabToVocabFileJob(bpe_vocab).out_vocab

    name = "_".join(filter(None, ["LSTM_LM", kwargs.get("name_suffix", "")]))
    lstm_net = make_lstm_lm_model(
        num_outputs=num_classes,
        embedding_args={
            "size": 256,
            "initializer": "random_normal_initializer(mean=0.0, stddev=0.1)",
        },
        lstm_args={
            "num_layers": kwargs.get("num_layers", 2),
            "size": kwargs.get("layer_size", 2048),
            "l2": kwargs.get("l2", 0.0),
            "dropout": kwargs.get("dropout", 0.2),
            "initializer": "random_normal_initializer(mean=0.0, stddev=0.1)",
        },
        output_args={
            "dropout": kwargs.get("dropout", 0.2),
            "initializer": "random_normal_initializer(mean=0.0, stddev=0.1)",
        },
    )

    num_subepochs = kwargs.get("num_subepochs", 60)

    config = get_returnn_config(
        lstm_net,
        target="data",
        num_inputs=num_classes,
        num_outputs=num_classes,
        num_epochs=num_subepochs,
        extern_data_config=False,
        use_chunking=False,
        grad_noise=kwargs.get("grad_noise", 0.0),
        grad_clip_global_norm=kwargs.get("grad_clip", 2.0),
        batch_size=kwargs.get("batch_size", 320),
        max_seqs=16,
        optimizer=Optimizers.SGD,
        schedule=LearningRateSchedules.NewbobAbs,
        learning_rate=kwargs.get("learning_rate", 0.1),
        decay=kwargs.get("decay", 0.8),
        multi_num_epochs=15,
        error_measure="dev_score_output:exp",
        extra_config={
            "num_outputs": {
                "data": {
                    "dim": num_classes,
                    "sparse": True,
                    "dtype": "int32" if bpe_size > 200 else "uint8",
                },
                "delayed": {
                    "dim": num_classes,
                    "sparse": True,
                    "dtype": "int32" if bpe_size > 200 else "uint8",
                },
            },
            "calculate_exp_loss": True,
            "learning_rate_control_error_measure": "dev_score_output:exp",
            "train": get_lm_dataset_config(corpora["train"], vocab_file, "laplace:6721", "UNK", 15),
            "dev": get_lm_dataset_config(corpora["dev"], vocab_file, "sorted", "UNK", 1),
            "test": get_lm_dataset_config(corpora["test"], vocab_file, "default", "UNK", 1),
        },
    )

    train_job = ReturnnTrainingJob(
        config,
        log_verbosity=5,
        num_epochs=num_subepochs,
        cpu_rqmt=3,
        mem_rqmt=16,
        time_rqmt=168,
    )
    train_job.add_alias(f"train_nn/{name}_train")
    tk.register_output(
        f"train_nn/{name}_learning_rates",
        train_job.out_learning_rates,
    )

    return train_job.out_checkpoints[num_subepochs]


def py(lm_cleaning: bool = True) -> Checkpoint:
    dir_handle = os.path.dirname(__file__).split("config/")[1]
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"

    train_models = {}
    for batch_size in [320, 640]:
        for lr in [1.0]:
            for num_layers, layer_size in [(2, 2048)]:
                train_models[(batch_size, lr, num_layers, layer_size)] = run_exp(
                    name_suffix=f"{num_layers}x{layer_size}_lr-{lr}_bs-{batch_size}{'_clean' if lm_cleaning else ''}",
                    batch_size=batch_size,
                    learning_rate=lr,
                    num_layers=num_layers,
                    layer_size=layer_size,
                    lm_cleaning=lm_cleaning,
                )

    return train_models[(640, 1.0, 2, 2048)]
