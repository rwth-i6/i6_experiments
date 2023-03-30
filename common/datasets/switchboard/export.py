import os
from sisyphus import tk

from i6_experiments.common.datasets.switchboard.constants import SUBDIR_PREFIX
from i6_experiments.common.datasets.switchboard.corpus_eval import (
    get_hub5e00,
    get_hub5e01,
    get_rt03s,
)
from i6_experiments.common.datasets.switchboard.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.switchboard.corpus_train import (
    get_train_bliss_corpus_ldc,
    get_train_bliss_corpus_i6_legacy,
    get_spoken_form_train_bliss_corpus_ldc,
)


def _export_eval(subdir_prefix: str = SUBDIR_PREFIX):
    """
    Registers bliss, stm and glm for the eval parts
    :param subdir_prefix:
    """
    hub5e00 = get_hub5e00()
    tk.register_output(os.path.join(subdir_prefix, "hub5e00", "hub5e00.xml.gz"), hub5e00.bliss_corpus)
    tk.register_output(os.path.join(subdir_prefix, "hub5e00", "hub5e00.stm"), hub5e00.stm)
    tk.register_output(os.path.join(subdir_prefix, "hub5e00", "hub5e00.glm"), hub5e00.glm)
    hub5e01 = get_hub5e01()
    tk.register_output(os.path.join(subdir_prefix, "hub5e01", "hub5e01.xml.gz"), hub5e01.bliss_corpus)
    tk.register_output(os.path.join(subdir_prefix, "hub5e01", "hub5e01.stm"), hub5e01.stm)
    tk.register_output(os.path.join(subdir_prefix, "hub5e01", "hub5e01.glm"), hub5e01.glm)
    rt03s = get_rt03s()
    tk.register_output(os.path.join(subdir_prefix, "rt03s", "rt03s.xml.gz"), rt03s.bliss_corpus)
    tk.register_output(os.path.join(subdir_prefix, "rt03s", "rt03s.stm"), rt03s.stm)
    tk.register_output(os.path.join(subdir_prefix, "rt03s", "rt03s.glm"), rt03s.glm)


def _export_lexicon(subdir_prefix: str = SUBDIR_PREFIX):
    """
    Registers output for switchboard lexicon

    :param str subdir_prefix: alias prefix name
    """
    lex = get_bliss_lexicon(subdir_prefix)
    tk.register_output(os.path.join(subdir_prefix, "lexicon.xml.gz"), lex)


def _export_train_bliss_corpus(subdir_prefix: str = SUBDIR_PREFIX):
    """
    Registers output for switchboard training corpus based on i6 internal audio directory

    :param str subdir_prefix: alias prefix name
    """
    train_bliss_corpus_ldc = get_train_bliss_corpus_ldc()
    train_bliss_corpus_legacy = get_train_bliss_corpus_i6_legacy()
    train_bliss_corpus_ldc_subword = get_spoken_form_train_bliss_corpus_ldc()
    tk.register_output(os.path.join(subdir_prefix, "train.corpus.gz"), train_bliss_corpus_ldc)
    tk.register_output(
        os.path.join(subdir_prefix, "train.subword_processed.corpus.gz"),
        train_bliss_corpus_ldc_subword,
    )
    tk.register_output(
        os.path.join(subdir_prefix, "Switchboard-i6-legacy", "train.corpus.gz"),
        train_bliss_corpus_legacy,
    )


def export_all(subdir_prefix: str = SUBDIR_PREFIX):
    """
    :param subdir_prefix: alias prefix name
    """
    _export_eval(subdir_prefix)
    _export_lexicon(subdir_prefix)
    _export_train_bliss_corpus(subdir_prefix)
