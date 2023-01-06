import os

from sisyphus import gs, tk

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.lexicon.conversion import LexiconToWordListJob

from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.tedlium2.lexicon import (
    get_g2p_augmented_bliss_lexicon,
)
from i6_experiments.common.datasets.tedlium2.textual_data import get_text_data_dict
from i6_experiments.common.baselines.tedlium2.default_tools import SRILM_PATH

from i6_private.users.luescher.setups.lm.srilm_system import SriLmSystem


SRILM_PATH = tk.Path("/work/tools/users/luescher/srilm-1.7.3/bin/i686-m64")


def run_tedlium2_ngram_lm(alias_prefix="baselines/tedlium2/lm/ngram"):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    train_data = get_text_data_dict()
    dev_data = {
        "dev": get_bliss_corpus_dict(audio_format="wav", output_prefix="corpora")["dev"]
    }
    test_data = {
        "dev": get_bliss_corpus_dict(audio_format="wav", output_prefix="corpora")[
            "dev"
        ],
        "test": get_bliss_corpus_dict(audio_format="wav", output_prefix="corpora")[
            "test"
        ],
    }

    vocab = LexiconToWordListJob(get_g2p_augmented_bliss_lexicon())

    ngram_system = SriLmSystem(
        name="tedlium2",
        train_data=train_data,
        dev_data=dev_data,
        eval_data=test_data,
        ngram_order=[3, 4, 5],
        vocab=vocab,
        ngram_args=None,
        perplexity_args=None,
        srilm_path=SRILM_PATH,
        ngram_rqmt=None,
        perplexity_rqmt=None,
        mail_address=gs.MAIL_ADDRESS,
    )
    ngram_system.run()
