from sisyphus import gs

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.lexicon.conversion import LexiconToWordListJob

from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.tedlium2.lexicon import (
    get_g2p_augmented_bliss_lexicon,
)
from i6_experiments.common.datasets.tedlium2.textual_data import get_text_data_dict
from i6_experiments.common.baselines.tedlium2.default_tools import SRILM_PATH

from i6_experiments.common.setups.lm.srilm_system import SriLmSystem


def run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping: bool = False, alias_prefix="baselines/tedlium2/lm/ngram"):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    dev_data = CorpusToTxtJob(get_bliss_corpus_dict(audio_format="wav", output_prefix="corpora")["dev"]).out_txt
    test_data = CorpusToTxtJob(get_bliss_corpus_dict(audio_format="wav", output_prefix="corpora")["test"]).out_txt

    train_data_dict = get_text_data_dict()
    dev_data_dict = {"dev": dev_data}
    test_data_dict = {
        "dev": dev_data,
        "test": test_data,
    }

    vocab = LexiconToWordListJob(
        get_g2p_augmented_bliss_lexicon(
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping, output_prefix="lexicon"
        )
    ).out_word_list

    ngram_system = SriLmSystem(
        name="tedlium2",
        train_data=train_data_dict,
        dev_data=dev_data_dict,
        eval_data=test_data_dict,
        ngram_order=[3, 4, 5],
        vocab=vocab,
        ngram_args=[
            "-gt1min 1",
            "-gt2min 1",
            "-gt3min 1",
            "-gt4min 1",
            "-gt5min 1",
            "-gt6min 1",
            "-interpolate",
            "-kndiscount",
        ],
        perplexity_args="-debug 2",
        srilm_path=SRILM_PATH,
        ngram_rqmt=None,
        perplexity_rqmt=None,
        mail_address=gs.MAIL_ADDRESS,
    )
    ngram_system.run_training()

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
    return ngram_system
