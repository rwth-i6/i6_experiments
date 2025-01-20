from i6_experiments.common.datasets.switchboard.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.switchboard.bpe import get_subword_nmt_bpe
from i6_experiments.users.berger.recipe.lexicon.bpe_lexicon import CreateBPELexiconJob
from i6_experiments.users.berger.recipe.lexicon.conversion import BlissLexiconToWordLexicon
from sisyphus import tk

from ...tools import subword_nmt_repo


def get_bpe_bliss_lexicon(bpe_size: int) -> tk.Path:
    bpe_settings = get_subword_nmt_bpe(bpe_size=bpe_size)
    lexicon = get_bliss_lexicon()
    lexicon = CreateBPELexiconJob(
        base_lexicon_path=lexicon,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        subword_nmt_repo=subword_nmt_repo,
    ).out_lexicon
    return lexicon


def get_bpe_word_lexicon(bpe_size: int) -> tk.Path:
    return BlissLexiconToWordLexicon(get_bpe_bliss_lexicon(bpe_size)).out_lexicon
