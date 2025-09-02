import os

from i6_core.lexicon import LexiconFromTextFileJob
from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob, MergeLexiconJob, WriteLexiconJob
from i6_core.lib.lexicon import Lemma, Lexicon
from i6_experiments.common.datasets.switchboard.constants import SUBDIR_PREFIX
from i6_experiments.common.datasets.switchboard.lexicon import get_text_lexicon
from i6_experiments.users.berger.recipe.lexicon.conversion import BlissLexiconToWordLexicon
from sisyphus import tk

from ...tools import subword_nmt_repo
from .bpe import get_bpe_settings


def get_raw_bliss_lexicon() -> tk.Path:
    mapped_raw_lexicon_file = get_text_lexicon(subdir_prefix=SUBDIR_PREFIX)

    bliss_lexicon = LexiconFromTextFileJob(mapped_raw_lexicon_file, compressed=True)
    bliss_lexicon.add_alias(os.path.join(SUBDIR_PREFIX, "create_lexicon_job"))

    return bliss_lexicon.out_bliss_lexicon


def get_bpe_bliss_lexicon(bpe_size: int, add_blank: bool) -> tk.Path:
    bpe_settings = get_bpe_settings(bpe_size)
    lexicon = get_raw_bliss_lexicon()
    bpe_lexicon_file = CreateBPELexiconJob(
        base_lexicon_path=lexicon,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        unk_label="<unk>",
        vocab_blacklist=["</s>"],
        subword_nmt_repo=subword_nmt_repo,
        keep_special_lemmas=False,
    ).out_lexicon

    lexicon_ext = Lexicon()
    lexicon_ext.add_lemma(
        Lemma(orth=["[SENTENCE-BEGIN]"], phon=["<s>"], synt=["<s>"], eval=[[]], special="sentence-begin")
    )
    lexicon_ext.add_lemma(
        Lemma(orth=["[SENTENCE-END]"], phon=["<s>"], synt=["</s>"], eval=[[]], special="sentence-end")
    )
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<unk>"], eval=[[]], special="unknown"))

    if add_blank:
        lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], special="blank", eval=[[]]))
        lexicon_ext.add_phoneme("<blank>", variation="none")

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    return MergeLexiconJob([bpe_lexicon_file, lexicon_ext_file]).out_bliss_lexicon


def get_bliss_phoneme_lexicon() -> tk.Path:
    lexicon_file = get_raw_bliss_lexicon()
    eow_lexicon_file = AddEowPhonemesToLexiconJob(bliss_lexicon=lexicon_file).out_lexicon

    lexicon_ext = Lexicon()
    lexicon_ext.add_phoneme("<laughter>", variation="none")
    lexicon_ext.add_phoneme("<noise>", variation="none")
    lexicon_ext.add_phoneme("<vocalized-noise>", variation="none")
    lexicon_ext.add_phoneme("<unk>", variation="none")
    lexicon_ext.add_phoneme("<blank>", variation="none")

    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], synt=["<s>"], eval=[[]], special="sentence-begin"))
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-END]"], synt=["</s>"], eval=[[]], special="sentence-end"))
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<unk>"], eval=[[]], special="unknown"))
    lexicon_ext.add_lemma(Lemma(orth=["[LAUGHTER]"], phon=["<laughter>"], eval=[[]]))
    lexicon_ext.add_lemma(Lemma(orth=["[NOISE]"], phon=["<noise>"], eval=[[]]))
    lexicon_ext.add_lemma(Lemma(orth=["[VOCALIZED-NOISE]"], phon=["<vocalized-noise>"], eval=[[]]))
    lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], eval=[[]], special="blank"))

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    return MergeLexiconJob([eow_lexicon_file, lexicon_ext_file]).out_bliss_lexicon


def get_bpe_word_lexicon(bpe_size: int) -> tk.Path:
    return BlissLexiconToWordLexicon(get_bpe_bliss_lexicon(bpe_size, add_blank=False)).out_lexicon
