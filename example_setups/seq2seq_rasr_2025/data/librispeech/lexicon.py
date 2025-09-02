from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob, MergeLexiconJob, WriteLexiconJob
from i6_core.lib.lexicon import Lemma, Lexicon
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe_v2
from i6_experiments.common.datasets.tedlium2.lexicon import get_bliss_lexicon as get_tedlium_bliss_lexicon
from i6_experiments.users.berger.recipe.lexicon.conversion import BlissLexiconToWordLexicon
from sisyphus import tk

from ...tools import subword_nmt_repo
from ..base import RemoveSpecialLemmasFromLexiconJob, UppercaseLexiconJob


def get_bpe_bliss_lexicon(bpe_size: int, add_blank: bool) -> tk.Path:
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key="train-other-960", bpe_size=bpe_size)
    lexicon = get_bliss_lexicon()
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
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], phon=["<s>"], synt=["<s>"], special="sentence-begin"))
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-END]"], phon=["<s>"], synt=["</s>"], special="sentence-end"))
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<UNK>"], special="unknown"))

    if add_blank:
        lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], special="blank"))
        lexicon_ext.add_phoneme("<blank>", variation="none")

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    return MergeLexiconJob([bpe_lexicon_file, lexicon_ext_file]).out_bliss_lexicon


def get_bliss_phoneme_lexicon() -> tk.Path:
    lexicon_file = get_bliss_lexicon(add_silence=False, add_unknown_phoneme_and_mapping=False)
    cleaned_lexicon_file = RemoveSpecialLemmasFromLexiconJob(lexicon_file).out_lexicon
    eow_lexicon_file = AddEowPhonemesToLexiconJob(bliss_lexicon=cleaned_lexicon_file).out_lexicon

    lexicon_ext = Lexicon()
    lexicon_ext.add_phoneme("<unk>", variation="none")
    lexicon_ext.add_phoneme("<blank>", variation="none")

    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], synt=["<s>"], special="sentence-begin"))
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-END]"], synt=["</s>"], special="sentence-end"))
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<UNK>"], special="unknown"))
    lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], synt=[], eval=[], special="silence"))
    lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], special="blank"))

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    return MergeLexiconJob([eow_lexicon_file, lexicon_ext_file]).out_bliss_lexicon


def get_bpe_word_lexicon(bpe_size: int) -> tk.Path:
    return BlissLexiconToWordLexicon(get_bpe_bliss_lexicon(bpe_size, add_blank=False)).out_lexicon


def get_tedlium2_bpe_bliss_lexicon(bpe_size: int, add_blank: bool) -> tk.Path:
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key="train-other-960", bpe_size=bpe_size)
    tedlium_lexicon_file = CreateBPELexiconJob(
        base_lexicon_path=UppercaseLexiconJob(
            get_tedlium_bliss_lexicon(add_unknown_phoneme_and_mapping=False, add_silence=False),
        ).out_bliss_lexicon,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        subword_nmt_repo=subword_nmt_repo,
        unk_label="<unk>",
        vocab_blacklist=["</s>"],
        keep_special_lemmas=False,
    ).out_lexicon

    lexicon_ext = Lexicon()
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], phon=["<s>"], synt=["<s>"], special="sentence-begin"))
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-END]"], phon=["<s>"], synt=["</s>"], special="sentence-end"))
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<UNK>"], special="unknown"))

    if add_blank:
        lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], special="blank"))
        lexicon_ext.add_phoneme("<blank>", variation="none")

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    tedlium_lexicon_file = MergeLexiconJob([tedlium_lexicon_file, lexicon_ext_file]).out_bliss_lexicon

    return tedlium_lexicon_file
