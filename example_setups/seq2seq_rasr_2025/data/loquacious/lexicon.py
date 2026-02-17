from i6_core.lexicon import AddEowPhonemesToLexiconJob
from ..base import RemoveSpecialLemmasFromLexiconJob
from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_core.lexicon.modification import MergeLexiconJob, WriteLexiconJob
from i6_core.lib.lexicon import Lemma, Lexicon
from i6_experiments.common.datasets.loquacious.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.loquacious.vocab import get_subword_nmt_bpe
from sisyphus import tk

from ...tools import subword_nmt_repo


def get_bpe_bliss_lexicon(bpe_size: int, corpus_key: str, add_blank: bool, add_sentence_end_pron: bool) -> tk.Path:
    bpe_settings = get_subword_nmt_bpe(corpus_key=corpus_key, bpe_size=bpe_size)
    lexicon = get_bliss_lexicon(use_stress_marker=False, add_silence=False, add_unknown_phoneme_and_mapping=False)
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
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], phon=[""], synt=["<s>"], special="sentence-begin"))
    lexicon_ext.add_lemma(
        Lemma(
            orth=["[SENTENCE-END]"],
            phon=["<s>" if add_sentence_end_pron else ""],
            synt=["</s>"],
            special="sentence-end",
        )
    )
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<unk>"], special="unknown"))

    if add_blank:
        lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], synt=[], special="blank"))
        lexicon_ext.add_phoneme("<blank>", variation="none")

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    return MergeLexiconJob([bpe_lexicon_file, lexicon_ext_file]).out_bliss_lexicon


def get_bliss_phoneme_lexicon() -> tk.Path:
    lexicon_file = get_bliss_lexicon(add_silence=False, add_unknown_phoneme_and_mapping=False)
    cleaned_lexicon_file = RemoveSpecialLemmasFromLexiconJob(lexicon_file).out_lexicon
    eow_lexicon_file = AddEowPhonemesToLexiconJob(bliss_lexicon=cleaned_lexicon_file).out_lexicon

    lexicon_ext = Lexicon()
    lexicon_ext.add_phoneme("<unk>", variation="none")

    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], phon=[""], synt=["<s>"], special="sentence-begin"))
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-END]"], phon=[""], synt=["</s>"], special="sentence-end"))
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<unk>"], special="unknown"))

    lexicon_ext.add_phoneme("<blank>", variation="none")
    lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], synt=[], special="blank"))

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    return MergeLexiconJob([eow_lexicon_file, lexicon_ext_file]).out_bliss_lexicon
