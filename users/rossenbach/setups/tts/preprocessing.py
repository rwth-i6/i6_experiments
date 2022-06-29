from sisyphus import tk

from i6_core.lib import lexicon

from i6_core.corpus.convert import CorpusToTxtJob, CorpusReplaceOrthFromTxtJob
from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.lexicon.modification import MergeLexiconJob, WriteLexiconJob
from i6_core.text.processing import PipelineJob


def get_tts_extension_lexicon():
    """
    Add the phoneme and lemma entries for special TTS symbols

    :param bool include_punctuation:
    :return: the lexicon with special lemmas and phonemes
    :rtype: lexicon.Lexicon
    """
    lex = lexicon.Lexicon()

    lex.add_lemma(
        lexicon.Lemma(orth=["[space]"], phon=["[space]"])
    )
    lex.add_phoneme("[space]", variation="none")

    lex.add_lemma(
        lexicon.Lemma(orth=["[start]"], phon=["[start]"])
    )
    lex.add_phoneme("[start]", variation="none")

    lex.add_lemma(
        lexicon.Lemma(orth=["[end]"], phon=["[end]"])
    )
    lex.add_phoneme("[end]", variation="none")

    return lex


def extend_lexicon(lexicon: tk.Path) -> tk.Path:
    """
    Extends a bliss lexicon with TTS specific makers

    :return: bliss lexicon
    :rtype: Path
    """
    static_bliss_lexcion = WriteLexiconJob(get_tts_extension_lexicon()).out_bliss_lexicon
    lexicon = MergeLexiconJob(bliss_lexica=[static_bliss_lexcion, lexicon],
                              sort_phonemes=True,
                              sort_lemmata=False).out_bliss_lexicon
    return lexicon


def process_corpus_text_with_extended_lexicon(bliss_corpus: tk.Path, lexicon: tk.Path) -> tk.Path:
    """
    Apply the lexicon to a corpus file, and insert [start], [end] and [space] tokens.

    :param bliss_corpus: A bliss corpus for conversion
    :param lexicon: A lexicon that includes the TTS markers, e.g. use `extend_lexicon`
    """
    bliss_text = CorpusToTxtJob(bliss_corpus).out_txt

    add_start_command = "sed 's/^/[start] /g'"
    add_end_command = "sed 's/$/ [end]/g'"

    tokenized_text = PipelineJob(
        bliss_text,
        [add_start_command,
         add_end_command]).out
    processed_bliss_corpus = CorpusReplaceOrthFromTxtJob(bliss_corpus, tokenized_text).out_corpus

    converted_bliss_corpus = ApplyLexiconToCorpusJob(processed_bliss_corpus, lexicon, word_separation_orth="[space]").out_corpus
    return converted_bliss_corpus
