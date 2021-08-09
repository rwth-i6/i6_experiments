"""

"""
import os

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from i6_core.corpus.convert import CorpusToTxtJob, CorpusReplaceOrthFromTxtJob
from i6_core.corpus.stats import ExtractOovWordsFromCorpusJob
from i6_core.g2p.train import TrainG2PModelJob
from i6_core.g2p.apply import ApplyG2PModelJob
from i6_core.g2p.convert import G2POutputToBlissLexiconJob
from i6_core.lexicon.cmu import DownloadCMUDictJob, CMUDictToBlissJob
from i6_core.lib import lexicon
from i6_core.text import PipelineJob
from i6_core.lexicon.conversion import LexiconFromTextFileJob
from i6_core.lexicon.modification import MergeLexiconJob, WriteLexiconJob

from i6_experiments.users.rossenbach.scripts.tokenizer import tokenizer_tts
from i6_experiments.users.rossenbach.g2p.number_conversion.tacotron import TacotronNumberConversionJob
from i6_experiments.users.rossenbach.text.pipeline_commands import *


def get_static_lexicon():
    """
    Add the phoneme and lemma entries for special and punctuation

    :param bool include_punctuation:
    :return: the lexicon with special lemmas and phonemes
    :rtype: lexicon.Lexicon
    """
    lex = lexicon.Lexicon()

    # def add_punctuation_lemma(lex, symbol):
    #     lex.add_lemma(
    #         lexicon.Lemma(
    #             orth=[symbol],
    #             phon=[symbol],
    #             synt=[symbol],
    #         )
    #     )
    #     lex.add_phoneme(symbol, variation="none")

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

    lex.add_lemma(lexicon.Lemma(orth=["."], phon=["[dot]"]))
    lex.add_phoneme("[dot]", variation="none")
    lex.add_lemma(lexicon.Lemma(orth=[","], phon=["[comma]"]))
    lex.add_phoneme("[comma]", variation="none")
    lex.add_lemma(lexicon.Lemma(orth=["?"], phon=["[question_mark]"]))
    lex.add_phoneme("[question_mark]", variation="none")
    lex.add_lemma(lexicon.Lemma(orth=["!"], phon=["[exclamation_mark]"]))
    lex.add_phoneme("[exclamation_mark]", variation="none")
    lex.add_lemma(lexicon.Lemma(orth=["-"], phon=["[hyphen]"]))
    lex.add_phoneme("[hyphen]", variation="none")
    lex.add_lemma(lexicon.Lemma(orth=['"'], phon=["[quotation]"]))
    lex.add_phoneme("[quotation]", variation="none")

    return lex


def get_uppercase_lexicon(create_alias_with_prefix=None):
    """
    Get the uppercase CMU dict that is used for the LJSpeech default setup

    :param create_alias_with_prefix:
    :return: a text-based CMU lexicon (uppercased)
    :rtype: Path
    """
    download_cmu_lexicon_job = DownloadCMUDictJob()
    cmu_lexicon_lowercase = download_cmu_lexicon_job.out_cmu_lexicon
    convert_cmu_lexicon_uppercase_job = PipelineJob(
        cmu_lexicon_lowercase, ["tr '[:lower:]' '[:upper:]'", "sed \"s/([0-9])//g\"", "sed \"s/#.*//g\""],
        zip_output=False,
        mini_task=True)
    cmu_lexicon_uppercase = convert_cmu_lexicon_uppercase_job.out

    if create_alias_with_prefix:
        path_prefix = os.path.join(create_alias_with_prefix, "LJSpeech", "G2P")
        download_cmu_lexicon_job.add_alias(os.path.join(path_prefix, "download_cmu"))
        convert_cmu_lexicon_uppercase_job.add_alias(os.path.join(path_prefix, "convert_cmu"))

    return cmu_lexicon_uppercase


def get_uppercase_cmu_g2p(create_alias_with_prefix=None):
    """
    Create a default Sequitur-G2P model based on the CMU dict

    This G2P model can only be used with "uppercase" data

    :param create_alias_with_prefix:
    :return: a sequitur model trained on the uppercased CMU lexicon
    :rtype: Path
    """
    cmu_lexicon_uppercase = get_uppercase_lexicon(create_alias_with_prefix)

    train_cmu_default_sequitur = TrainG2PModelJob(cmu_lexicon_uppercase)
    sequitur_cmu_uppercase_model = train_cmu_default_sequitur.out_best_model

    if create_alias_with_prefix:
        path_prefix = os.path.join(create_alias_with_prefix, "LJSpeech", "G2P")
        train_cmu_default_sequitur.add_alias(os.path.join(path_prefix, "train_sequitur"))

    return sequitur_cmu_uppercase_model


def apply_uppercase_cmu_corpus_processing(bliss_corpus):
    """

    :param Path bliss_corpus:
    :param path_prefix:
    :return:
    """

    cmu_wordlist_lexicon = get_uppercase_lexicon()
    tk.register_output("cmu_lexicon_test.txt", cmu_wordlist_lexicon)
    cmu_bliss_lexicon = LexiconFromTextFileJob(cmu_wordlist_lexicon).out_bliss_lexicon
    static_bliss_lexcion = WriteLexiconJob(get_static_lexicon()).out_bliss_lexicon
    cmu_bliss_lexicon = MergeLexiconJob(bliss_lexica=[static_bliss_lexcion, cmu_bliss_lexicon],
                                        sort_phonemes=True,
                                        sort_lemmata=False).out_bliss_lexicon
    cmu_uppercase_g2p = get_uppercase_cmu_g2p()

    bliss_text = CorpusToTxtJob(bliss_corpus).out_txt

    whitelist_filter_list = list(".,?!-")
    whitelist_pipe = get_whitelist_filter_pipe(allow_list=whitelist_filter_list)
    tokenizer_command = DelayedFormat("{} -a -l en -no-escape", tokenizer_tts)
    tokenized_text = PipelineJob(
        bliss_text,
        [whitelist_pipe,
         tokenizer_command,
         "tr '[:lower:]' '[:upper:]'",
         pipe_normalize_double_hyphen,
         pipe_split_dots,
         pipe_squeeze_repeating_whitespaces]).out
    processed_bliss_corpus = CorpusReplaceOrthFromTxtJob(bliss_corpus, tokenized_text).out_corpus

    tk.register_output("ljspeech_test/processed_corpus.xml.gz", processed_bliss_corpus)

    oovs = ExtractOovWordsFromCorpusJob(processed_bliss_corpus, cmu_bliss_lexicon).out_oov_words
    g2p_oov_lexicon = ApplyG2PModelJob(cmu_uppercase_g2p, oovs).out_g2p_lexicon
    complete_bliss_lexcion = G2POutputToBlissLexiconJob(cmu_bliss_lexicon, g2p_oov_lexicon, merge=True).out_oov_lexicon

    tk.register_output("ljspeech_test/complete-lexicon.xml.gz", complete_bliss_lexcion)

    # phoneme_text = ApplySequitur(tokenized_text, cmu_lexicon_uppercase, sequitur_cmu_uppercase_v1).out
    # bliss_corpus = BlissMergeText(bliss_corpus, phoneme_text).out
    # # text processing
    # process_text_job = ProcessBlissText(bliss_corpus, [('uppercase', {}),
    #                                                    ('end_token', {'token': ' ~'})],
    #                                     vocabulary=self.vocabulary,
    #                                     vocabulary_mode="word")
    # self.job_alias(process_text_job, "text_preprocessing_%s" % name)
    # bliss_corpus = process_text_job.out






