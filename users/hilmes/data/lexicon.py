from i6_core.corpus import CorpusToTxtJob, CorpusReplaceOrthFromTxtJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_core.lib import lexicon
from i6_core.text import PipelineJob
from i6_experiments.common.datasets.librispeech import (
  get_g2p_augmented_bliss_lexicon_dict,
)
from i6_experiments.users.rossenbach.corpus.transform import (
  ApplyLexiconToTranscriptions,
)


def get_static_lexicon():
  """
  Add the phoneme and lemma entries for special TTS symbols

  :param bool include_punctuation:
  :return: the lexicon with special lemmas and phonemes
  :rtype: lexicon.Lexicon
  """
  lex = lexicon.Lexicon()

  lex.add_lemma(lexicon.Lemma(orth=["[space]"], phon=["[space]"]))
  lex.add_phoneme("[space]", variation="none")

  lex.add_lemma(lexicon.Lemma(orth=["[start]"], phon=["[start]"]))
  lex.add_phoneme("[start]", variation="none")

  lex.add_lemma(lexicon.Lemma(orth=["[end]"], phon=["[end]"]))
  lex.add_phoneme("[end]", variation="none")

  return lex


def get_lexicon():
  """
  creates the TTS specific Librispeech-100 lexicon using the default G2P pipeline including stress markers

  :return: bliss lexicon
  :rtype: Path
  """
  librispeech_100_lexicon = get_g2p_augmented_bliss_lexicon_dict(
    use_stress_marker=True
  )["train-clean-100"]
  static_bliss_lexcion = WriteLexiconJob(get_static_lexicon()).out_bliss_lexicon
  lexicon = MergeLexiconJob(
    bliss_lexica=[static_bliss_lexcion, librispeech_100_lexicon],
    sort_phonemes=True,
    sort_lemmata=False,
  ).out_bliss_lexicon
  return lexicon


def get_360_lexicon():
  """
  creates the TTS specific Librispeech-100 lexicon using the default G2P pipeline including stress markers

  :return: bliss lexicon
  :rtype: Path
  """
  librispeech_100_lexicon = get_g2p_augmented_bliss_lexicon_dict(
    use_stress_marker=True
  )["train-clean-360"]
  static_bliss_lexcion = WriteLexiconJob(get_static_lexicon()).out_bliss_lexicon
  lexicon = MergeLexiconJob(
    bliss_lexica=[static_bliss_lexcion, librispeech_100_lexicon],
    sort_phonemes=True,
    sort_lemmata=False,
  ).out_bliss_lexicon
  return lexicon


def process_corpus_text_with_lexicon(bliss_corpus, lexicon):
  """
  :param tk.Path bliss_corpus:
  :param tk.Path lexicon:
  :return: bliss_corpus
  :rtype: tk.Path
  """
  bliss_text = CorpusToTxtJob(bliss_corpus).out_txt

  add_start_command = "sed 's/^/[start] /g'"
  add_end_command = "sed 's/$/ [end]/g'"

  tokenized_text = PipelineJob(bliss_text, [add_start_command, add_end_command]).out
  processed_bliss_corpus = CorpusReplaceOrthFromTxtJob(
    bliss_corpus, tokenized_text
  ).out_corpus

  converted_bliss_corpus = ApplyLexiconToTranscriptions(
    processed_bliss_corpus, lexicon, word_separation_orth="[space]"
  ).out_corpus
  return converted_bliss_corpus
