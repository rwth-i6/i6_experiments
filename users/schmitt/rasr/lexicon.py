from sisyphus import *

import subprocess
import os
import sys
import shutil
import ast
import xml.etree.ElementTree as ET
from typing import Optional, List

import i6_core.lib.corpus as libcorpus
from i6_core.lib.lexicon import Lexicon, Lemma
import i6_core.util as util


class JsonSubwordVocabToLemmaLexiconJob(Job):
  """
  Takes a json vocab file (mapping from subword to idx) and creates a lexicon with one lemma for each subword.
  The resulting lexicon only has lemmas and no phonemes.
  """

  def __init__(self, json_vocab_path, blank_idx):
    self.blank_idx = blank_idx
    self.json_vocab_path = json_vocab_path
    self.out_lexicon = self.output_path("out_lexicon")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 1, "time": 1}, mini_task=True)

  @staticmethod
  def get_special_lemma_dict(
          blank: Optional[str], unknown: Optional[str], sos: Optional[str], eos: Optional[str],
  ):
    special_lemma_dict = []
    if blank:
      special_lemma_dict.append(('silence', blank, [blank]))
    if unknown:
      special_lemma_dict.append(('unknown', "<UNK>", [unknown]))
    if sos:
      special_lemma_dict.append(('sentence-begin', sos, None))
    if eos:
      special_lemma_dict.append(('sentence-end', eos, None))

    return special_lemma_dict


  @staticmethod
  def make_base_lexicon_xml(phoneme_list=None, special_lemma_dict=None) -> Lexicon:

    lexicon = Lexicon()
    # list of phonemes(or bpes)
    if phoneme_list is not None:
      for phon in phoneme_list:
        lexicon.add_phoneme(phon, "none")

    # special lemma #
    if special_lemma_dict is not None:
      for sp_lemma, token, pron in special_lemma_dict:
        lexicon.add_lemma(Lemma(orth=[token], phon=pron, special=sp_lemma, synt=[token]))

    return lexicon

  def run(self):
    special_lemma_dict = self.get_special_lemma_dict(unknown=None, sos="<s>", eos="</s>", blank=None)
    lexicon = self.make_base_lexicon_xml(special_lemma_dict=special_lemma_dict)
    with open(self.json_vocab_path.get_path(), "r") as f:
      vocab = ast.literal_eval(f.read())
    special_tokens = ['<s>', '</s>', '<unk>']

    for v in sorted(vocab.keys()):
      if v in special_tokens:
        continue
      lexicon.add_lemma(Lemma(orth=[v]))

    elem = lexicon.to_xml()
    tree = ET.ElementTree(elem)
    util.write_xml(self.out_lexicon.get_path(), tree)


class BlissCorpusToBpeLexiconJob(Job):
  """
    Takes a bliss corpus and BPE data and creates a lexicon with BPE units as phonemes and words as lemmas with
    corresponding BPE segmentations.
  """

  def __init__(
          self,
          word_list_path: Path,
          bpe_codes: Path,
          bpe_vocab: Path,
          subword_nmt_repo: Path,
          unk_token: str = "<UNK>",
          blank_token: str = "[blank]",
          sos_token: str = "<s>",
          eos_token: str = "</s>",
  ):
    self.word_list_path = word_list_path
    self.bpe_codes = bpe_codes
    self.bpe_vocab = bpe_vocab
    self.subword_nmt_repo = subword_nmt_repo

    # the tokens which we want to add to the lexicon
    self.unk_token = unk_token
    self.blank_token = blank_token
    self.sos_token = sos_token
    self.eos_token = eos_token

    # the tokens which we want to exclude from the lexicon but which are in the bpe vocab
    self.vocab_special_tokens = ['<s>', '</s>', '<unk>']

    self.out_lexicon = self.output_path("out_lexicon")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 1, "time": 1}, mini_task=True)

  def run(self):
    # create base lexicon with bpe units as phonemes
    with open(self.bpe_vocab.get_path(), "r") as f:
      bpe_units = list(ast.literal_eval(f.read()).keys())

    bpe_units = [b for b in bpe_units if b not in self.vocab_special_tokens]
    bpe_phonemes = [self.blank_token, self.unk_token] + bpe_units

    special_lemma_dict = JsonSubwordVocabToLemmaLexiconJob.get_special_lemma_dict(
      blank=self.blank_token, unknown=self.unk_token, sos=self.sos_token, eos=self.eos_token)
    lexicon = JsonSubwordVocabToLemmaLexiconJob.make_base_lexicon_xml(
      phoneme_list=bpe_phonemes, special_lemma_dict=special_lemma_dict)

    # get all words from corpora
    words = set()
    with open(self.word_list_path, "r") as f:
      for line in f:
        words.add(line.strip())
    # for bliss_corpus in self.bliss_corpora:
    #   corpus = libcorpus.Corpus()
    #   corpus.load(bliss_corpus.get_path())
    #   for s in corpus.segments():
    #     for w in s.orth.strip().split():
    #       words.add(w)
    words = list(words)  # convert to list to maintain order

    # write word vocab to file for the subword nmt repo
    with open("subword_nmt_word_vocab", "w+") as f:
      for w in words:
        f.write(w + "\n")
    # write bpe vocab (without special tokens) to file for the subword nmt repo
    with open("subword_nmt_bpe_vocab", "w+") as f:
      for b in bpe_units:
        f.write(b + " -1\n")

    apply_binary = os.path.join(
      self.subword_nmt_repo.get_path(), "apply_bpe.py"
    )
    args = [
      sys.executable,
      apply_binary,
      "--input",
      "subword_nmt_word_vocab",
      "--codes",
      self.bpe_codes.get_path(),
      "--vocabulary",
      "subword_nmt_bpe_vocab",
      "--output",
      "bpe_segmentation",
    ]
    subprocess.run(args, check=True)

    with open("bpe_segmentation", "r") as f:
      bpe_segmentations = [l.strip() for l in f]
    word_to_bpe_segmentation = dict(zip(words, bpe_segmentations))

    # lexicon.add_lemma(Lemma(orth=[self.blank_token], phon=[self.blank_token], special="blank"))
    # lexicon.add_lemma(Lemma(special="unknown", phon=[self.unk_token]))
    # lexicon.add_lemma(Lemma(special="sentence-begin"))
    # lexicon.add_lemma(Lemma(special="sentence-end"))

    for w in words:
      lexicon.add_lemma(Lemma(orth=[w], phon=[word_to_bpe_segmentation[w]], synt=[w]))

    elem = lexicon.to_xml()
    tree = ET.ElementTree(elem)
    util.write_xml(self.out_lexicon.get_path(), tree)