# As said 'Christoph' is author This just exists here as *exact* copy so the setup runs out of the box!
# Author: Christoph M. Luescher <luescher@cs.rwth-aachen.de>

__all__ = ["CORPUS_PATH", "DURATIONS", "CONCURRENT", "OFFICIAL_CORPORA", "CORPORA",
           "CORPORA_ALLOWED_TRAINING", "MERGED_AUDIO_PATH",
           "CreateMergedBlissCorpus", "GetPhonemes", "GetCharacters",
           "CreateCharBlissLexicon", "ConvertToBlissLexicon",
           "PHONEME_CHAR_MAPPING", "ConvertAllophoneSet",
           "OFFICIAL_LANGUAGE_MODELS", "OFFICIAL_LEXICON", "OFFICIAL_VOCABULARY",
           "CART_PHONEMES", "CART_PHONEME_STEPS",
           "CART_CHARACTERS", "CART_CHARACTER_STEPS"]

import collections
import glob
import os

from i6_core.lib.corpus import Corpus, Speaker, Recording, Segment
from i6_core.util import uopen

from sisyphus import *
import sisyphus.toolkit as tk
Path = setup_path(__package__)

# -------------------- Settings --------------------
CORPUS_PATH = "/u/corpora/speech/LibriSpeech/LibriSpeech/"

DURATIONS = {
  "train-clean-100": 100.6,
  "train-clean-360": 363.7,
  "train-clean-460": 464.3,
  "train-other-500": 496.7,
  "train-other-960": 961.0,
  "dev-clean": 5.4,
  "dev-other": 5.3,
  "dev": 10.7,
  "train-clean-100-dev": 111.3,
  "train-dev": 971.7,
  "test-clean": 5.4,
  "test-other": 5.1,
  "test": 10.5,
}

CONCURRENT = dict()

for name, duration in DURATIONS.items():
  CONCURRENT[name] = min(max(int(round(1.5*duration, -1)), 10), 300)

OFFICIAL_CORPORA = [
  "train-clean-100",
  "train-clean-360",
  "train-other-500",
  "dev-clean",
  "dev-other",
  "test-clean",
  "test-other",
]

CORPORA = [
  "train-clean-100",
  "train-clean-360",
  "train-clean-460",
  "train-other-500",
  "train-other-960",
  "dev-clean",
  "dev-other",
  "dev",
  "train-clean-100-dev",
  "train-dev",
  "test-clean",
  "test-other",
  "test",
]

CORPORA_ALLOWED_TRAINING = [
  "train-clean-100",
  "train-clean-360",
  "train-clean-460",
  "train-other-500",
  "train-other-960",
  "dev-clean",
  "dev-other",
  "dev",
  "train-clean-100-dev",
  "train-dev",
]

CORPUS_PATH = "/u/corpora/speech/LibriSpeech/LibriSpeech/"
MERGED_AUDIO_PATH = "/u/corpora/speech/LibriSpeech/audio-merged/"
SPEAKER_PATH = "/u/corpora/speech/LibriSpeech/LibriSpeech/SPEAKERS.TXT"


# ******************** Jobs ********************


class CreateMergedBlissCorpus(Job):
  def __init__(self, corpus_name,
               corpus_path=CORPUS_PATH,
               speaker_path=SPEAKER_PATH,
               audio_path=MERGED_AUDIO_PATH,
               audio_format="wav"):
    self.set_vis_name("Create Librispeech Bliss Corpus")

    self.corpus_name = corpus_name
    self.corpus_path = corpus_path
    self.speaker_path = speaker_path

    self.audio_path = audio_path
    self.audio_format = audio_format

    self.search_dirs = list()
    self.audio_files = list()

    self.corpus = Corpus()
    self.corpus.name = corpus_name

    self.corpus_file = self.output_path("corpus.xml.gz")

  def tasks(self):
    yield Task('run', rqmt={'cpu': 1, 'mem': 2, 'time': 2})

  def run(self):
    self.get_search_dirs()
    self.get_audio_files()
    self.get_recordings()
    self.dump()

  def get_search_dirs(self):
    if self.corpus_name in OFFICIAL_CORPORA:
        self.search_dirs.append(self.corpus_name)
    elif self.corpus_name in CORPORA:
        if self.corpus_name == "train-clean-460":
          self.search_dirs.append("train-clean-100")
          self.search_dirs.append("train-clean-360")
        elif self.corpus_name == "train-other-960":
          self.search_dirs.append("train-clean-100")
          self.search_dirs.append("train-clean-360")
          self.search_dirs.append("train-other-500")
        elif self.corpus_name == "dev":
          self.search_dirs.append("dev-clean")
          self.search_dirs.append("dev-other")
        elif self.corpus_name == "train-clean-100-dev":
          self.search_dirs.append("train-clean-100")
          self.search_dirs.append("dev-clean")
          self.search_dirs.append("dev-other")
        elif self.corpus_name == "train-dev":
          self.search_dirs.append("train-clean-100")
          self.search_dirs.append("train-clean-360")
          self.search_dirs.append("train-other-500")
          self.search_dirs.append("dev-clean")
          self.search_dirs.append("dev-other")
        elif self.corpus_name == "test":
          self.search_dirs.append("test-clean")
          self.search_dirs.append("test_other")
    else:
        raise NotImplementedError

  def get_audio_files(self):
    for subcorpus in self.search_dirs:
      path = os.path.join(self.audio_path, subcorpus, f"*.{self.audio_format}")
      paths = glob.glob(path)
      self.audio_files.extend(paths)

  def get_recordings(self):
    for path in self.audio_files:
      assert os.path.isfile(path), f"file does not exist: {path}"
      cur_dir = os.path.dirname(path)
      corpus_name = cur_dir.split("/")[-1]
      fn_suffix = os.path.basename(path)
      fn, suffix = os.path.splitext(fn_suffix)

      rec = Recording()
      rec.name = fn
      rec.audio = path

      spk_id, spk = self._get_speaker(fn)

      if spk_id not in rec.speakers.keys():
        rec.speakers[spk_id] = spk

      self.corpus.speakers[spk_id] = spk

      rec.segments = self._get_segments(corpus_name, fn)
      self.corpus.recordings.append(rec)

  def _get_speaker(self, fn):
    spk = Speaker()

    with open(self.speaker_path, "rt") as spk_file:
      for line in spk_file:
        if line.startswith(";"):
            continue
        line = line.split("|")
        name = line[0].strip()
        n = fn.split("-")[0]
        if name == n:
          spk.name = name
          gender_short = line[1].strip().lower()
          gender = "male" if gender_short == "m" else "female"
          spk.attribs['gender'] = f"{gender}"

    return spk.name,  spk

  def _get_segments(self, corpus_name, fn):
    segments = list()  # of segments
    spk_id, talk_id = fn.split("-")

    durations = dict()
    durations_path = os.path.join(self.audio_path, f"seg.{corpus_name}.txt")
    with open(durations_path, "rt") as dur_file:
      for line in dur_file:
        if not line.strip():
          continue
        name, dur = line.split()
        name = name.split(".")[0]
        dur = float(dur)
        durations[name] = dur

    trans_path = os.path.join(self.corpus_path, corpus_name,
                              spk_id, talk_id, f"{spk_id}-{talk_id}.trans.txt")
    cur_time = 0.0
    with open(trans_path, "rt") as trans_file:
      for line in trans_file:
        name, trans = line.split(maxsplit=1)
        offset = durations[name]
        segm = Segment()
        segm.name = name
        segm.track = int(name.split("-")[-1])
        segm.speaker_name = spk_id
        segm.orth = trans.strip()
        segm.start = cur_time
        cur_time += offset
        segm.end = cur_time
        segments.append(segm)

    return segments

  def dump(self):
    self.corpus.dump(self.corpus_file.get_path())


OFFICIAL_LEXICON = "/u/corpora/speech/LibriSpeech/librispeech-lexicon.txt"
OFFICIAL_VOCABULARY = "/u/corpora/speech/LibriSpeech/librispeech-vocab.txt"


class GetPhonemes(Job):
  def __init__(self, path=OFFICIAL_LEXICON):
    self.set_vis_name("Get phoneme set from lexicon")
    self.path = path
    self.out = self.output_path("phonemes.txt")
    self.out_sorted = self.output_path("phonemes_sorted.txt")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    path = tk.uncached_path(self.path)

    unique_phonemes = set()

    with open(path, "rt") as file:
      for line in file:
        line = line.split(maxsplit=1)
        phonemes = line[1].split()
        for p in phonemes:
          unique_phonemes.add(p.strip())

    unique_phonemes = sorted(list(unique_phonemes))

    with open(self.out.get_path(), "wt") as f1:
      for up in unique_phonemes:
        print(up, file=f1)

    len3 = [x for x in unique_phonemes if len(x) == 3]
    with open(self.out_sorted.get_path(), "wt") as f2:
      for up in unique_phonemes:
        if len(up) > 2:
          assert up[2].isdigit(), "phoneme error"
          matches = [x for x in len3 if x[0:2] == up[0:2]]
          len3 = [x for x in len3 if x not in matches]
          if matches:
            print(f"{up[0:2]}", " ".join(matches), file=f2)
        else:
          print(f"{up}", file=f2)


IGNORE_CHARACTERS = ["'"]


class GetCharacters(Job):
  def __init__(self, path=OFFICIAL_VOCABULARY,
               ignore_characters=IGNORE_CHARACTERS):
    self.set_vis_name("Get character set from vocabulary")
    self.path = path
    self.ignore_characters = ignore_characters
    self.out = self.output_path("characters.txt")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    unique_characters = set()

    with uopen(self.path, "rt") as f:
      for line in f:
        for c in line:
          c = c.strip()
          if c:
            unique_characters.add(c)

    unique_characters = sorted(list(unique_characters))
    unique_characters_reduced = [x for x in unique_characters if x not in self.ignore_characters]

    with open(self.out.get_path(), "wt") as f:
      for c in unique_characters_reduced:
        print(c, file=f)


class Phoneme:
  def __init__(self, symbol, variation=None):
    self.symbol = symbol
    self.variation = variation

  def dump(self, out, identation="    "):
    out.write(f"{identation}<phoneme>\n")
    out.write(f"{identation}  <symbol>{self.symbol}</symbol>\n")
    if self.variation is not None:
      out.write(f"{identation}  <variation>{self.variation}</variation>\n")
    out.write(f"{identation}</phoneme>\n")


class Lemma:
      def __init__(self, orth, phon=None, score=[0.0],
                   special=None, synt=None, evl=None):
        self.orth = orth
        self.phon = phon
        self.score = score
        self.special = special
        self.synt = synt
        self.evl = evl

      def dump(self, out, indentation="  "):
        if self.special is not None:
          out.write(f'{indentation}<lemma special="{self.special}">\n')
        else:
          out.write(f"{indentation}<lemma>\n")
        out.write(f"{indentation}  <orth>{self.orth}</orth>\n")

        if self.phon is not None:
          for idx, p in enumerate(self.phon):
            out.write(f'{indentation}  <phon score="{self.score[idx]}">\n')
            out.write(f"{indentation}    {p}\n")
            out.write(f"{indentation}  </phon>\n")

        if self.special is not None:
          if self.synt is not None:
            out.write(f'{indentation}  <synt>\n')
            out.write(f"{indentation}    <tok>{self.synt}</tok>\n")
            out.write(f"{indentation}  </synt>\n")
          else:
            out.write(f"{indentation}  <synt/>\n")
          out.write(f"{indentation}  <eval/>\n")

        out.write(f"{indentation}</lemma>\n")


class CreateCharBlissLexicon(Job):
  def __init__(self, char_file, vocab_file=OFFICIAL_VOCABULARY):
    self.set_vis_name("Create Librispeech Character Bliss Corpus")
    self.char_file = char_file
    self.vocab_file = vocab_file
    self.out = self.output_path("lexicon.txt.gz")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    characters = list()
    allowed_chars = list()
    with uopen(self.char_file, "rt") as file:
      for line in file:
        line = line.strip()
        characters.append(Phoneme(line))
        allowed_chars.append(line)
    characters.append(Phoneme("[SILENCE]", variation="none"))
    characters.append(Phoneme("[UNKNOWN]", variation="none"))

    lemmas = list()

    lemmas.append(Lemma(orth="[SILENCE]", phon=["[SILENCE]"], special="silence"))
    lemmas.append(Lemma(orth="[UNKNOWN]", phon=["[UNKNOWN]"], synt="&lt;UNK&gt;", special="unknown"))
    lemmas.append(Lemma(orth="[sentence-begin]", synt="&lt;s&gt;", special="sentence-begin"))
    lemmas.append(Lemma(orth="[sentence-end]", synt="&lt;/s&gt;", special="sentence-end"))

    with uopen(self.vocab_file, "rt") as file:
      for orth in file:
        char_list = [x for x in orth if x in allowed_chars]
        orth = orth.strip()
        chars = " ".join(char_list)
        lemmas.append(Lemma(orth=orth, phon=[chars]))

    with open(self.out.get_path(), "wt") as out:
      out.write('<?xml version="1.0" encoding="utf-8"?>\n')
      out.write("<lexicon>\n")

      out.write("  <phoneme-inventory>\n")
      for c in characters:
        c.dump(out)
      out.write("  </phoneme-inventory>\n\n")

      for l in lemmas:
        l.dump(out)

      out.write("</lexicon>\n")


class ConvertToBlissLexicon(Job):
  def __init__(self, phonemes, lexicon_file=OFFICIAL_LEXICON):
    self.set_vis_name("Convert Librispeech Corpus: Official to Bliss")
    self.phonemes = phonemes
    self.lexicon_file = lexicon_file
    self.out = self.output_path("lexicon.txt.gz")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    phonemes = list()
    allowed_chars = list()
    with uopen(self.phonemes, "rt") as file:
      for line in file:
        line = line.strip()
        phonemes.append(Phoneme(line))
        allowed_chars.append(line)
    phonemes.append(Phoneme("[SILENCE]", variation="none"))
    phonemes.append(Phoneme("[UNKNOWN]", variation="none"))

    lemmas = list()

    lemmas.append(Lemma(orth="[SILENCE]", phon=["[SILENCE]"], special="silence"))
    lemmas.append(Lemma(orth="[UNKNOWN]", phon=["[UNKNOWN]"], synt="&lt;UNK&gt;", special="unknown"))
    lemmas.append(Lemma(orth="[sentence-begin]", synt="&lt;s&gt;", special="sentence-begin"))
    lemmas.append(Lemma(orth="[sentence-end]", synt="&lt;/s&gt;", special="sentence-end"))

    lemmas_tmp = collections.defaultdict(list)
    with uopen(self.lexicon_file, "rt") as file:
      for line in file:
        line = line.strip()
        line = line.split(maxsplit=1)
        orth = line[0]
        pronunciation = line[1]
        if pronunciation not in lemmas_tmp[orth]:
          lemmas_tmp[orth].append(pronunciation)

    for k, v in lemmas_tmp.items():
      s = [0.0 for x in v]
      lemmas.append(Lemma(orth=k, phon=v, score=s))

    with open(self.out.get_path(), "wt") as out:
      out.write('<?xml version="1.0" encoding="utf-8"?>\n')
      out.write("<lexicon>\n")

      out.write("  <phoneme-inventory>\n")
      for p in phonemes:
        p.dump(out)
      out.write("  </phoneme-inventory>\n\n")

      for l in lemmas:
        l.dump(out)

      out.write("</lexicon>\n")


PHONEME_CHAR_MAPPING = {
  'AA0': "A", 'AA1': "A", 'AA2': "A",
  'AE0': "A", 'AE1': "A", 'AE2': "A",
  'AH0': "A", 'AH1': "A", 'AH2': "A",
  'AO0': "A", 'AO1': "A", 'AO2': "A",
  'AW0': "A", 'AW1': "A", 'AW2': "A",
  'AY0': "A", 'AY1': "A", 'AY2': "A",
  'B': "B", 'CH': "C", 'D': "D", 'DH': "D",
  'EH0': "E", 'EH1': "E", 'EH2': "E",
  'ER0': "E", 'ER1': "E", 'ER2': "E",
  'EY0': "E", 'EY1': "E", 'EY2': "E",
  'F': "F", 'G': "G", 'HH': "H",
  'IH0': "I", 'IH1': "I", 'IH2': "I",
  'IY0': "I", 'IY1': "I", 'IY2': "I",
  'JH': "J", 'K': "K", 'L': "L", 'M': "M", 'N': "N", 'NG': "N",
  'OW0': "O", 'OW1': "O", 'OW2': "O",
  'OY0': "O", 'OY1': "O", 'OY2': "O",
  'P': "P", 'R': "R", 'S': "S", 'SH': "S", 'T': "T", 'TH': "T",
  'UH0': "U", 'UH1': "U", 'UH2': "U",
  'UW0': "U", 'UW1': "U", 'UW2': "U",
  'V': "V", 'W': "W", 'Y': "Y", 'Z': "Z", 'ZH': "Z",
  '[SILENCE]': "[SILENCE]", '[UNKNOWN]': "[UNKNOWN]", '#': "#",
}


class ConvertAllophoneSet(Job):
  def __init__(self, allophones, mapping=PHONEME_CHAR_MAPPING):
    self.set_vis_name("Convert Allophone Set")

    self.allophones = allophones
    self.mapping = mapping

    self.phonemes = list(mapping.keys())
    self.characters = list(mapping.values())

    self.out = self.output_path("allophones")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    file_read = uopen(self.allophones, "rt")
    file_write = open("allophones", "wt")

    for line in file_read:
      if line.startswith("#"):
        continue
      allo, line = line.split("{", maxsplit=1)
      left_context, line = line.split("+", maxsplit=1)
      right_context, line = line.split("}", maxsplit=1)
      print(f"{self.mapping[allo]}{{{self.mapping[allo]}}}+{{{self.mapping[allo]}}}{line}", file=file_write)

    file_read.close()
    file_write.close()

    _relink("allophones", self.out.get_path())


def _relink(src, dst):
  if os.path.exists(dst):
    os.remove(dst)
  os.link(src, dst)


# -------------------- Resources --------------------

RESOURCES_PATH = "/u/corpora/speech/LibriSpeech/"

OFFICIAL_LANGUAGE_MODELS = {
    '3gram': Path(f"{RESOURCES_PATH}lm/3-gram.arpa.gz", cached=True),
    '4gram': Path(f"{RESOURCES_PATH}lm/4-gram.arpa.gz", cached=True),
}


# -------------------- CART --------------------

CART_PHONEMES = ["#", "[SILENCE]", "[UNKNOWN]", "AA0", "AA1", "AA2", "AE0",
                 "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0",
                 "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0",
                 "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F",
                 "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K",
                 "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
                 "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW0",
                 "UW1", "UW2", "V", "W", "Y", "Z", "ZH"]

MIN_OBS = 500

CART_PHONEME_STEPS = [
  {
    'name': "silence",
    'action': "cluster",
    'questions': [{'type': "question",
                   'description': "silence",
                   'key': "central",
                   'value': "[SILENCE]"}]
  },
  {
    'name': 'central',
    'action': 'partition',
    'min-obs': MIN_OBS,
    'questions': [
      {'type': "for-each-value", 'questions': [{'type': "question",
                                                'description': "central-phone",
                                                'key': "central",
                                                'values': "AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW0 UW1 UW2 V W Y Z ZH"}]},
      {'type': "question",
       'description': "noise",
       'key': "central",
       'value': "[UNKNOWN]"},
    ]
  },
  {
    'name': "hmm-state",
    'action': "partition",
    'min-obs': MIN_OBS,
    'questions': [{'type': "for-each-value", 'questions': [{'type': "question",
                                                            'description': "hmm-state",
                                                            'key': "hmm-state"}]}],
  },
  {
    'name': 'linguistics',
    'min-obs': MIN_OBS,
    'questions': [{'type': "for-each-value", 'questions': [{'type': "question",
                                                            'description': "boundary",
                                                            'key': "boundary" }]},
                  {'type': "for-each-key",
                   'keys': "history[0] central future[0]",
                   'questions': [{'type': "for-each_value",
                                  'questions': [{'type': "question", 'description': "context-phone", 'values': "# AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW0 UW1 UW2 V W Y Z ZH"},
                                                {'type': "question", 'description': "CONSONANT", 'values': "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH"},
                                                {'type': "question", 'description': "CONSONANT-OBSTRUENT", 'values': "B CH D DH F G HH JH K P S SH T TH V Z ZH"},
                                                {'type': "question", 'description': "CONSONANT-OBSTRUENT-PLOSIVE", 'values': "B D G K P T"},
                                                {'type': "question", 'description': "CONSONANT-OBSTRUENT-AFFRICATE", 'values': "CH JH"},
                                                {'type': "question", 'description': "CONSONANT-OBSTRUENT-FRICATIV", 'values': "DH F HH S SH TH V Z ZH"},
                                                {'type': "question", 'description': "CONSONANT-SONORANT", 'values': "L  M N  NG R W Y"},
                                                {'type': "question", 'description': "CONSONANT-SONORANT-NASAL", 'values': "M N  NG"},
                                                {'type': "question", 'description': "CONSONANT-SONORANT-LIQUID", 'values': "R L"},
                                                {'type': "question", 'description': "CONSONANT-SONORANT-GLIDE", 'values': "W Y"},
                                                {'type': "question", 'description': "CONSONANT-APPROX", 'values': "R Y"},
                                                {'type': "question", 'description': "CONSONANT-BILABIAL", 'values': "P B M"},
                                                {'type': "question", 'description': "CONSONANT-LABIODENTAL", 'values': "F V"},
                                                {'type': "question", 'description': "CONSONANT-DENTAL", 'values': "TH DH"},
                                                {'type': "question", 'description': "CONSONANT-ALVEOLAR", 'values': "T D N S Z R L"},
                                                {'type': "question", 'description': "CONSONANT-POSTALVEOLAR", 'values': "SH ZH"},
                                                {'type': "question", 'description': "CONSONANT-VELAR", 'values': "K G NG"},
                                                {'type': "question", 'description': "VOWEL", 'values': "AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2  AY0 AY1 AY2 EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 IH0 IH1 IH2 IY0 IY1 IY2 OW0 OW1 OW2 OY0 OY1 OY2 UH0 UH1 UH2 UW0 UW1 UW2"},
                                                {'type': "question", 'description': "VOWEL-CHECKED", 'values': "AE0 AE1 AE2 AH0 AH1 AH2 EH0 EH1 EH2 IH0 IH1 IH2 UH0 UH1 UH2"},
                                                {'type': "question", 'description': "VOWEL-FREE", 'values': "AA0 AA1 AA2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 ER0 ER1 ER2 EY0 EY1 EY2 IY0 IY1 IY2 OW0 OW1 OW2 OY0 OY1 OY2 UW0 UW1 UW2"},
                                                {'type': "question", 'description': "VOWEL-FREE-PHTHONGS1", 'values': "AY0 AY1 AY2 EY0 EY1 EY2 IY0 IY1 IY2 OY0 OY1 OY"},
                                                {'type': "question", 'description': "VOWEL-FREE-PHTHONGS2", 'values': "AW0 AW1 AW2 OW0 OW1 OW2 UW0 UW1 UW2"},
                                                {'type': "question", 'description': "VOWEL-FREE-PHTHONGS3", 'values': "AA0 AA1 AA2 AO0 AO1 AO2 ER0 ER1 ER2"},
                                                {'type': "question", 'description': "VOWEL-CLOSE", 'values': "IY0 IY1 IY2 UW0 UW1 UW2 IH0 IH1 IH2 UH0 UH1 UH2"},
                                                {'type': "question", 'description': "VOWEL-OPEN", 'values': "EH0 EH1 EH2 ER0 ER1 ER2 AH0 AH1 AH2 AO0 AO1 AO2 AE0 AE1 AE2 AA0 AA1 AA2"},
                                                {'type': "question", 'description': "VOWEL-OPENFULL", 'values': "AA0 AA1 AA2"},
                                                {'type': "question", 'description': "VOWEL-OPENNEAR", 'values': "AE0 AE1 AE2"},
                                                {'type': "question", 'description': "VOWEL-OPENMID", 'values': "EH0 EH1 EH2 ER0 ER1 ER2 AH0 AH1 AH2 AO0 AO1 AO2"},
                                                {'type': "question", 'description': "VOWEL-CLOSEFULL", 'values': "IY0 IY1 IY2 UW0 UW1 UW2"},
                                                {'type': "question", 'description': "VOWEL-CLOSENEAR", 'values': "IH0 IH1 IH2 UH0 UH1 UH2"},
                                                {'type': "question", 'description': "VOWEL-UNROUNDED", 'values': "IY0 IY1 IY2 EH0 EH1 EH2 AE0 AE1 AE2 IH0 IH1 IH2 ER0 ER1 ER2 AH0 AH1 AH2 AA0 AA1 AA2"},
                                                {'type': "question", 'description': "VOWEL-ROUNDED", 'values': "UH0 UH1 UH2 UW0 UW1 UW2 AO0 AO1 AO2"},
                                                {'type': "question", 'description': "VOWEL-FRONT", 'values': "IY0 IY1 IY2 EH0 EH1 EH2 AE0 AE1 AE2 IH0 IH1 IH2"},
                                                {'type': "question", 'description': "VOWEL-FRONTNEAR", 'values': "IH0 IH1 IH2"},
                                                {'type': "question", 'description': "VOWEL-CENTRAL", 'values': "ER0 ER1 ER2"},
                                                {'type': "question", 'description': "VOWEL-BACK", 'values': "UW0 UW1 UW2 UH0 UH1 UH2 AH0 AH1 AH2 AO0 AO1 AO2 AA0 AA1 AA2"},
                                                {'type': "question", 'description': "VOWEL-BACKNEAR", 'values': "UH0 UH1 UH2"},
                                                {'type': "question", 'description': "VOWEL-SAMPA-a", 'values': "AW0 AW1 AW2 AY0 AY1 AY2"},
                                                {'type': "question", 'description': "VOWEL-SAMPA-U", 'values': "UH0 UH1 UH2 AW0 AW1 AW2 OW0 OW1 OW2"},
                                                {'type': "question", 'description': "VOWEL-SAMPA-I", 'values': "IH0 IH1 IH2 AY0 AY1 AY2 EY0 EY1 EY2 OY0 OY1 OY2"},
                                                {'type': "question", 'description': "VOWEL-SAMPA-@", 'values': "OW0 OW1 OW2"},
                                                {'type': "question", 'description': "VOWEL-SAMPA-e", 'values': "EY0 EY1 EY2"},
                                                {'type': "question", 'description': "stress0", 'values': "IY0 AE0 UW0 AA0 EH0 AH0 AO0 IH0 EY0 AW0 AY0 ER0 UH0 OY0 OW0"},
                                                {'type': "question", 'description': "stress1", 'values': "IY1 AE1 UW1 AA1 EH1 AH1 AO1 IH1 EY1 AW1 AY1 ER1 UH1 OY1 OW1"},
                                                {'type': "question", 'description': "stress2", 'values': "IY2 AE2 UW2 AA2 EH2 AH2 AO2 IH2 EY2 AW2 AY2 ER2 UH2 OY2 OW2"},
                                                {'type': "question", 'description': "fricat", 'values': "F V TH S Z SH ZH HH"},
                                                {'type': "question", 'description': "voiced", 'values': "B D G V Z M N NG L R W JH TH"},
                                                {'type': "question", 'description': "voiceless", 'values': " T K F S SH CH HH"},
                                                {'type': "question", 'description': "vowel_UW", 'values': "UW0 UW1 UW2"},
                                                {'type': "question", 'description': "pb", 'values': "P B"},
                                                {'type': "question", 'description': "vowel_UH", 'values': "UH0 UH1 UH2"},
                                                {'type': "question", 'description': "td", 'values': "T D"},
                                                {'type': "question", 'description': "vowel_AW", 'values': "AW0 AW1 AW2"},
                                                {'type': "question", 'description': "vowel_AY", 'values': "AY0 AY1 AY2"},
                                                {'type': "question", 'description': "vowel_AA", 'values': "AA0 AA1 AA2"},
                                                {'type': "question", 'description': "vowel_AE", 'values': "AE0 AE1 AE2"},
                                                {'type': "question", 'description': "vowel_AH", 'values': "AH0 AH1 AH2"},
                                                {'type': "question", 'description': "vowel_AO", 'values': "AO0 AO1 AO2"},
                                                {'type': "question", 'description': "vowel_IY", 'values': "IY0 IY1 IY2"},
                                                {'type': "question", 'description': "sz", 'values': "S Z"},
                                                {'type': "question", 'description': "kg", 'values': "K G"},
                                                {'type': "question", 'description': "vowel_EH", 'values': "EH0 EH1 EH2"},
                                                {'type': "question", 'description': "vowel_OY", 'values': "OY0 OY1 OY2"},
                                                {'type': "question", 'description': "vowel_OW", 'values': "OW0 OW1 OW2"},
                                                {'type': "question", 'description': "vowel_IH", 'values': "IH0 IH1 IH2"},
                                                {'type': "question", 'description': "vowel_EY", 'values': "EY0 EY1 EY2"},
                                                {'type': "question", 'description': "vowel_ER", 'values': "ER0 ER1 ER2"}]},]},]
  },
]

CART_CHARACTERS = ["#", "[SILENCE]", "[UNKNOWN]", "A", "B", "C", "D", "E", "F",
                   "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                   "S", "T", "U", "V", "W", "X", "Y", "Z" ]

# adapted from CART_PHONEME_STEPS
CART_CHARACTER_STEPS = [
  {
    'name': "silence",
    'action': "cluster",
    'questions': [{'type': "question",
                   'description': "silence",
                   'key': "central",
                   'value': "[SILENCE]"}]
  },
  {
    'name': 'central',
    'action': 'partition',
    'min-obs': MIN_OBS,
    'questions': [
      {'type': "for-each-value", 'questions': [{'type': "question",
                                                'description': "central-phone",
                                                'key': "central",
                                                'values': "A B C D E F G H I J K L M N O P R S T U V W Y Z"}]},
      {'type': "question",
       'description': "noise",
       'key': "central",
       'value': "[UNKNOWN]"},
    ]
  },
  {
    'name': "hmm-state",
    'action': "partition",
    'min-obs': MIN_OBS,
    'questions': [{'type': "for-each-value", 'questions': [{'type': "question",
                                                            'description': "hmm-state",
                                                            'key': "hmm-state"}]}],
  },
  {
    'name': 'linguistics',
    'min-obs': MIN_OBS,
    'questions': [{'type': "for-each-value", 'questions': [{'type': "question",
                                                            'description': "boundary",
                                                            'key': "boundary" }]},
                  {'type': "for-each-key",
                   'keys': "history[0] central future[0]",
                   'questions': [{'type': "for-each_value",
                                  'questions': [{'type': "question", 'description': "context-phone", 'values': "# A B C D E F G H I J K L M N O P R S T U V W Y Z"},
                                                {'type': "question", 'description': "CONSONANT", 'values': "B C D F G H J K L M N N P R S T V W Y Z"},
                                                {'type': "question", 'description': "CONSONANT-OBSTRUENT", 'values': "B C D D G H J K P S T V Z"},
                                                {'type': "question", 'description': "CONSONANT-OBSTRUENT-PLOSIVE", 'values': "B D G K P T"},
                                                {'type': "question", 'description': "CONSONANT-OBSTRUENT-AFFRICATE", 'values': "C J"},
                                                {'type': "question", 'description': "CONSONANT-OBSTRUENT-FRICATIV", 'values': "D F H S T V Z"},
                                                {'type': "question", 'description': "CONSONANT-SONORANT", 'values': "L M N R W Y"},
                                                {'type': "question", 'description': "CONSONANT-SONORANT-NASAL", 'values': "M N"},
                                                {'type': "question", 'description': "CONSONANT-SONORANT-LIQUID", 'values': "R L"},
                                                {'type': "question", 'description': "CONSONANT-SONORANT-GLIDE", 'values': "W Y"},
                                                {'type': "question", 'description': "CONSONANT-APPROX", 'values': "R Y"},
                                                {'type': "question", 'description': "CONSONANT-BILABIAL", 'values': "P B M"},
                                                {'type': "question", 'description': "CONSONANT-LABIODENTAL", 'values': "F V"},
                                                {'type': "question", 'description': "CONSONANT-DENTAL", 'values': "T D"},
                                                {'type': "question", 'description': "CONSONANT-ALVEOLAR", 'values': "T D N S Z R L"},
                                                {'type': "question", 'description': "CONSONANT-POSTALVEOLAR", 'values': "S Z"},
                                                {'type': "question", 'description': "CONSONANT-VELAR", 'values': "K G N"},
                                                {'type': "question", 'description': "VOWEL", 'values': "A E I O U"},
                                                {'type': "question", 'description': "VOWEL-CHECKED", 'values': "A E I U"},
                                                {'type': "question", 'description': "VOWEL-FREE-PHTHONGS1", 'values': "A E I O"},
                                                {'type': "question", 'description': "VOWEL-FREE-PHTHONGS2", 'values': "A O U"},
                                                {'type': "question", 'description': "VOWEL-FREE-PHTHONGS3", 'values': "A E"},
                                                {'type': "question", 'description': "VOWEL-CLOSE", 'values': "I U"},
                                                {'type': "question", 'description': "VOWEL-UNROUNDED", 'values': "I E A"},
                                                {'type': "question", 'description': "VOWEL-ROUNDED", 'values': "U A"},
                                                {'type': "question", 'description': "fricat", 'values': "F V T S Z H"},
                                                {'type': "question", 'description': "voiced", 'values': "B D G V Z M N L R W J T"},
                                                {'type': "question", 'description': "voiceless", 'values': " T K F S C H"},
                                                {'type': "question", 'description': "vowel_U", 'values': "U"},
                                                {'type': "question", 'description': "pb", 'values': "P B"},
                                                {'type': "question", 'description': "vowel_A", 'values': "A"},
                                                {'type': "question", 'description': "vowel_I", 'values': "I"},
                                                {'type': "question", 'description': "kg", 'values': "K G"},
                                                {'type': "question", 'description': "vowel_E", 'values': "E"},
                                                {'type': "question", 'description': "vowel_O", 'values': "O"}]},]},]
  },
]
