# Author: Christoph M. Luescher <luescher@cs.rwth-aachen.de>

import os
import itertools
import gzip
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict

from sisyphus import *
import sisyphus.toolkit as tk
import sisyphus.global_settings as gs
Path = setup_path(__package__)

# -------------------- Recipes --------------------

import recipe.i6_core.lib.lexicon as reclex
from i6_core.meta.system import CorpusObject

# -------------------- Settings --------------------

LBS_ROOT = '/u/corpora/speech/LibriSpeech/'
LBS_RES = '/work/asr3/luescher/setups-data/librispeech/resources/'

concurrent = {
    'train': 100,
    'dev': 50,
    'test': 50
}

"""
original sub corpora:
corpora:
audio:
    train-clean-100
    train-clean-360 (NA)
    train-other-500 (NA)
    train-other-960 (not original but still listed here)
    dev-clean
    dev-other
    test-clean
    test-other
audio files merged into one dir:
corpora: /work/asr3/luescher/setups-data/librispeech/resources/corpus/
audio: /u/corpora/speech/LibriSpeech/audio-merged/
    train-clean-100-merged
    train-clean-360-merged
    train-clean-460-merged
    train-other-500-merged
    train-other-960-merged
    dev-clean-merged
    dev-other-merged
    test-clean-merged
    test-other-merged
subcorpora created via sisyphus:
    train-clean-10
    train-other-10
    train-other-100
subcorpus created by andre:
corpora: /work/asr3/luescher/setups-data/librispeech/resources/corpus/
audio: /u/corpora/speech/LibriSpeech/audio-merged/
    train-andre-10
    train-andre-100
"""

train_corpus_list = [
    'train-clean-100',
    #'train-clean-360',
    #'train-other-500',
    'train-other-960',

    'train-clean-100-merged',
    'train-clean-360-merged',
    'train-clean-460-merged',
    'train-other-500-merged',
    'train-other-960-merged',

    'train-clean-10',
    'train-other-10',
    'train-other-100',

    'train-andre-10',
    'train-andre-100',
]

train_own_corpus_list = [
    'train-clean-10',
    'train-other-10',
    'train-other-100',
]

test_corpus_list = [
    'test-clean',
    'test-other',
    'test-clean-merged',
    'test-other-merged'
]

dev_corpus_list = [
    'dev-clean',
    'dev-other',
    'dev-clean-merged',
    'dev-other-merged',
    #'dev-albert-other-3000',
    'dev-complete'
]

generate_subcorpus_mapping = {
    'train': train_corpus_list,
    'test': test_corpus_list,
    'dev': dev_corpus_list
}

parentcorpus = {
    'train-clean-100': 'train',
    # 'train-clean-360': 'train',
    # 'train-other-500': 'train',
    'train-other-960': 'train',
    'train-clean-100-merged': 'train',
    'train-clean-360-merged': 'train',
    'train-clean-460-merged': 'train',
    'train-other-500-merged': 'train',
    'train-other-960-merged': 'train',
    'train-clean-10': 'train',
    'train-other-10': 'train',
    'train-other-100': 'train',
    'train-andre-10': 'train',
    'train-andre-100': 'train',
    'test-clean': 'test',
    'test-other': 'test',
    'test-clean-merged': 'test',
    'test-other-merged': 'test',
    'dev-clean': 'dev',
    'dev-other': 'dev',
    'dev-clean-merged': 'dev',
    'dev-other-merged': 'dev',
    # 'dev-albert-other-3000': 'dev',
    'dev-complete': 'dev'
}

durations = {
    'train-clean-100': 100.6,
    'train-clean-360': 363.6,
    'train-other-500': 496.7,
    'train-clean-100-merged': 100.6,
    'train-clean-360-merged': 363.6,
    'train-other-500-merged': 496.7,
    'dev-clean': 5.4,
    'dev-other': 5.3,
    'dev-clean-merged': 5.4,
    'dev-other-merged': 5.3,
    'test-clean': 5.4,
    'test-other': 5.1,
    'test-clean-merged': 5.4,
    'test-other-merged': 5.1,
}

durations['train-clean-460-merged'] = durations['train-clean-100'] + durations['train-clean-360']
durations['train-other-960'] = durations['train-clean-460-merged'] + durations['train-other-500']
durations['train-other-960-merged'] = durations['train-other-960']
durations['train-clean-10'] = durations['train-clean-100-merged'] * 10 / 100
durations['train-other-10'] = durations['train-other-960-merged'] * 10 / 960
durations['train-other-100'] = durations['train-other-960-merged'] * 100 / 960
durations['train-andre-10'] = 10
durations['train-andre-100'] = 100
# durations['dev-albert-other-3000'] = durations['dev-clean'] + durations['dev-other']
durations['dev-complete'] = durations['dev-clean'] + durations['dev-other']


class LibriSpeechCorpora:
    def __init__(self):

        self.corpora = {'train': {k: CorpusObject() for k in train_corpus_list},
                        'test': {k: CorpusObject() for k in test_corpus_list},
                        'dev': {k: CorpusObject() for k in dev_corpus_list}}

        for c in train_corpus_list:
            if c == 'train-clean-100':
                cfp = Path('{}corpora/{}.corpus.gz'.format(LBS_ROOT, c))
                adp = Path('{}LibriSpeech/{}/'.format(LBS_ROOT, c))
            elif c == 'train-other-960':
                cfp = Path('/work/speech/golik/setups/librispeech/resources/corpus/corpus.train-merged-960.corpus.gz')
                adp = Path('{}LibriSpeech/{}/'.format(LBS_ROOT, c))
            elif c == 'train-clean-100-merged':
                cfp = Path('{}corpus/corpus.train-clean-100.corpus.gz'.format(LBS_RES))
                adp = Path('{}audio-merged/'.format(LBS_ROOT))
            elif c == 'train-clean-360-merged':
                cfp = Path('{}corpus/corpus.train-clean-360.corpus.gz'.format(LBS_RES))
                adp = Path('{}audio-merged/'.format(LBS_ROOT))
            elif c == 'train-clean-460-merged':
                cfp = Path('{}corpus/corpus.train-merged-460.corpus.gz'.format(LBS_RES))
                adp = Path('{}audio-merged/'.format(LBS_ROOT))
            elif c == 'train-other-500-merged':
                cfp = Path('{}corpus/corpus.train-other-500.corpus.gz'.format(LBS_RES))
                adp = Path('{}audio-merged/'.format(LBS_ROOT))
            elif c == 'train-other-960-merged':
                cfp = Path('{}corpus/corpus.train-merged-960.corpus.gz'.format(LBS_RES))
                adp = Path('{}audio-merged/'.format(LBS_ROOT))
            elif c == 'train-clean-10':
                cfp = None
                adp = Path('{}audio-merged/'.format(LBS_ROOT))
            elif c == 'train-other-10':
                cfp = None
                adp = Path('{}audio-merged/'.format(LBS_ROOT))
            elif c == 'train-other-100':
                cfp = None
                adp = Path('{}audio-merged/'.format(LBS_ROOT))
            elif c == 'train-andre-10':
                cfp = Path('{}corpus/corpus.train-other-10.speaker.corpus.gz'.format(LBS_RES))
                adp = Path('{}LibriSpeech/'.format(LBS_ROOT))
            elif c == 'train-andre-100':
                cfp = Path('{}corpus/corpus.train-other-100.speaker.corpus.gz'.format(LBS_RES))
                adp = Path('{}LibriSpeech/'.format(LBS_ROOT))
            else:
                assert False, ("Something went wrong during corpora creation", c)

            if c not in train_own_corpus_list:
                self.corpora['train'][c].corpus_file = cfp
            self.corpora['train'][c].audio_dir = adp
            self.corpora['train'][c].audio_format = 'wav'
            self.corpora['train'][c].duration = durations[c]

        for c in test_corpus_list:
            if "merged" in c:
                cfp = Path('{}corpus/corpus.{}.corpus.gz'.format(LBS_RES, c))
                adp = Path('{}/audio-merged/{}/'.format(LBS_ROOT, c.rstrip('-merged')))
            else:
                cfp = Path('{}corpora/{}.corpus.gz'.format(LBS_ROOT, c))
                adp = Path('{}LibriSpeech/{}/'.format(LBS_ROOT, c))

            self.corpora['test'][c].corpus_file = cfp
            self.corpora['test'][c].audio_dir = adp
            self.corpora['test'][c].audio_format = 'wav'
            self.corpora['test'][c].duration = durations[c]

        for c in dev_corpus_list:
            if "merged" in c:
                cfp = Path('{}corpus/corpus.{}.corpus.gz'.format(LBS_RES, c))
            elif "complete" in c:
                cfp = None
                adp = Path('{}LibriSpeech/dev-complete/'.format(LBS_ROOT))
            else:
                cfp = Path('{}corpora/{}.corpus.gz'.format(LBS_ROOT, c))
                adp = Path('{}LibriSpeech/{}/'.format(LBS_ROOT, c))

            self.corpora['dev'][c].corpus_file = cfp
            self.corpora['dev'][c].audio_dir = adp
            self.corpora['dev'][c].audio_format = 'wav'
            self.corpora['dev'][c].duration = durations[c]

# lexica not working
# /u/corpora/speech/LibriSpeech/lexicon.new.recog.xml
# /u/corpora/speech/LibriSpeech/lexicon.new.train.xml
# /u/corpora/speech/LibriSpeech/lexicon.new.xml
# /u/corpora/speech/LibriSpeech/lexicon.recog.xml
# /u/corpora/speech/LibriSpeech/lexicon.xml

lexica = {
    'dev': Path('{}lexicon/original.lexicon.golik.xml.gz'.format(LBS_ROOT)),
    'train': Path('{}lexicon/original.lexicon.golik.xml.gz'.format(LBS_ROOT)),
    'test': Path('{}lexicon/original.lexicon.golik.xml.gz'.format(LBS_ROOT)),
    'eval': Path('{}lexicon/original.lexicon.golik.xml.gz'.format(LBS_ROOT))
}

LM_PATH = '/work/asr3/luescher/setups-data/librispeech/2018-10-24--lm/5gram-model-base/'
IRIE_PATH = '/work/asr3/irie/data/librispeech/lm/count_models/KN4/'

lms = {
    '3gram': Path('{}lm/3-gram.arpa.gz'.format(LBS_ROOT), cached=True),
    '4gram': Path('{}lm/4-gram.arpa.gz'.format(LBS_ROOT), cached=True),
    '5gram': Path(f'{LM_PATH}librispeech.5.gram.txt.lm.gz', cached=True),
    '6gram': Path(f'{LM_PATH}librispeech.6.gram.txt.lm.gz', cached=True),
    '7gram': Path(f'{LM_PATH}librispeech.7.gram.txt.lm.gz', cached=True),
    '4gram_small': Path(f'{IRIE_PATH}kn4.default_co.train-merged-960.gz', cached=True),
    'g2p': Path('{}lm/g2p-model-5.txt'.format(LBS_ROOT), cached=True)
}

stm_path  = {
    'dev-clean':    Path(LBS_ROOT + 'corpora/dev-clean.stm'),
    'test-clean':   Path(LBS_ROOT + 'corpora/test-clean.stm'),
    'dev-other':    Path(LBS_ROOT + 'corpora/dev-other.stm'),
    'test-other':   Path(LBS_ROOT + 'corpora/test-other.stm'),
}


# -------------------- CaRT --------------------

cart_phonemes = ['#', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
                 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
                 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
                 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1',
                 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0',
                 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', '[UNKNOWN]', '[SILENCE]']

cart_steps = [
    {
        'name': 'silence',
        'action': 'cluster',
        'questions': [{
            'type': 'question',
            'description': 'silence',
            'key': 'central',
            'value': 'SI'
        }]
    },
    {
        'name': 'silence',
        'action': 'cluster',
        'questions': [{
            'type': 'question',
            'description': 'silence',
            'key': 'central',
            'value': 'MUL'
        }]
    },
    {
        'name': 'central',
        'action': 'partition',
        'min-obs': 1000,
        'questions': [{
            'type': 'for-each-value',
            'questions': [{
                'type': 'question',
                'description': 'central-phone',
                'key': 'central'
            }]
        }]
    },
    {
        'name': 'hmm-state',
        'action': 'partition',
        'min-obs': 1000,
        'questions': [{
            'type': 'for-each-value',
            'questions': [{
                'type': 'question',
                'description': 'hmm-state',
                'key': 'hmm-state'
            }]
        }]
    },
    {
        'name': 'linguistics',
        'min-obs': 1000,
        'questions': [
            {
                'type': 'for-each-value',
                'questions': [{
                    'type': 'question',
                    'description': 'boundary',
                    'key': 'boundary'
                }]
            },
            {
                'type': 'for-each-key',
                'keys': 'history[0] future[0]',
                'questions': [
                    {
                        'type': 'for-each-value',
                        'questions':
                            [{'type': 'question', 'description': 'context-phone'}]
                    },
                    {'type': 'question', 'description': 'MANNER-STOP', 'values': 'P B T D K G'},
                    {'type': 'question', 'description': 'MANNER-STOP-BILABIAL', 'values': 'P B'},
                    {'type': 'question', 'description': 'MANNER-STOP-LINGUA-ALVEOLAR', 'values': 'T D'},
                    {'type': 'question', 'description': 'MANNER-STOP-LINGUA-VELAR', 'values': 'K G'},
                    {'type': 'question', 'description': 'MANNER-FRICATIVE', 'values': 'F V TH S Z SH ZH HH'},
                    {'type': 'question', 'description': 'MANNER-FRICATIVE-LABIODENTAL', 'values': 'F V'},
                    {'type': 'question', 'description': 'MANNER-FRICATIVE-LINGUA-DENTAL',  'values': 'TH'},
                    {'type': 'question', 'description': 'MANNER-FRICATIVE-LINGUA-ALVEOLAR', 'values': 'S Z'},
                    {'type': 'question', 'description': 'MANNER-FRICATIVE-LINGUA-PALATAL', 'values': 'SH ZH'},
                    {'type': 'question', 'description': 'MANNER-FRICATIVE-GLOTTAL', 'values': 'HH'},
                    {'type': 'question', 'description': 'MANNFER-AFFRICATE', 'values': 'CH JH'},
                    {'type': 'question', 'description': 'MANNER-NASAL', 'values': 'M N NG'},
                    {'type': 'question', 'description': 'MANNER-LIQUID', 'values': 'L R'},
                    {'type': 'question', 'description': 'MANNER-GLIDE', 'values': 'W JH'},
                    {'type': 'question', 'description': 'VOICELESS', 'values': 'B D G V Z M N NG L R W JH TH'},
                    {'type': 'question', 'description': 'VOWEL',
                     'values': 'AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 EH0 EH1 EH2 '
                               'ER0 ER1 ER2 EY0 EY1 EY2 IH0 IH1 IH2 IY0 IY1 IY2 OW0 OW1 OW2 OY0 OY1 OY2 UH0 UH1 UH2 '
                               'UW0 UW1 UW2'},
                    {'type': 'question', 'description': 'VOWEL-AA', 'values': 'AA0 AA1 AA2'},
                    {'type': 'question', 'description': 'VOWEL-AE', 'values': 'AE0 AE1 AE2'},
                    {'type': 'question', 'description': 'VOWEL-AH', 'values': 'AH0 AH1 AH2'},
                    {'type': 'question', 'description': 'VOWEL-AO', 'values': 'AO0 AO1 AO2'},
                    {'type': 'question', 'description': 'VOWEL-AW', 'values': 'AW0 AW1 AW2'},
                    {'type': 'question', 'description': 'VOWEL-AY', 'values': 'AY0 AY1 AY2'},
                    {'type': 'question', 'description': 'VOWEL-EH', 'values': 'EH0 EH1 EH2'},
                    {'type': 'question', 'description': 'VOWEL-EY', 'values': 'EY0 EY1 EY2'},
                    {'type': 'question', 'description': 'VOWEL-ER', 'values': 'ER0 ER1 ER2'},
                    {'type': 'question', 'description': 'VOWEL-IH', 'values': 'IH0 IH1 IH2'},
                    {'type': 'question', 'description': 'VOWEL-IY', 'values': 'IY0 IY1 IY2'},
                    {'type': 'question', 'description': 'VOWEL-OW', 'values': 'OW0 OW1 OW2'},
                    {'type': 'question', 'description': 'VOWEL-OY', 'values': 'OY0 OY1 OY2'},
                    {'type': 'question', 'description': 'VOWEL-UH', 'values': 'UH0 UH1 UH2'},
                    {'type': 'question', 'description': 'VOWEL-UW', 'values': 'UW0 UW1 UW2'},
                    {'type': 'question', 'description': 'DIPHTHONG', 'values': 'AW0 AW1 AW2 EY0 EY1 EY2 OW0 OW1 OW2'},
                    {'type': 'question', 'description': 'VOWEL-FRONT', 'values': 'AA0 AA1 AA2 AE0 AE1 AE2 AW0 AW1 AW2 '
                                                                                 'AY0 AY1 AY2 IH0 IH1 IH2 IY0 IY1 IY2 '
                                                                                 'EH0 EH1 EH2'},
                    {'type': 'question', 'description': 'VOWEL-CENTRAL', 'values': 'AH0 AH1 AH2 EH0 EH1 EH2 ER0 ER1 '
                                                                                   'ER2'},
                    {'type': 'question', 'description': 'VOWEL-BACK', 'values': 'AO0 AO1 AO2 OW0 OW1 OW2 OY0 OY1 OY2 '
                                                                                'UH0 UH1 UH2 UW0 UW1 UW2'},
                ]
            }
        ]
    }
]

# -------------------- Jobs --------------------


class CreateSubcorpus(Job):
    def __init__(self, corpus, size):
        """
        Creates a Subcorpus from the given Corpus
        :param str corpus: one of the standard Librispeech corpora plus 960h and merged
                           'train-clean-100'
                           'train-clean-360'
                           'train-other-500'
                           'train-other-960'
                           'train-clean-100-merged'
                           'train-clean-360-merged'
                           'train-other-500-merged'
                           'train-other-960-merged'
        :param int size: percentage of the corpus to use
        """
        self.set_vis_name("Create Subcorpus")
        self. corpus = corpus
        self.size = size

    def tasks(self):
        yield Task('create_files')
        yield Task('run')

    def create_files(self):
        pass

    def run(self):
        pass


class CreateXmlLexicon(Job):
    def __init__(self, phoneme_path, lexicon_path):
        self.set_vis_name("Create Lexicon File")
        self.phoneme_path = phoneme_path
        self.lexicon_path = lexicon_path
        self.lexicon_output = self.output_path('lexicon.create.xml.gz', cached=True)

        self.phonemes = []
        self.phonemes_variations = OrderedDict()
        self.lexicon = OrderedDict()

    def tasks(self):
        yield Task('run')

    def run(self):
        self.get_data()
        self.make_xml()

    def get_data(self):
        with open(self.phoneme_path) as phoneme_file:
            for phon_line in phoneme_file:
                phon_line_list = phon_line.split(' ')
                phon_key = phon_line_list[0]
                phon_val = ' '.join(phon_line[1:])
                if phon_key in self.phonemes_variations.keys():
                    if phon_val not in self.phonemes_variations[phon_key]:
                        self.phonemes_variations[phon_key].append(phon_val)
                    else:
                        self.phonemes_variations[phon_key] = [phon_val]
                for phon_symbol in phon_line_list:
                    if phon_symbol != '':
                        self.phonemes.append(phon_symbol.strip())

        with open(self.lexicon_path) as lexicon_file:
            for lex_line in lexicon_file:
                lex_line_list = lex_line.split()
                line_key = lex_line_list[0]
                line_val = ' '.join(lex_line_list[1:])
                if line_key in self.lexicon.keys():
                    if line_val not in self.lexicon[line_key]:
                        self.lexicon[line_key].append(line_val)
                else:
                    self.lexicon[line_key] = [line_val]

    def make_xml(self):
        with gzip.open(self.lexicon_output.get_path(), 'wt', encoding='utf-8') as out:
            out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            out.write('<lexicon>\n')

            out.write('  <phoneme-inventory>\n')

            for phon in self.phonemes:
                out.write('    <phoneme>\n')
                out.write('      <symbol>{}</symbol>\n'.format(phon))
                out.write('    </phoneme>\n')

            out.write('    <phoneme>\n')
            out.write('      <symbol>[SILENCE]</symbol>\n')
            out.write('      <variation>none</variation>\n')
            out.write('    </phoneme>\n')

            out.write('    <phoneme>\n')
            out.write('      <symbol>[UNKNOWN]</symbol>\n')
            out.write('      <variation>none</variation>\n')
            out.write('    </phoneme>\n')

            out.write('  </phoneme-inventory>\n')

            out.write('  <lemma special="silence">\n')
            out.write('    <orth>[SILENCE]</orth>\n')
            out.write('    <phon score="0.0">\n')
            out.write('      SI\n')
            out.write('    </phon>\n')
            out.write('    <synt/>\n')
            out.write('    <eval/>\n')
            out.write('  </lemma>\n')

            out.write('  <lemma special="unknown">\n')
            out.write('    <orth>[UNKNOWN]</orth>\n')
            out.write('    <phon score="0.0">\n')
            out.write('      MUL\n')
            out.write('    </phon>\n')
            out.write('    <synt>\n')
            out.write('      <tok>&lt;UNK&gt;</tok>\n')
            out.write('    </synt>\n')
            out.write('    <eval/>\n')
            out.write('  </lemma>\n')

            out.write('  <lemma special="sentence-begin">\n')
            out.write('    <orth>[sentence-begin]</orth>\n')
            out.write('    <synt>\n')
            out.write('      <tok>&lt;s&gt;</tok>\n')
            out.write('    </synt>\n')
            out.write('    <eval/>\n')
            out.write('  </lemma>\n')

            out.write('  <lemma special="sentence-end">\n')
            out.write('    <orth>[sentence-end]</orth>\n')
            out.write('    <synt>\n')
            out.write('      <tok>&lt;/s&gt;</tok>\n')
            out.write('    </synt>\n')
            out.write('    <eval/>\n')
            out.write('  </lemma>\n')

            for key, val in self.lexicon.items():
                out.write('  <lemma>\n')
                out.write('    <orth>{}</orth>\n'.format(key))
                for v in val:
                    out.write('    <phon score="0.0">\n')
                    out.write('      {}\n'.format(v))
                    out.write('    </phon>\n')
                out.write('  </lemma>\n')

            out.write('</lexicon>\n')


class CreateCorpus(Job):
    def __init__(self, corpus):
        self.set_vis_name("Create Corpus File")
        self.corpus = corpus
        self.corpus_file = self.output_path('{}.corpus.gz'.format(self.corpus), cached=True)

        self.corpus_root = '{}LibriSpeech/'.format(LBS_ROOT)
        self.audio_format = 'wav'
        self.speakers = {}  # dict(key: id, value: [sex, subset, min, name]
        self.transcripts = []  # [dict(name, chapter, segment, orth, path)]
        self.corpusspeakers = []  # [list(name, chapter)]

    def tasks(self):
        yield Task('run')

    def run(self):
        self.get_speakers()
        self.get_transcript()
        self.make_xml()

    def get_speakers(self):
        speakerspath = '{}LibriSpeech/SPEAKERS.TXT'.format(LBS_ROOT)
        with open(speakerspath, 'r') as speakersfile:
            for line in speakersfile:
                if line[0] != ';':
                    procline = list(map(str.strip, line.split('|')))
                    self.speakers[int(procline[0])] = [
                        procline[1],
                        procline[2],
                        float(procline[3]),
                        procline[4]
                    ]

    def get_transcript(self):
        corpuspath = os.path.join(self.corpus_root, self.corpus)
        for dirpath, dirs, files in os.walk('{}'.format(corpuspath), followlinks=True):
            for file in files:
                if file.endswith('.trans.txt'):
                    with open(os.path.join(dirpath, file), 'r') as transcription:
                        for line in transcription:
                            line_t = list(map(str.strip, line.split(' ', 1)))
                            orth = line_t[1]
                            procline = line_t[0].split('-')
                            transcript = {'name': int(procline[0]),
                                          'chapter': int(procline[1]),
                                          'segment': int(procline[2]),
                                          'orth': orth,
                                          'path': dirpath}
                            self.transcripts.append(transcript)
                            cs = [int(procline[0]), int(procline[1])]
                            if cs not in self.corpusspeakers:
                                self.corpusspeakers.append(cs)

    def make_xml(self):
        with gzip.open(self.corpus_file.get_path(), 'wt', encoding='utf-8') as out:
            out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            out.write('<corpus name="{}">\n'.format(self.corpus))

            for cs in self.corpusspeakers:
                out.write('  <speaker-description name="{}-{}" gender="{}"/>\n'.format(
                    cs[0],
                    cs[1],
                    self.speakers[cs[0]][0].lower()))

            for transcript in self.transcripts:
                out.write(
                    '  <recording name="{0}-{1}-{2:04d}" audio="{3}/{0}-{1}-{2:04d}.{4}">\n'.format(
                        transcript['name'],
                        transcript['chapter'],
                        transcript['segment'],
                        transcript['path'],
                        self.audio_format
                    ))
                out.write('    <speaker name="{}-{}"/>\n'.format(transcript['name'], transcript["chapter"]))
                out.write('    <segment start="0" end="inf" name="{0}-{1}-{2:04d}">\n'.format(
                    transcript['name'],
                    transcript['chapter'],
                    transcript['segment']
                ))
                out.write('      <orth>{}</orth>\n'.format(transcript['orth']))
                out.write('    </segment>\n')
                out.write('  </recording>\n')

            out.write('</corpus>\n')
