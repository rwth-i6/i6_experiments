__all__ = ['concurrent', 'corpora', 'lexica', 'stm_path', 'glm_path']

from sisyphus import *
Path = setup_path(__package__)

from i6_core.meta.system import CorpusObject

SWB_ROOT = '/u/corpora/speech/switchboard-1/'
HUB500_ROOT = '/u/corpora/speech/hub-5-00/'
HUB501_ROOT = '/u/corpora/speech/hub-5-01/'
HUB5E00_ROOT = '/u/corpora/speech/hub5e_00/'

concurrent = { 'train' : { 'full'    : 200, '100k'    : 50 },
               'eval'  : { 'hub5-00' : 20,  'hub5-01' : 20 } }

corpora = { 'train': { k: CorpusObject() for k in ['full', '100k', '2000h'] },
            'dev':   { k: CorpusObject() for k in ['hub5-00', 'hub5-01', 'hub5e-00', 'dev_zoltan']},
            'eval':  { k: CorpusObject() for k in ['hub5-00', 'hub5-01', 'hub5e-00', 'dev_zoltan']}}

corpora['train']['full'].corpus_file  = Path(SWB_ROOT + 'xml/swb1-all/swb1-all.corpus.gz')
corpora['train']['full'].audio_dir    = Path(SWB_ROOT + 'audio/')
corpora['train']['full'].audio_format = 'wav'
corpora['train']['full'].duration     = 311.78

corpora['train']['100k'].corpus_file  = Path(SWB_ROOT + 'xml/swb1-100k/swb1-100k.corpus.gz')
corpora['train']['100k'].audio_dir    = Path(SWB_ROOT + 'audio/')
corpora['train']['100k'].audio_format = 'wav'
corpora['train']['100k'].duration     = 123.60

corpora['train']['2000h'].corpus_file  = Path(SWB_ROOT + 'xml/swb-2000/train.2000.xml.gz')
corpora['train']['2000h'].audio_dir    = Path('/work/speechcorpora/fisher/en/audio/')
corpora['train']['2000h'].audio_format = 'wav'
corpora['train']['2000h'].duration     = 2033.0

for c in ['dev', 'eval']:
    corpora[c]['hub5-00'].corpus_file   = Path(HUB500_ROOT + 'xml/hub-5-00-all.corpus.gz')
    corpora[c]['hub5-00'].audio_dir     = Path(HUB500_ROOT + 'audio/')
    corpora[c]['hub5-00'].audio_format  = 'wav'
    corpora[c]['hub5-00'].duration      = 3.65

    corpora[c]['hub5e-00'].corpus_file   = Path(HUB5E00_ROOT + 'xml/hub5e_00.corpus.gz')
    corpora[c]['hub5e-00'].audio_dir     = Path(HUB5E00_ROOT + 'english/')
    corpora[c]['hub5e-00'].audio_format  = 'wav'
    corpora[c]['hub5e-00'].duration      = 3.65

    corpora[c]['hub5-01'].corpus_file   = Path('/u/tuske/work/ASR/switchboard/corpus/xml/hub5e_01.corpus.gz')
    corpora[c]['hub5-01'].audio_dir     = Path('/u/corpora/speech/hub-5-01/audio/')
    corpora[c]['hub5-01'].audio_format  = 'wav' # nist
    corpora[c]['hub5-01'].duration      = 6.2

    corpora[c]['dev_zoltan'].corpus_file   = Path('/u/tuske/work/ASR/switchboard/corpus/xml/dev.corpus.gz')
    corpora[c]['dev_zoltan'].audio_dir     = Path(HUB500_ROOT + 'audio/')
    corpora[c]['dev_zoltan'].audio_format  = 'wav'
    corpora[c]['dev_zoltan'].duration      = 3.65


lexica = { 'train': Path(SWB_ROOT + 'lexicon/train.lex.v1_0_3.ci.gz'),
           'train-2000': Path(SWB_ROOT + 'xml/swb-2000/train.merge.g2p.lex.gz'),
           'eval':  Path(SWB_ROOT + 'lexicon/train.lex.v1_0_4.ci.gz') }

lms    = { 'train': Path(SWB_ROOT + 'lm/switchboardFisher.lm.gz'),
           'eval':  Path("/home/tuske/work/ASR/switchboard/corpus/lm/data/mylm/swb.fsh.4gr.voc30k.LM.gz") } #TODO Check if correct lms are used

stm_path = {'hub5-00':    Path(HUB500_ROOT + 'scoring/hub-5-00-all.stm'),
            'hub5e-00':   Path(HUB5E00_ROOT + 'xml/hub5e_00.stm'),
            'hub5-01':    Path(HUB501_ROOT +  'raw/hub5e_01/data/transcr/hub5e01.english.20010402.stm'),
            'dev_zoltan': Path('/u/tuske/bin/switchboard/hub5e_00.2.stm')
            }
glm_path = {'hub5-00':    Path(HUB500_ROOT + 'scoring/en20000405_hub5.glm'),
            'hub5e-00':   Path(HUB5E00_ROOT + 'xml/glm'),
            'hub5-01':    Path(HUB500_ROOT + 'scoring/en20000405_hub5.glm'),
            'dev_zoltan': Path(HUB500_ROOT + 'scoring/en20000405_hub5.glm')
            }

cart_phonemes = ['#', '[LAUGHTER]', '[NOISE]', '[SILENCE]', '[VOCALIZEDNOISE]',
                 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'el', 'en', 'er', 'ey', 'f', 'g',
                 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw',
                 'v', 'w', 'y', 'z', 'zh']

cart_steps = [{'name': 'silence',
               'action': 'cluster',
               'questions': [{'type': 'question', 'description': 'silence', 'key': 'central', 'value': '[SILENCE]'}]
               },
              {'name': 'noise',
               'action': 'cluster',
               'questions': [{'type': 'question', 'description': 'noise_%s' % phn, 'key': 'central', 'value': phn}
                             for phn in ('[LAUGHTER]', '[NOISE]', '[VOCALIZEDNOISE]')]
               },
              {'name': 'central',
               'action': 'partition',
               'min-obs': 1000,
               'questions': [{'type': 'for-each-value', 'questions': [{'type': 'question',
                                                                       'description': 'central-phone',
                                                                       'key': 'central'}]
                              }]
               },
              {'name': 'hmm-state',
               'action': 'partition',
               'min-obs': 1000,
               'questions': [{'type': 'for-each-value', 'questions': [{'type': 'question',
                                                                       'description': 'hmm-state',
                                                                       'key': 'hmm-state'}]
                              }]
               },
              {'name': 'linguistics',
               'min-obs': 1000,
               'questions': [{'type': 'for-each-value', 'questions': [{'type': 'question',
                                                                       'description': 'boundary',
                                                                       'key': 'boundary'}]
                              },
                             {'type': 'for-each-key',
                              'keys': 'history[0] central future[0]',
                              'questions': [{'type': 'for-each-value',
                                             'questions': [{'type': 'question', 'description': 'context-phone'}]},
                                                           {'type': 'question', 'description': 'CONSONANT', 'values': 'b ch d dh f g hh jh k l el m n en ng p r s sh t th v w y z zh'},
                                                           {'type': 'question', 'description': 'CONSONANT-OBSTRUENT', 'values': 'b ch d dh f g hh jh k p s sh t th v z zh'},
                                                           {'type': 'question', 'description': 'CONSONANT-OBSTRUENT-PLOSIVE', 'values': 'b d g k p t'},
                                                           {'type': 'question', 'description': 'CONSONANT-OBSTRUENT-AFFRICATE', 'values': 'ch jh'},
                                                           {'type': 'question', 'description': 'CONSONANT-OBSTRUENT-FRICATIVE', 'values': 'dh f hh s sh th v z zh'},
                                                           {'type': 'question', 'description': 'CONSONANT-SONORANT', 'values': 'l el m n en ng r w y '},
                                                           {'type': 'question', 'description': 'CONSONANT-SONORANT-NASAL', 'values': 'm n en ng'},
                                                           {'type': 'question', 'description': 'CONSONANT-SONORANT-LIQUID', 'values': 'r l el'},
                                                           {'type': 'question', 'description': 'CONSONANT-SONORANT-GLIDE', 'values': 'w y'},
                                                           {'type': 'question', 'description': 'CONSONANT-APPROX', 'values': 'r y'},
                                                           {'type': 'question', 'description': 'CONSONANT-BILABIAL', 'values': 'p b m'},
                                                           {'type': 'question', 'description': 'CONSONANT-LABIODENTAL', 'values': 'f v'},
                                                           {'type': 'question', 'description': 'CONSONANT-DENTAL', 'values': 'th dh'},
                                                           {'type': 'question', 'description': 'CONSONANT-ALVEOLAR', 'values': 't d n en s z r l el'},
                                                           {'type': 'question', 'description': 'CONSONANT-POSTALVEOLAR', 'values': 'sh zh'},
                                                           {'type': 'question', 'description': 'CONSONANT-VELAR', 'values': 'k g ng'},
                                                           {'type': 'question', 'description': 'VOWEL', 'values': 'aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw'},
                                                           {'type': 'question', 'description': 'VOWEL-CHECKED', 'values': 'ae ah eh ih uh '},
                                                           {'type': 'question', 'description': 'VOWEL-SHORTCENTRAL', 'values': 'ax '},
                                                           {'type': 'question', 'description': 'VOWEL-FREE', 'values': 'aa ao aw ay er ey iy ow oy uw'},
                                                           {'type': 'question', 'description': 'VOWEL-FREE-PHTHONGS1', 'values': 'ay ey iy oy'},
                                                           {'type': 'question', 'description': 'VOWEL-FREE-PHTHONGS2', 'values': 'aw ow uw '},
                                                           {'type': 'question', 'description': 'VOWEL-FREE-PHTHONGS3', 'values': 'aa ao er'},
                                                           {'type': 'question', 'description': 'VOWEL-CLOSE', 'values': 'iy uw ih uh'},
                                                           {'type': 'question', 'description': 'VOWEL-OPEN', 'values': 'eh er ah ao ae aa'},
                                                           {'type': 'question', 'description': 'VOWEL-OPENFULL', 'values': 'aa'},
                                                           {'type': 'question', 'description': 'VOWEL-OPENNEAR', 'values': 'ae'},
                                                           {'type': 'question', 'description': 'VOWEL-OPENMID', 'values': 'eh er ah ao'},
                                                           {'type': 'question', 'description': 'VOWEL-CLOSEFULL', 'values': 'iy uw'},
                                                           {'type': 'question', 'description': 'VOWEL-CLOSENEAR', 'values': 'ih uh'},
                                                           {'type': 'question', 'description': 'VOWEL-UNROUNDED', 'values': 'iy eh ae ih er ah aa'},
                                                           {'type': 'question', 'description': 'VOWEL-ROUNDED', 'values': 'uh uw ao'},
                                                           {'type': 'question', 'description': 'VOWEL-FRONT', 'values': 'iy eh ae ih'},
                                                           {'type': 'question', 'description': 'VOWEL-FRONTNEAR', 'values': 'ih'},
                                                           {'type': 'question', 'description': 'VOWEL-CENTRAL', 'values': 'ax er'},
                                                           {'type': 'question', 'description': 'VOWEL-BACK', 'values': 'uw uh ah ao aa'},
                                                           {'type': 'question', 'description': 'VOWEL-BACKNEAR', 'values': 'uh'},
                                                           {'type': 'question', 'description': 'VOWEL-SAMPA-a', 'values': 'aw ay'},
                                                           {'type': 'question', 'description': 'VOWEL-SAMPA-U', 'values': 'uh aw ow'},
                                                           {'type': 'question', 'description': 'VOWEL-SAMPA-I', 'values': 'ih ay ey oy'},
                                                           {'type': 'question', 'description': 'VOWEL-SAMPA-@', 'values': 'ax ow'},
                                                           {'type': 'question', 'description': 'VOWEL-SAMPA-e', 'values': 'ey '},
                                            ]
                              }
                             ]
               }
              ]

