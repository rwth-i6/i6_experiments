"""
Defines the data inputs for any RASR based LibriSpeech task
"""
from sisyphus import tk
from dataclasses import dataclass
from typing import Dict

from i6_experiments.common.datasets.switchboard.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.switchboard.corpus_eval import (
    get_hub5e00,
    get_hub5e01,
    get_rt03s,
    get_hub5e00_corpus_object,
    get_hub5e01_corpus_object,
    get_rt03s_corpus_object
)
from i6_experiments.common.datasets.switchboard.corpus_train import get_train_corpus_object_ldc, \
    get_train_corpus_object_i6_legacy

from i6_experiments.common.setups.rasr import RasrDataInput


@dataclass()
class CorpusData:
    """
    Helper class to define all RasrDataInputs to be passed to the `System` class
    """

    train_data: Dict[str, RasrDataInput]
    dev_data: Dict[str, RasrDataInput]
    test_data: Dict[str, RasrDataInput]


def get_corpus_data_inputs(use_legacy=True, use_legacy_lexicon=False, normalize_pronunciation=False) -> CorpusData:
    """
    Create the corpus data for any LibriSpeech RASR setup

    :return: (train_data, dev_data, test_data)
    """

    # Dictionary containing all LibriSpeech CorpusObject entries
    if use_legacy:
        corpus_object_dict = get_train_corpus_object_i6_legacy()
    else:
        corpus_object_dict = get_train_corpus_object_ldc()

    # Definition of the official 4-gram LM to be used as default LM
    # lm = {
    #     "filename": get_arpa_lm_dict()["4gram"],
    #     "type": "ARPA",
    #     "scale": 10,
    # }

    temporary_lm = {
        "filename": tk.Path("/home/tuske/work/ASR/switchboard/corpus/lm/data/mylm/swb.fsh.4gr.voc30k.LM.gz"),
        "type": "ARPA",
        "scale": 10,
    }


    # This is the standard LibriSpeech lexicon
    if use_legacy_lexicon:
        lexicon = {
            "filename": tk.Path("/u/corpora/speech/switchboard-1/lexicon/train.lex.v1_0_3.ci.gz"),
            "normalize_pronunciation": normalize_pronunciation,
        }
    else:
        lexicon = {
            "filename": get_bliss_lexicon(),
            "normalize_pronunciation": normalize_pronunciation,
        }

    # Here we define all corpora that are used.
    # The training corpus is dynamic based on which subset we want to use,
    # but dev and test are fixed.
    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs["switchboard"] = RasrDataInput(
        corpus_object=corpus_object_dict,
        concurrent=60,
        lexicon=lexicon,
        lm=None,
    )


    hub5e00 = get_hub5e00()
    hub5e00_corpus_object = get_hub5e00_corpus_object()

    hub5e01 = get_hub5e01()
    hub5e01_corpus_object = get_hub5e01_corpus_object()

    rt03s = get_rt03s()
    rt03s_corpus_object = get_rt03s_corpus_object()

    dev_data_inputs["hub5e00"] = RasrDataInput(
        corpus_object=hub5e00_corpus_object,
        lexicon=lexicon,
        lm=temporary_lm,
        concurrent=10,
        stm=hub5e00.stm,
        glm=hub5e00.glm,
    )

    test_data_inputs["hub5e01"] = RasrDataInput(
        corpus_object=hub5e01_corpus_object,
        lexicon=lexicon,
        lm=temporary_lm,
        concurrent=10,
        stm=hub5e01.stm,
        glm=hub5e01.glm,
    )

    test_data_inputs["rt03s"] = RasrDataInput(
        corpus_object=rt03s_corpus_object,
        lexicon=lexicon,
        lm=temporary_lm,
        concurrent=10,
        stm=rt03s.stm,
        glm=rt03s.glm,
    )
    # for dev_key in ["dev-clean", "dev-other"]:
    #     dev_data_inputs[dev_key] = RasrDataInput(
    #         corpus_object=corpus_object_dict[dev_key],
    #         concurrent=constants.concurrent[dev_key],
    #         lexicon=lexicon,
    #         lm=lm,
    #     )

    # for test_key in ["test-clean", "test-other"]:
    #     test_data_inputs[test_key] = RasrDataInput(
    #         corpus_object=corpus_object_dict[test_key],
    #         concurrent=constants.concurrent[test_key],
    #         lexicon=lexicon,
    #         lm={},
    #     )

    return CorpusData(
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )


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
