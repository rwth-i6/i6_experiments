__all__ = ["BuildPickleVocabFromListJob"]

import os
import pickle

from sisyphus import *

from i6_core.util import uopen

class ReturnnVocab():
    """
    """
    def __init__(self, vocab, vocab_size):
        """
        :param Path vocab_pickle:
        :param tk.Variable vocab_size:
        """
        self.vocab = vocab
        self.vocab_size = vocab_size


def build_character_vocab(languages=["en"], uppercase=False, alias_prefix=None):
    """
    generate forward-tacotron compatible character dictionaries

    :param languages:
    :param uppercase:
    :param str alias_prefix:
    :return:
    :rtype: ReturnnVocab
    """
    pad = '_'
    eos = '~'
    bos = '^'
    space = ' '
    characters = 'abcdefghijklmnñopqrstuvwxyzàèìòùáéíóúïü!\'"(),-.:;?'

    if 'es' in languages:
        characters += '¿¡'
    if 'de' in languages:
        characters += 'äöß'
    if 'it' in languages:
        characters += 'îû'
    if 'ca' in languages:
        characters += 'ç'
    if 'fr' in languages:
        characters += 'çæœÿêôë'

    if uppercase:
        characters = characters.upper()

    symbols = [pad, eos, bos, space] + list(characters)
    return build_vocab(symbols, alias_prefix)

def build_cmu_vocab(include_vocal_stress=True, include_optional_tokens=False, alias_prefix=None):
    """

    :param include_vocal_stress:
    :param include_optional_tokens:
    :return:
    :rtype: ReturnnVocab
    """
    phonemes = ['#', '_', '[UNKNOWN]', 'B', 'CH', 'D', 'DH',  'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG',
                'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

    phonemes_with_stress = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
                            'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
                            'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
                            'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'OW0', 'OW1', 'OW2',
                            'OY0', 'OY1', 'OY2', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2'
                            ]

    phonemes_without_stress = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH',
                               'IY', 'OW', 'OY', 'UH', 'UW']

    special_tokens_mandatory = ['.', ',', '!', '?', '~']

    special_tokens_optional = ['-', '--', '\"', '\'', '&', '(', ')', '[', ']']

    symbols = phonemes
    symbols += phonemes_with_stress if include_vocal_stress else phonemes_without_stress
    symbols += special_tokens_mandatory
    if include_optional_tokens:
        symbols += special_tokens_optional

    return build_vocab(symbols, alias_prefix)


def build_vocab(vocab_list, alias_prefix=None):
    """
    :param list[str] vocab_list:
    :param str alias_prefix:
    :return:
    :rtype: ReturnnVocab
    """
    job = BuildPickleVocabFromListJob(vocab_list)
    if alias_prefix:
        job.add_alias(os.path.join(alias_prefix, "build_vocab_job"))
    return ReturnnVocab(job.out_vocab, job.out_vocab_size)


class BuildPickleVocabFromListJob(Job):
    """
    Creates a (pickled) vocabulary based on a list input
    """

    def __init__(self, token_list):
        """
        :param list[str] token_list:
        """
        self.token_list = token_list

        self.out_vocab = self.output_path("vocab.pkl")
        self.out_vocab_size = self.output_var("vocab_length")

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        vocab = {k: v for v, k in enumerate(self.token_list)}
        pickle.dump(vocab, uopen(self.out_vocab, "wb"))

        print("Vocab Size: %i" % len(self.token_list))
        self.out_vocab_size.set(len(self.token_list))
