from sisyphus import tk

from i6_experiments.users.rossenbach.setups.returnn_standalone.bpe import BPESettings


class VocabularyDatastream:
    """
    Defines a datastream using the default `Vocabulary` class of RETURNN

    this defines a word-(unit)-based vocabulary
    """

    def __init__(self, vocab, vocab_size, unk_label=None):
        """
        :param tk.Path vocab: bpe vocab file path (pickle)
        :Param tk.Variable|int vocab_size:
        :param str unk_label: unknown label
        """
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.unk_label = unk_label

    def as_returnn_data_opts(self, vocab_size, **kwargs):
        """
        :param tk.Variable|int vocab_size: number of labels
        :rtype: dict[str]
        """
        d = {'shape': (None,), 'dim': vocab_size, 'sparse': True}
        d.update(kwargs)
        return d

    def as_returnn_targets_opts(self, **kwargs):
        """
        :rtype: dict[str]
        """
        return {
            "vocab_file": self.vocab,
            "unknown_label": self.unk_label,
            **kwargs
        }


class BpeDatastream(VocabularyDatastream):
    """
    This defines a datastream using the BytePairEncoding(Vocabulary) class of RETURNN
    """

    def __init__(self, bpe_settings, seq_postfix=0, unk_label=None):
        """
        :param BPESettings bpe_settings:
        :param Path vocab: vocab file path
        :param str unk_label: unknown label
        """
        super(VocabularyDatastream, self).__init__(
            vocab=bpe_settings.bpe_vocab,
            vocab_size=bpe_settings.bpe_vocab_size,
            unk_label=unk_label)
        self.codes = bpe_settings.bpe_codes
        assert isinstance(seq_postfix, int)
        self.seq_postfix = seq_postfix

    def register_outputs(self, prefix):
        tk.register_output('%s.codes' % prefix, self.codes)
        tk.register_output('%s.vocab' % prefix, self.vocab)

    def as_returnn_targets_opts(self):
        opts = {
            'class': 'BytePairEncoding',
            'bpe_file': self.codes,
            'vocab_file': self.vocab,
            'unknown_label': self.unk_label
        }
        if self.seq_postfix is not None:
            opts['seq_postfix'] = [self.seq_postfix]
        return opts

