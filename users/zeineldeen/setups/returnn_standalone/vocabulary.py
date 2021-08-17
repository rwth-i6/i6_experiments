from sisyphus import tk

from i6_experiments.users.rossenbach.setups.returnn_standalone.bpe import BPESettings


class VocabularyDatastream:
    """
    Defines a datastream using the default `Vocabulary` class of RETURNN

    this defines a word-(unit)-based vocabulary
    """

    def __init__(self, available_for_inference, vocab, vocab_size, unk_label=None):
        """

        :param bool available_for_inference:
        :param tk.Path vocab: bpe vocab file path (pickle)
        :Param tk.Variable|int vocab_size:
        :param str unk_label: unknown label
        """
        self.available_for_inference = available_for_inference
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.unk_label = unk_label

    def as_returnn_data_opts(self, **kwargs):
        """
        :param tk.Variable|int vocab_size: number of labels
        :rtype: dict[str]
        """
        d = {
            'shape': (None,), 'dim': self.vocab_size, 'sparse': True,
            'available_for_inference': self.available_for_inference
        }
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

    def __init__(self, available_for_inference, bpe_settings, seq_postfix=0, use_unk_label=False):
        """
        :param BPESettings bpe_settings:
        :param Path vocab: vocab file path
        :param bool use_unk_label: unk_label should never be used for training
        """
        super(BpeDatastream, self).__init__(
            available_for_inference=available_for_inference,
            vocab=bpe_settings.bpe_vocab,
            vocab_size=bpe_settings.bpe_vocab_size,
            unk_label=bpe_settings.unk_label)
        self.codes = bpe_settings.bpe_codes
        assert isinstance(seq_postfix, int)
        self.seq_postfix = seq_postfix
        self.use_unk_label = use_unk_label

    def register_outputs(self, prefix):
        tk.register_output('%s.codes' % prefix, self.codes)
        tk.register_output('%s.vocab' % prefix, self.vocab)

    def as_returnn_targets_opts(self):
        opts = {
            'class': 'BytePairEncoding',
            'bpe_file': self.codes,
            'vocab_file': self.vocab,
            'unknown_label': self.unk_label if self.use_unk_label else None
        }
        if self.seq_postfix is not None:
            opts['seq_postfix'] = [self.seq_postfix]
        return opts

