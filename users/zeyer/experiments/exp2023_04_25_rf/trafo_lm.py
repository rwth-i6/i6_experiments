"""
Transformer LM

First: import our existing TF model

checkpoint: /work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/transfo_24_d00.4096_1024.sgd.lr1.8_heads/bk-net-model/network.023
config example: /work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.i6YlJ7HAXfGs/output/returnn.config
"""

from __future__ import annotations
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim
from returnn.frontend.decoder.transformer import TransformerDecoder


Model = TransformerDecoder


class MakeModel:
    """for import"""

    def __init__(self, vocab_dim: int, model_dim: int, *, num_layers: int):
        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.num_layers = num_layers

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        vocab_dim = Dim(self.vocab_dim, name="vocab")
        vocab_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(vocab_dim.dimension)],
        )
        model_dim = Dim(self.model_dim, name="model")

        return self.make_model(vocab_dim=vocab_dim, model_dim=model_dim, num_layers=self.num_layers)

    @classmethod
    def make_model(cls, vocab_dim: Dim, model_dim: Dim, *, num_layers: int, **extra) -> Model:
        """make"""
        return TransformerDecoder(
            encoder_dim=None,
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            **extra,
        )
