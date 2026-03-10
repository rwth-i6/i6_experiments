from i6_experiments.common.setups.returnn.datastreams.vocabulary import \
    SentencePieceDatastream as SentencePieceDatastream_OG


class SentencePieceDatastream(SentencePieceDatastream_OG):
    """
    Defines a label datastream for sentence-pieces. This does not inherit from LabelDatastream as it
    does not use am explicit vocab or unknown token.
    """

    # Overriding old method!! without EOS token addition
    def as_returnn_targets_opts(self):
        opts = {
            "class": "SentencePieces",
            "model_file": self.spm_model,
        }
        return opts