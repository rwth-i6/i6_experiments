from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
from sisyphus import tk

from i6_experiments.users.sheremeta.datasets.vocabulary import SentencePieceDatastreamV2

_MODEL = tk.Path("/fake/spm.model")


def test_targets_opts_without_an_alpha_take_the_best_segmentation_and_no_eos():
    stream = SentencePieceDatastreamV2(available_for_inference=False, spm_model=_MODEL, vocab_size=10)

    assert stream.as_returnn_targets_opts() == {"class": "SentencePieces", "model_file": _MODEL, "add_eos": False}


def test_targets_opts_with_an_alpha_sample_from_all_segmentations():
    stream = SentencePieceDatastreamV2(available_for_inference=False, spm_model=_MODEL, vocab_size=10)
    # the recipe sets the alpha after construction
    stream.sampling_alpha = 0.9

    assert stream.as_returnn_targets_opts() == {
        "class": "SentencePieces",
        "model_file": _MODEL,
        "add_eos": False,
        "enable_sampling": True,
        "alpha": 0.9,
        "nbest_size": -1,
    }


def test_sbpe_opts_read_the_pieces_of_the_same_model():
    stream = SentencePieceDatastreamV2(available_for_inference=False, spm_model=_MODEL, vocab_size=10)

    assert stream.as_returnn_sbpe_targets_opts(0.1) == {
        "class": "SamplingBytePairEncoding",
        "vocab_file": ExtractSentencePieceVocabJob(_MODEL).out_vocab,
        "breadth_prob": 0.1,
        "word_prefix_symbol": "▁",
        "unknown_label": "<unk>",
    }
