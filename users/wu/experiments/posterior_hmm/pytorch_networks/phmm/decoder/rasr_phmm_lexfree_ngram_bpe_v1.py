"""
Lexicon-free **BPE CTC** recognition with a count BPE n-gram (KenLM) LM.

BPE variant of :mod:`rasr_phmm_lexfree_ngram_v1`: the lexicon-free search, the AM ``-inf`` padding at
the ``<s>``/``</s>`` indices, and the KenLM Python label scorer (scoring BPE subword tokens via
``token_strings = self.label_inventory``) are all identical and inherited verbatim. The ONLY difference
is the output: the LibRASR traceback is a stream of BPE **subword** tokens, so :meth:`_traceback_to_string`
de-BPEs it back to WORDS. ``search_out.py`` then holds a word hypothesis and the standard ``search()`` ->
``SearchWordsToCTMJob`` -> ``ScliteJob`` pipeline scores a real **WER** -- the native metric for BPE
(subwords are losslessly invertible to words), unlike the phoneme lexfree path whose phoneme-stream
hypothesis is only scorable as PER.

The ``DecoderConfig`` is re-exported from :mod:`rasr_phmm_lexfree_ngram_v1` unchanged; the driver passes
``silence_label="[BLANK]"`` (the CTC index-0 blank lemma).
"""

# Re-exported so the serializer can import `<module>.forward_step` and `<module>.DecoderConfig`.
from .rasr_forward_base import forward_step  # noqa: F401
from .rasr_phmm_lexfree_ngram_v1 import DecoderConfig, NGRAM_SCORER_NAME  # noqa: F401
from .rasr_phmm_lexfree_ngram_v1 import ForwardCallback as _NgramForwardCallback
from .debpe import debpe_tokens


class ForwardCallback(_NgramForwardCallback):
    """Lexicon-free BPE CTC search; de-BPEs the traceback so the hypothesis is words (-> WER)."""

    @classmethod
    def _traceback_to_string(cls, traceback) -> str:
        # Token-wise strip of the special lemmata ([BLANK]/silence/<s>/</s>/...) BEFORE de-BPE -- see
        # debpe.debpe_tokens for why substring removal (the base-class behaviour) would mis-split words.
        return debpe_tokens((item.lemma for item in traceback), cls._STRIP_TOKENS)
