__all__ = ["DecoderConfig"]

from dataclasses import dataclass


@dataclass
class DecoderConfig:
    """Recognition options for the wav2vec-U GAN forward step (``recognition.wav2vec_u.forward_step``).

    Unlike the AED ``DecoderConfig`` there is no beam search (the GAN has no autoregressive decoder):
    inference is generator -> JOIN-segment -> per-segment phoneme argmax. The pipeline turns this into
    the forward step's ``forward_init_args`` via ``dataclasses.asdict`` (see ``tune_eval.eval_model``),
    so the field names must match the ``forward_step`` kwargs.
    """

    # merge consecutive-identical phonemes left after the segmenter's pooling (fairseq inference conv.)
    collapse_repetitions: bool = True
