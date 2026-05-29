"""
Shared forward step and callback base for the LibRASR pHMM decoders under **real
RETURNN** (``recipe/returnn``).

Real RETURNN drives the ``forward`` task differently from MiniReturnn:

* ``forward_step`` receives ``extern_data`` (a :class:`TensorDict`), not the old
  ``data``/``run_ctx`` kwargs, and produces its results by *marking outputs* via
  ``Tensor.mark_as_output`` / ``rf.get_run_ctx().mark_as_output``. The run context
  is reset every step, so it cannot hold cross-step state anymore.
* Per-sequence post-processing and all persistent state live in a
  :class:`returnn.forward_iface.ForwardCallbackIface` instance, configured via the
  ``forward_callback`` config key. RETURNN splits each marked output into
  per-sequence tensors (padding removed using the dynamic dims) and hands them to
  ``ForwardCallbackIface.process_seq``.

The model forward (AM) runs batched in :func:`forward_step`; the (expensive) RASR
search runs per sequence inside the callback's ``process_seq``. To keep the RTF
diagnostics that the MiniReturnn version printed, :func:`forward_step` also marks
the per-sequence audio length (in samples) and its share of the batch AM time as
outputs, which the callback accumulates.
"""

from time import perf_counter
from typing import List, Protocol

import numpy as np

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict


class TracebackItem(Protocol):
    lemma: str
    am_score: float
    lm_score: float
    start_time: int
    end_time: int


# Output keys marked by forward_step and consumed by the callback.
_LOG_PROBS = "log_probs"
_AUDIO_SAMPLES = "audio_samples"
_AM_TIME = "am_time"


def forward_step(*, model, extern_data: TensorDict, **kwargs):
    """
    Run the acoustic model on a batch and mark its outputs.

    Marked outputs (consumed by :class:`BaseRasrForwardCallback`):

    * ``log_probs``: ``[B, T, V_am]`` AM log-probabilities, with ``T`` a dynamic dim
      so RETURNN cuts the padding per sequence.
    * ``audio_samples``: ``[B]`` raw-audio length in samples (RTF denominator).
    * ``am_time``: ``[B]`` per-sequence share of the batch AM time (RTF numerator).
    """
    import torch
    import returnn.frontend as rf
    from returnn.tensor import Dim

    audio = extern_data["raw_audio"]
    batch_dim = audio.dims[0]
    raw_audio = audio.raw_tensor  # [B, T', 1]
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor  # [B]

    am_start = perf_counter()
    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len.to(raw_audio.device),
    )
    if isinstance(logprobs, list):
        logprobs = logprobs[-1]
    am_time = perf_counter() - am_start

    num_seqs = logprobs.shape[0]

    # Dynamic time dim for the (subsampled) AM output so the callback receives each
    # sequence trimmed to its real length.
    feat_len = audio_features_len.to(device="cpu", dtype=torch.int32)
    time_dim = Dim(None, name="am_out_time")
    time_dim.dyn_size_ext = rf.convert_to_tensor(feat_len, dims=[batch_dim], dtype="int32")
    feat_dim = Dim(int(logprobs.shape[-1]), name="am_out_labels")
    log_probs_t = rf.convert_to_tensor(logprobs, dims=[batch_dim, time_dim, feat_dim], name=_LOG_PROBS)
    log_probs_t.mark_as_output(_LOG_PROBS, shape=[batch_dim, time_dim, feat_dim])

    # Per-sequence RTF bookkeeping (audio length in samples + share of the AM time).
    audio_samples = raw_audio_len.to(device="cpu", dtype=torch.int64)
    rf.convert_to_tensor(audio_samples, dims=[batch_dim], name=_AUDIO_SAMPLES).mark_as_output(
        _AUDIO_SAMPLES, shape=[batch_dim]
    )
    am_time_share = torch.full((num_seqs,), am_time / max(num_seqs, 1), dtype=torch.float32)
    rf.convert_to_tensor(am_time_share, dims=[batch_dim], name=_AM_TIME).mark_as_output(_AM_TIME, shape=[batch_dim])


class BaseRasrForwardCallback(ForwardCallbackIface):
    """
    Base callback running the LibRASR search per sequence and writing ``search_out.py``.

    Subclasses implement :meth:`build_features` to turn the AM log-probabilities of one
    sequence into the cost matrix passed to ``recognize_segment``, and may extend
    :meth:`_setup_lexicon` for additional label-inventory checks.
    """

    def __init__(self, config: dict, extra_config: dict = None):
        # Imported lazily in init() to avoid importing the decoder configs at module load
        # time (the dataclasses live in the concrete decoder module).
        self._config_kwargs = dict(config)
        self._extra_config_kwargs = dict(extra_config or {})

    # -- hooks for subclasses ------------------------------------------------

    def _make_config(self):
        """:return: the parsed DecoderConfig (subclass provides the dataclass)."""
        raise NotImplementedError

    def _make_extra_config(self):
        """:return: the parsed ExtraConfig (subclass provides the dataclass)."""
        raise NotImplementedError

    def _setup_lexicon(self, lex):
        """Validate / derive label-inventory state from the loaded lexicon."""

    def build_features(self, seq_logprobs: np.ndarray) -> np.ndarray:
        """
        :param seq_logprobs: ``[T, V_am]`` AM log-probabilities for one sequence.
        :return: ``[T, V]`` contiguous float32 cost matrix for ``recognize_segment``.
        """
        raise NotImplementedError

    # -- ForwardCallbackIface ------------------------------------------------

    def init(self, *, model):
        from i6_core.lib import lexicon
        from librasr import Configuration, SearchAlgorithm
        from returnn.util.basic import cf

        self.config = self._make_config()
        self.extra_config = self._make_extra_config()

        self.recognition_file = open("search_out.py", "wt")
        self.recognition_file.write("{\n")

        lex = lexicon.Lexicon()
        lex.load(cf(self.config.lexicon))
        self.label_inventory = list(lex.phonemes.keys())
        self.num_labels = len(self.label_inventory)
        if self.config.silence_label not in lex.phonemes:
            raise ValueError(
                f"Silence label {self.config.silence_label!r} not found in lexicon {self.config.lexicon!r}"
            )
        self._setup_lexicon(lex)

        rasr_config = Configuration()
        rasr_config.set_from_file(cf(self.config.rasr_config_file))
        self.search_algorithm = SearchAlgorithm(config=rasr_config)

        self.sample_rate = self.extra_config.sample_rate
        self.print_rtf = self.extra_config.print_rtf
        self.print_hypothesis = self.extra_config.print_hypothesis

        self.running_audio_len_s = 0.0
        self.total_am_time = 0.0
        self.total_search_time = 0.0

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        seq_logprobs = outputs[_LOG_PROBS].raw_tensor  # [T, V_am] numpy, padding already removed
        features = self.build_features(seq_logprobs)

        search_start = perf_counter()
        traceback = self.search_algorithm.recognize_segment(features=features)
        search_time = perf_counter() - search_start

        hyp = self._traceback_to_string(traceback)
        if self.print_hypothesis:
            print(hyp)
        self.recognition_file.write("%s: %s,\n" % (repr(seq_tag), repr(hyp)))

        if self.print_rtf:
            audio_len_s = float(outputs[_AUDIO_SAMPLES].raw_tensor) / self.sample_rate
            self.running_audio_len_s += audio_len_s
            self.total_am_time += float(outputs[_AM_TIME].raw_tensor)
            self.total_search_time += search_time

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()

        if self.print_rtf and self.running_audio_len_s > 0:
            print(
                "Total-AM-Time: %.2fs, AM-RTF: %.3f"
                % (self.total_am_time, self.total_am_time / self.running_audio_len_s)
            )
            print(
                "Total-Search-Time: %.2fs, Search-RTF: %.3f"
                % (self.total_search_time, self.total_search_time / self.running_audio_len_s)
            )
            total_proc_time = self.total_am_time + self.total_search_time
            print("Total-time: %.2f, Batch-RTF: %.3f" % (total_proc_time, total_proc_time / self.running_audio_len_s))

    # -- helpers -------------------------------------------------------------

    _STRIP_TOKENS: List[str] = [
        "<s>",
        "</s>",
        "<blank>",
        "[BLANK]",
        "[SILENCE]",
        "<silence>",
        "[SENTENCE-BEGIN]",
        "[SENTENCE-END]",
    ]

    @classmethod
    def _traceback_to_string(cls, traceback: List[TracebackItem]) -> str:
        out = " ".join(item.lemma for item in traceback)
        for token in cls._STRIP_TOKENS:
            out = out.replace(token, "")
        return " ".join(out.split())
