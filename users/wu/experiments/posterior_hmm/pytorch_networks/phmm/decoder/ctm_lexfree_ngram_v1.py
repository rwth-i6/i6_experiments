"""
Lexicon-free count-LM (KenLM) pHMM / CTC decoder that ALSO emits a phoneme-level CTM from the
LibRASR search traceback, for phoneme / silence **duration** analysis.

The search is byte-for-byte the one in :mod:`rasr_phmm_lexfree_ngram_v1` (same ``no-op`` AM +
count-LM cascade, same ``collapse_repeated_labels`` / blank handling, same acoustic prior): this
subclasses that callback and only changes what is written out. Instead of keeping just the collapsed
phoneme string, it keeps the per-lemma ``start_time`` / ``end_time`` the LibRASR traceback already
carries (in AM-output frames; see ``src/Python/Search.cc`` -> ``getCurrentBestTraceback``) and writes
a CTM line ``<seg> 1 <start_s> <dur_s> <label>`` for every traceback item -- the phonemes AND the
index-0 blank / silence lemma (``[SILENCE]`` for the pHMM, ``[BLANK]`` for CTC).

This is the "proper CTM from RASR": the segment boundaries come from RASR's own time-synchronous
search, so label self-loops already extend a label's segment and a blank / silence run is a single
contiguous item -- no hand-rolled frame-argmax collapsing. A downstream
:class:`...pipeline.PhonemeDurationStatsJob` reads the CTM and aggregates the average durations
(EOW / non-EOW phoneme; leading / trailing / between-words silence for the pHMM, or -- for CTC --
the blank folded into the preceding phoneme).

Frame shift: one AM output frame == 40 ms (10 ms log-mel hop x sub-4), passed as
``frame_shift_seconds`` so the CTM is written in real seconds.
"""

from dataclasses import dataclass
from time import perf_counter

# Re-exported so the serializer can import `<module>.forward_step` (marks `log_probs`).
from .rasr_forward_base import forward_step, _LOG_PROBS, _AUDIO_SAMPLES, _AM_TIME  # noqa: F401
from .rasr_phmm_lexfree_ngram_v1 import (
    DecoderConfig as _NgramDecoderConfig,
    ForwardCallback as _NgramForwardCallback,
)


@dataclass
class DecoderConfig(_NgramDecoderConfig):
    # Seconds of audio per AM output frame (10 ms hop x sub-4 = 0.04); converts the traceback's
    # frame-index segment boundaries into CTM seconds.
    frame_shift_seconds: float = 0.04


# Non-acoustic tokens that may appear in the traceback (sentence boundaries) but are not real
# acoustic segments; excluded from the CTM.
_SKIP_LABELS = ("<s>", "</s>", "[SENTENCE-BEGIN]", "[SENTENCE-END]")


class ForwardCallback(_NgramForwardCallback):
    """Runs the count-LM lexicon-free search and writes a phoneme / silence CTM from the traceback."""

    def _make_config(self) -> DecoderConfig:
        return DecoderConfig(**self._config_kwargs)

    def init(self, *, model):
        # Sets up the lexicon, the KenLM label scorer, the RASR search algorithm, and opens
        # search_out.py + the RTF accumulators (see BaseRasrForwardCallback.init).
        super().init(model=model)
        self._frame_shift = self.config.frame_shift_seconds
        self._ctm_file = open("durations.ctm", "wt")

    def process_seq(self, *, seq_tag: str, outputs):
        seq_logprobs = outputs[_LOG_PROBS].raw_tensor  # [T, V_am], padding removed
        features = self.build_features(seq_logprobs)

        search_start = perf_counter()
        traceback = self.search_algorithm.recognize_segment(features=features)
        search_time = perf_counter() - search_start

        # One CTM row per traceback item, keeping RASR's segment boundaries. The pHMM emits a label
        # every frame and a label self-loop (or a blank/silence run) is already merged by the search
        # into one item, so each row is one phoneme / silence occurrence with its real duration.
        for item in traceback:
            label = item.lemma
            if label in _SKIP_LABELS:
                continue
            start_s = item.start_time * self._frame_shift
            dur_s = (item.end_time - item.start_time) * self._frame_shift
            if dur_s <= 0.0:
                continue
            self._ctm_file.write("%s 1 %.3f %.3f %s\n" % (seq_tag, start_s, dur_s, label))

        # Keep a valid search_out.py as well (cheap; lets this job double as an ordinary recog).
        hyp = self._traceback_to_string(traceback)
        if self.print_hypothesis:
            print(hyp)
        self.recognition_file.write("%s: %s,\n" % (repr(seq_tag), repr(hyp)))

        if self.print_rtf:
            self.running_audio_len_s += float(outputs[_AUDIO_SAMPLES].raw_tensor) / self.sample_rate
            self.total_am_time += float(outputs[_AM_TIME].raw_tensor)
            self.total_search_time += search_time

    def finish(self):
        super().finish()  # writes "}" + closes search_out.py + prints the RTF summary
        self._ctm_file.close()
