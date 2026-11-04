"""Recover original output orthography from Qwen's alternative lexical LM routes."""

from types import SimpleNamespace

from .recog import BpeTracebackFormatter


def original_qwen_orth(name):
    if not name.startswith("qwen-v1:"):
        return name
    fields = name.split(":", 3)
    if len(fields) != 4 or not fields[1].isdigit() or not fields[2].isdigit() or not fields[3]:
        raise ValueError(f"Malformed Qwen variant lemma name: {name!r}")
    return fields[3]


class QwenTextVariantTracebackFormatter(BpeTracebackFormatter):
    @staticmethod
    def _original_words(traceback):
        # Do not mutate native objects: raw tracebacks must retain the chosen
        # case/spacing variant for diagnostics. Preserve cumulative scores/timing.
        return [
            SimpleNamespace(
                lemma=original_qwen_orth(item.lemma),
                am_score=item.am_score,
                lm_score=item.lm_score,
                start_time=item.start_time,
                end_time=item.end_time,
                confidence_score=getattr(item, "confidence_score", None),
            )
            for item in traceback
        ]

    def traceback_to_transcription(self, traceback):
        return super().traceback_to_transcription(self._original_words(traceback))

    def traceback_to_ctm_str(self, traceback, ms_per_frame):
        return super().traceback_to_ctm_str(self._original_words(traceback), ms_per_frame)
