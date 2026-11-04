"""Mixed ordinary-word / Granite fallback-piece traceback formatting."""

import re
import unicodedata

from .recog import HuggingFaceTracebackFormatter


class GraniteOovTracebackFormatter(HuggingFaceTracebackFormatter):
    """Join marked piece runs; preserve known lemma orths and real word spans.

    ``&&`` identifies piece output, not an unknown LM event: the prototype can
    resolve a piece run to a known word while retaining its piece traceback.
    """

    def _decode_run(self, items):
        """Decode complete prefixes, retaining timing through byte-sequence rewrites.

        Decoding individual pieces is incorrect for UTF-8 byte fallback. A newly
        added byte may replace the preceding prefix's replacement character, so
        transfer the contributing pieces of the rewritten suffix to its decoded
        replacement. The final text is always the tokenizer's full-run output.
        """
        tokens = []
        decoded = ""
        owners = []
        pending = set()
        for index, item in enumerate(items):
            token = item.lemma[:-2]
            tokens.append(token)
            updated = self._get_tokenizer().convert_tokens_to_string(tokens)
            common = 0
            while common < min(len(decoded), len(updated)) and decoded[common] == updated[common]:
                common += 1
            affected = {index} | pending
            for sources in owners[common:]:
                affected.update(sources)
            owners[common:] = [affected.copy() for _ in updated[common:]]
            # Some decoders buffer incomplete bytes instead of emitting U+FFFD.
            # Do not give contentless standalone separators ownership of a word.
            pending = affected if not updated[common:] and token != "▁" else set()
            decoded = updated
        if tokens[0].startswith("▁"):
            # The tokenizer removes the run's initial marker. Restore the boundary
            # when combining its output with preceding ordinary word lemmas.
            decoded = " " + decoded
            owners.insert(0, set())
        return decoded, owners

    def _words(self, traceback):
        surface = ""
        spans = []
        piece_run = []

        def flush_run():
            nonlocal surface
            if piece_run:
                decoded, owners = self._decode_run(piece_run)
                offset = len(surface)
                surface += decoded
                for position, sources in enumerate(owners):
                    for index in sources:
                        spans.append((offset + position, offset + position + 1, piece_run[index]))
                piece_run.clear()

        for item in traceback:
            orth = item.lemma
            if (
                not orth
                or (orth.startswith("[") and orth.endswith("]"))
                or (orth.startswith("<") and orth.endswith(">"))
            ):
                continue
            if orth.endswith("&&"):
                piece_run.append(item)
                continue
            else:
                flush_run()
                # Existing neutral punctuation is attached, not made into a word.
                punctuation = all(unicodedata.category(c).startswith("P") for c in orth)
                text = orth if punctuation else " " + orth
            start = len(surface)
            surface += text
            spans.append((start, len(surface), item))
        flush_run()
        result = []
        for match in re.finditer(r"\S+", surface):
            # Several decoded characters/bytes may refer to the same trace item.
            items = list(
                {id(item): item for start, end, item in spans if start < match.end() and end > match.start()}.values()
            )
            confidences = [getattr(item, "confidence_score", None) for item in items]
            confidences = [c for c in confidences if c is not None]
            result.append(
                (
                    match.group().upper(),
                    min(i.start_time for i in items),
                    max(i.end_time for i in items),
                    sum(confidences) / len(confidences) if confidences else 0.99,
                )
            )
        return result

    def traceback_to_transcription(self, traceback):
        return " ".join(word for word, *_ in self._words(traceback))

    def traceback_to_ctm_str(self, traceback, ms_per_frame):
        words = self._words(traceback)
        if not words:
            return "[REC_NAME] 1 0.000 0.010 <empty-sequence> 0.99\n"
        return "".join(
            f"[REC_NAME] 1 {start * ms_per_frame / 1000:.3f} "
            f"{max(end - start, 1) * ms_per_frame / 1000:.3f} {word} {confidence:.2f}\n"
            for word, start, end, confidence in words
        )
