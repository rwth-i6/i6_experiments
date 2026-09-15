"""Round-trip WER of the synthetic corpus: what we ASKED the TTS to say vs what Whisper HEARD.

Backlog E4. Both sides already exist on disk, so this costs no annotation and no GPU:

* **Reference** -- the text *we authored* and handed to Chatterbox. It survives in the TTS output's
  ``turns`` column (``chatterbox_inference.dialogue_features``), one entry per turn with a
  ``speaker``.
* **Hypothesis** -- the Whisper transcript the annotate stage already produced. ``MoshiAnnotate``
  runs Whisper on ``stereo[0]``, the **assistant** channel, and stores it as ``alignments``
  (word-level, every entry labelled ``SPEAKER_MAIN``).

A high-WER row means the synthesiser said something other than what we asked for, and we are
currently training on it silently -- ground truth is free here precisely because we wrote the text.

**This is a STATISTIC, not a filter** (the `DialogueCorpusStats` shape). It is attached to a corpus
and read before training. Filtering on WER would change what the corpus contains and therefore
re-hash every arm that trains on it, which is a separate and much more expensive decision.

⚠ **There is a WER floor that is not TTS error, and it must not be read as one.** Whisper
normalises what it hears into its own orthography: digits vs words ("1998" vs "nineteen
ninety-eight"), symbols vs words ("%" vs "percent"), contractions, and hyphenation. Those land as
substitutions no matter how perfect the audio is. So the number to act on is the *tail* -- rows far
above the corpus median -- not the median itself. The report prints the worst rows verbatim so the
distinction is visible rather than assumed.

Why not ``jobs/fastwer.py::FastButInaccurateWer``: it computes ONE corpus-level WER from two
line-aligned text files. E4 needs a per-row number (to build a distribution and to name the bad
rows) and a join by ``id`` rather than by line position. The edit-distance core is the same
``Levenshtein.distance`` over word lists.
"""

from __future__ import annotations

import json
import re

from sisyphus import Job, Task, tk

#: Role whose speech the annotate stage transcribes, and whose turns are therefore the reference.
#: `MoshiAnnotate.process_row` annotates `stereo[0]` == the assistant channel; the user channel is
#: stored but never transcribed. Comparing against user turns would be measuring nothing.
ASSISTANT_SPEAKER = "assistant"

#: Buckets for the WER distribution, as (lower_inclusive, label). Open-ended at the top because the
#: interesting rows are the ones where the TTS lost the plot entirely, and those are unbounded.
_BUCKETS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0)

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_for_wer(text: str) -> list[str]:
    """Lowercase, drop punctuation, collapse whitespace -> word list.

    Deliberately minimal and symmetric: the same function runs on both sides, so it can only remove
    *shared* sources of disagreement (case, punctuation, spacing). It does NOT map numbers to words
    or expand contractions -- doing that would quietly absorb real TTS errors into the
    normalisation, which is the opposite of what this job is for.
    """
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip().split()


def row_wer(ref_words: list[str], hyp_words: list[str]) -> float | None:
    """Word-level edit distance / reference length. ``None`` when the reference is empty.

    An empty reference is not WER 0 and not WER 1 -- it is unmeasurable, and averaging a made-up
    value over those rows is how a corpus statistic becomes a fiction. They are counted separately.
    """
    if not ref_words:
        return None
    import Levenshtein

    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)


class SyntheticSpeechWer(Job):
    """Per-row TTS round-trip WER for one corpus, plus a distribution and the worst offenders."""

    def __init__(
        self,
        *,
        tts_hf: tk.Path,
        annotated_hf: tk.Path,
        sample: int = 0,
        seed: int = 1234,
        worst_n: int = 15,
    ):
        """``sample``: rows to score (0 = all). Sampling is seeded, so numbers are reproducible."""
        self.tts_hf = tts_hf
        self.annotated_hf = annotated_hf
        self.sample = sample
        self.seed = seed
        self.worst_n = worst_n
        self.out_json = self.output_path("wer.json")
        self.out_report = self.output_path("report.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _text_only(path: str, keep: tuple[str, ...]):
        """Load an arrow dataset with everything but ``keep`` dropped.

        ⚠ Load-bearing for memory, not tidiness. These datasets carry HF ``Audio()`` columns, and
        indexing a row DECODES every column in it -- so touching rows on the full schema pulls the
        whole corpus's audio through RAM one row at a time (the failure mode that OOM-killed four
        evals on 2026-09-11). Dropping the columns first is a schema operation on the memory map and
        costs nothing.
        """
        from datasets import load_from_disk

        ds = load_from_disk(path)
        drop = [c for c in ds.column_names if c not in keep]
        return ds.remove_columns(drop) if drop else ds

    def run(self):
        import random

        tts = self._text_only(self.tts_hf.get(), ("id", "turns"))
        ann = self._text_only(self.annotated_hf.get(), ("id", "alignments"))

        # Join by id, never by row order: the annotate stage DROPS failed rows (missing audio or a
        # Whisper error), so the two datasets are not positionally aligned and a zip would silently
        # compare row i's text against row j's audio -- producing a plausible, entirely meaningless
        # WER. This is the class of bug the standing "never address clips by position" rule exists
        # for.
        hyp_by_id: dict[str, str] = {}
        for i, rid in enumerate(ann["id"]):
            words = [a["text"] for a in ann[i]["alignments"]]
            hyp_by_id[str(rid)] = " ".join(words)

        ids = [str(x) for x in tts["id"]]
        idx = list(range(len(ids)))
        # Random, seeded, and SAY SO. Every corpus here is a shard merge in sorted key order, so a
        # head slice reads one template, not the corpus -- which has already produced one confident
        # and completely wrong readout (see CLAUDE.md).
        if self.sample and self.sample < len(idx):
            idx = random.Random(self.seed).sample(idx, self.sample)
            provenance = f"sampled {len(idx)} of {len(ids)} rows at random (seed {self.seed})"
        else:
            provenance = f"all {len(ids)} rows"
        print(f"[wer] {provenance}", flush=True)

        rows, empty_ref, missing_hyp = [], 0, 0
        for n, i in enumerate(idx):
            rid = ids[i]
            ref_words: list[str] = []
            for turn in tts[i]["turns"]:
                if turn.get("speaker") == ASSISTANT_SPEAKER:
                    ref_words += normalize_for_wer(turn.get("text", ""))
            if rid not in hyp_by_id:
                # The annotate stage dropped this row; it is not a WER of 1, it simply has no
                # hypothesis. Counted and excluded rather than scored.
                missing_hyp += 1
                continue
            hyp_words = normalize_for_wer(hyp_by_id[rid])
            w = row_wer(ref_words, hyp_words)
            if w is None:
                empty_ref += 1
                continue
            rows.append(
                {
                    "id": rid,
                    "wer": w,
                    "ref_words": len(ref_words),
                    "hyp_words": len(hyp_words),
                    "ref": " ".join(ref_words),
                    "hyp": " ".join(hyp_words),
                }
            )
            if (n + 1) % 2000 == 0:
                print(f"[wer] {n + 1}/{len(idx)} scored", flush=True)

        assert rows, (
            f"scored 0 rows from {self.tts_hf.get()!r} -- {missing_hyp} had no annotate output and "
            f"{empty_ref} had no assistant text. Refusing to write an empty statistic."
        )

        rows.sort(key=lambda r: r["wer"], reverse=True)
        wers = sorted(r["wer"] for r in rows)

        def pct(p: float) -> float:
            return round(wers[min(len(wers) - 1, int(p * len(wers)))], 4)

        hist = {}
        for lo, hi in zip(_BUCKETS, (*_BUCKETS[1:], None)):
            label = f">={lo:.2f}" if hi is None else f"{lo:.2f}-{hi:.2f}"
            hist[label] = sum(1 for w in wers if w >= lo and (hi is None or w < hi))

        # Total-word WER as well as the mean of per-row WERs: they answer different questions, and
        # quoting one as the other is a classic mis-read. The corpus number weights long rows more;
        # the mean weights every row equally and is what the distribution above describes.
        total_ref = sum(r["ref_words"] for r in rows)
        corpus_wer = sum(r["wer"] * r["ref_words"] for r in rows) / total_ref if total_ref else None

        summary = {
            "provenance": provenance,
            "scored": len(rows),
            "skipped_no_annotate_output": missing_hyp,
            "skipped_empty_reference": empty_ref,
            "corpus_wer_word_weighted": round(corpus_wer, 4) if corpus_wer is not None else None,
            "mean_row_wer": round(sum(wers) / len(wers), 4),
            "median_row_wer": pct(0.50),
            "p90_row_wer": pct(0.90),
            "p99_row_wer": pct(0.99),
            "rows_above_0p35": sum(1 for w in wers if w >= 0.35),
            "histogram": hist,
            "worst": rows[: self.worst_n],
        }
        with open(self.out_json.get_path(), "w") as f:
            json.dump(summary, f, indent=2)

        lines = [
            "TTS round-trip WER -- authored text vs Whisper transcript of the synthesised audio",
            f"  {provenance}",
            f"  scored {len(rows)}   skipped: {missing_hyp} no-annotate-output, {empty_ref} empty-reference",
            "",
            f"  corpus WER (word-weighted) : {summary['corpus_wer_word_weighted']}",
            f"  mean row WER               : {summary['mean_row_wer']}",
            f"  median / p90 / p99         : {summary['median_row_wer']} / {summary['p90_row_wer']} / {summary['p99_row_wer']}",
            f"  rows at WER >= 0.35        : {summary['rows_above_0p35']}",
            "",
            "  distribution:",
        ]
        for label, n in hist.items():
            bar = "#" * min(60, round(60 * n / max(len(wers), 1)))
            lines.append(f"    {label:>12}  {n:>6}  {bar}")
        lines += [
            "",
            "  NOTE: a floor of disagreement is Whisper's orthography (digits vs words, symbols,",
            "  hyphenation), not TTS error. Read the TAIL, not the median.",
            "",
            f"  worst {min(self.worst_n, len(rows))} rows (verbatim, normalised for comparison):",
        ]
        for r in rows[: self.worst_n]:
            lines += [
                f"    [{r['id']}] wer={r['wer']:.3f}  ref_words={r['ref_words']} hyp_words={r['hyp_words']}",
                f"        asked: {r['ref'][:300]}",
                f"        heard: {r['hyp'][:300]}",
            ]
        with open(self.out_report.get_path(), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines[:12]), flush=True)


def synthetic_wer_py(*, tag: str, tts_hf: tk.Path, annotated_hf: tk.Path, sample: int = 0):
    """Attach a WER readout to one corpus and register it under ``output/corpus_wer/<tag>/``."""
    job = SyntheticSpeechWer(tts_hf=tts_hf, annotated_hf=annotated_hf, sample=sample)
    tk.register_output(f"corpus_wer/{tag}/wer.json", job.out_json)
    tk.register_output(f"corpus_wer/{tag}/report.txt", job.out_report)
    return job
