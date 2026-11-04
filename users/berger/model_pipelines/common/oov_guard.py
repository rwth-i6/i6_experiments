"""Opt-in selected-trace degeneracy checks for the 101ed636 experiments.

This is deliberately separate from scoring normalization. Piece count and word
boundaries are properties of the raw tokenizer stream, not normalized text.
"""

import json

from ...data.oov_diagnostics import read_lexicon
from .oov_traceback import GraniteOovTracebackFormatter
from .recog import BpeTracebackFormatter


class GuardedOovTracebackFormatter:
    """Log per-utterance/cumulative diagnostics and abort degenerate recognition.

    Thresholds are explicit experimental guardrails, not universal validity rules.
    Actual LM-token events are not exposed by librasr's traceback: exact lexical
    matches below describe the expected resolution, not an observed UNK event.
    """

    def __init__(
        self,
        lexicon,
        tokenization,
        huggingface_repo_dir=None,
        max_unknown_pieces=64,
        long_reference_words=10,
        min_word_ratio=0.5,
    ):
        if tokenization not in {"continuation-marked", "word-start-marked"}:
            raise ValueError(tokenization)
        if tokenization == "word-start-marked" and huggingface_repo_dir is None:
            raise ValueError("Granite detokenization requires its tokenizer")
        self.tokenization = tokenization
        self.formatter = (
            GraniteOovTracebackFormatter(huggingface_repo_dir)
            if tokenization == "word-start-marked"
            else BpeTracebackFormatter()
        )
        root = read_lexicon(lexicon)
        self.roles = {}
        self.known_prons = set()
        for lemma in root.findall("lemma"):
            special = lemma.get("special", "")
            if special.startswith("unknown-"):
                prons = [tuple((p.text or "").split()) for p in lemma.findall("phon")]
                if len(prons) != 1 or len(prons[0]) != 1:
                    raise ValueError("Guard requires one fallback label per lemma")
                for orth in lemma.findall("orth"):
                    if orth.text in self.roles:
                        raise ValueError(f"Ambiguous fallback orthography: {orth.text}")
                    self.roles[orth.text] = (special, prons[0][0])
            elif not special:
                self.known_prons.update(tuple((p.text or "").split()) for p in lemma.findall("phon"))
        self.max_unknown_pieces = max_unknown_pieces
        self.long_reference_words = long_reference_words
        self.min_word_ratio = min_word_ratio
        self.num_utts = self.ref_words = self.hyp_words = self.short_utts = 0

    def validate_traceback(self, traceback, reference, seq_tag):
        pending, runs = [], []

        def flush():
            if pending:
                runs.append(tuple(pending))
                pending.clear()

        for item in traceback:
            orth = item.lemma
            if orth in self.roles:
                role, label = self.roles[orth]
                if role.startswith("unknown-word-start"):
                    flush()
                if role.endswith("+nonword"):
                    continue
                pending.append(label)
                if role == "unknown-final":
                    flush()
            elif orth and not (orth.startswith("[") or orth.startswith("<")):
                flush()
        flush()
        unknown_lengths = [len(p) for p in runs if p not in self.known_prons]
        hypothesis = self.formatter.traceback_to_transcription(traceback)
        hyp_count = len(hypothesis.split())
        ref_count = len(str(reference).split())
        self.num_utts += 1
        self.ref_words += ref_count
        self.hyp_words += hyp_count
        self.short_utts += int(hyp_count <= 2)
        report = dict(
            tag=str(seq_tag),
            reference_words=ref_count,
            recognized_words=hyp_count,
            max_unknown_pieces=max(unknown_lengths, default=0),
            fallback_words=len(runs),
            exact_lexical_pronunciations=sum(p in self.known_prons for p in runs),
            num_utterances=self.num_utts,
            mean_reference_words=self.ref_words / self.num_utts,
            mean_recognized_words=self.hyp_words / self.num_utts,
            fraction_one_or_two_words=self.short_utts / self.num_utts,
        )
        reasons = []
        if report["max_unknown_pieces"] > self.max_unknown_pieces:
            reasons.append("unknown-word piece limit")
        if ref_count >= self.long_reference_words and hyp_count <= 2:
            reasons.append("long reference collapsed to <=2 words")
        if ref_count >= self.long_reference_words and hyp_count < self.min_word_ratio * ref_count:
            reasons.append("recognized/reference word ratio")
        report["failures"] = reasons
        if reasons:
            # This check runs before the regular traceback printer: preserve the
            # offending output and raw pronunciations in the failing job's log.
            report["hypothesis"] = hypothesis
            report["unattested_fallback_pronunciations"] = [p for p in runs if p not in self.known_prons]
        print("OOV_DIAGNOSTIC " + json.dumps(report), flush=True)
        if reasons:
            raise ValueError(f"Degenerate OOV recognition in {seq_tag}: {', '.join(reasons)}")

    def traceback_to_transcription(self, traceback):
        return self.formatter.traceback_to_transcription(traceback)

    def traceback_to_ctm_str(self, traceback, ms_per_frame):
        return self.formatter.traceback_to_ctm_str(traceback, ms_per_frame)
