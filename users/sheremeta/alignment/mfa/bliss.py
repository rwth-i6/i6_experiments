"""MFA integration for Bliss-format datasets (like LibriSpeech)."""

import os
import shutil
from pathlib import Path as SysPath
from typing import Any, Dict, Optional

from sisyphus import Job, Task, tk
from i6_core.lib import corpus

from .core import (
    _DEFAULT_MFA_OVERRIDES,
    EnsureMfaModelsJob,
    _MfaAlignBase,
    _normalize_orth,
)


class PrepareMfaCorpusFromBlissJob(Job):
    """Extract audio paths and transcripts from a Bliss corpus into a TSV manifest (``utt_id \\t audio_path \\t transcript``)."""

    def __init__(self, bliss_corpus: tk.Path, max_utts: int | None = None):
        self.bliss_corpus = bliss_corpus
        self.max_utts = max_utts

        self.out_manifest = self.output_path("manifest.tsv")
        self.out_num_utts = self.output_var("num_utts")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        num_utts = 0
        seen_utt_ids: set[str] = set()

        with open(self.out_manifest.get_path(), "w", encoding="utf-8") as out_f:
            for segment in c.segments():
                if not getattr(segment, "orth", None):
                    continue
                if self.max_utts is not None and num_utts >= self.max_utts:
                    break

                audio_path = SysPath(segment.recording.audio)
                # audio stem is the downstream join key: RunMfaAlignJob maps stem -> segment.fullname() (the seq_tag)
                utt_id = audio_path.stem

                # mfa flattens everything into one corpus dir keyed by stem, so colliding stems overwrite silently
                if utt_id in seen_utt_ids:
                    raise RuntimeError(f"Duplicate utterance id {utt_id!r}")
                seen_utt_ids.add(utt_id)

                text = _normalize_orth(segment.orth)
                if "\t" in text or "\n" in text:
                    raise RuntimeError(
                        f"Transcript for {utt_id!r} contains tab/newline; unsupported by TSV"
                    )

                out_f.write(f"{utt_id}\t{audio_path}\t{text}\n")
                num_utts += 1

        self.out_num_utts.set(num_utts)


class RunMfaAlignJob(_MfaAlignBase):
    """Run MFA on a Bliss corpus staged from a TSV manifest, emitting word alignments as JSONL."""

    def __init__(
        self,
        bliss_corpus: tk.Path,
        corpus_manifest: tk.Path,
        ensure_models_done: tk.Variable,
        dictionary: str = "english_us_arpa",
        acoustic_model: str = "english_us_arpa",
        mfa_config_overrides: Optional[Dict[str, Any]] = None,
        use_symlinks: bool = True,
        skip_missing: bool = True,
        debug: bool = False,
    ):
        self.bliss_corpus = bliss_corpus
        self.corpus_manifest = corpus_manifest
        self.ensure_models_done = ensure_models_done
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.use_symlinks = use_symlinks
        self.skip_missing = skip_missing
        self.debug = debug

        if mfa_config_overrides is not None:
            self.mfa_config_overrides = mfa_config_overrides
        else:
            self.mfa_config_overrides = _DEFAULT_MFA_OVERRIDES

        self.out_alignments = self.output_path("word_alignments.jsonl.gz")
        self.out_missing = self.output_path("missing.txt")
        self.out_num_seqs = self.output_var("num_seqs")
        self.out_num_words = self.output_var("num_words")
        self.out_align_log = self.output_path("align.log")

    def _materialize_corpus(self, corpus_dir: SysPath) -> None:
        # mfa expects a flat corpus dir: one audio + one same-stem .lab per utt
        corpus_dir.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_manifest.get_path(), encoding="utf-8") as f:
            for line in f:
                utt_id, audio_path, text = line.rstrip("\n").split("\t", 2)
                audio_src = SysPath(audio_path)
                audio_dst = corpus_dir / audio_src.name
                lab_dst = corpus_dir / f"{utt_id}.lab"

                if not audio_dst.exists():
                    if self.use_symlinks:
                        os.symlink(audio_src, audio_dst)
                    else:
                        shutil.copy2(audio_src, audio_dst)

                with open(lab_dst, "w", encoding="utf-8") as lf:
                    lf.write(text + "\n")

    def _build_seq_tag_map(self) -> dict:
        """Map each audio stem to the bliss segment fullname (the RETURNN seq_tag the store is keyed by)."""
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        seq_tag_map: dict = {}
        for recording in c.all_recordings():
            # one segment per recording, the invariant PrepareMfaCorpusFromBlissJob relies on
            if len(recording.segments) != 1:
                raise ValueError(
                    f"Expected exactly one segment per recording (matches "
                    f"PrepareMfaCorpusFromBlissJob); got {len(recording.segments)} "
                    f"for recording {recording.fullname()!r}."
                )
            segment = recording.segments[0]
            audio_stem = SysPath(recording.audio).stem
            seq_tag_map[audio_stem] = segment.fullname()
        return seq_tag_map

    def _materialize(self, corpus_dir: SysPath) -> dict:
        self._materialize_corpus(corpus_dir)
        return self._build_seq_tag_map()


def align_bliss_corpus(
    bliss: tk.Path,
    *,
    max_utts: Optional[int] = None,
    dictionary: str = "english_us_arpa",
    acoustic_model: str = "english_us_arpa",
    mfa_config_overrides: Optional[Dict[str, Any]] = None,
    use_symlinks: bool = True,
    skip_missing: bool = True,
    alias: Optional[str] = None,
) -> tk.Path:
    """Main pipeline for aligning a Bliss dataset."""
    ensure = EnsureMfaModelsJob(acoustic_model=acoustic_model, dictionary=dictionary)
    prep = PrepareMfaCorpusFromBlissJob(bliss_corpus=bliss, max_utts=max_utts)
    align = RunMfaAlignJob(
        bliss_corpus=bliss,
        corpus_manifest=prep.out_manifest,
        ensure_models_done=ensure.out_done,
        dictionary=dictionary,
        acoustic_model=acoustic_model,
        mfa_config_overrides=mfa_config_overrides,
        use_symlinks=use_symlinks,
        skip_missing=skip_missing,
    )
    if alias:
        align.add_alias(alias)
        tk.register_output(f"{alias}/word_alignments.jsonl.gz", align.out_alignments)
    return align.out_alignments
