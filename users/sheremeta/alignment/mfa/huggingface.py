"""Sharded MFA alignment for HuggingFace datasets, one scratch-staged shard per job."""

import re
from pathlib import Path as SysPath
from typing import Any, Dict, List, Optional, Set

from sisyphus import tk

from .core import (
    _DEFAULT_MFA_OVERRIDES,
    EnsureMfaModelsJob,
    MergeWordAlignmentStoresJob,
    _MfaAlignBase,
    _normalize_orth,
)


_STEM_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_AVAILABLE_FORMATS = ("ogg", "wav", "flac", "aiff", "vorbis")


def _unique_stem(seq_tag: str, used: Set[str]) -> str:
    """Generate a filename-safe, collision-free stem for a sequence tag."""
    # cap length to stay under filesystem name limits once the format suffix is appended
    base = (_STEM_UNSAFE.sub("_", seq_tag).strip("._") or "seq")[:180]
    stem = base
    i = 1
    while stem in used:
        i += 1
        stem = f"{base}_{i}"
    used.add(stem)
    return stem


class RunMfaAlignHuggingFaceShardJob(_MfaAlignBase):
    """Align a single shard of a HuggingFace dataset split, emitting word alignments as JSONL."""

    __sis_hash_exclude__ = {"after": None}

    def __init__(
        self,
        *,
        hf_dataset_dir: tk.Path,
        ensure_models_done: tk.Variable,
        num_shards: int,
        shard_index: int,
        split: str = "train",
        id_column: str = "id",
        audio_column: str = "audio",
        text_column: str = "text",
        sample_rate: int = 16000,
        materialize_format: str = "ogg",
        dictionary: str = "english_us_arpa",
        acoustic_model: str = "english_us_arpa",
        mfa_config_overrides: Optional[Dict[str, Any]] = None,
        skip_missing: bool = True,
        debug: bool = False,
        after: Optional[tk.Path] = None,
    ):
        assert 0 <= shard_index < num_shards, (shard_index, num_shards)
        assert materialize_format in _AVAILABLE_FORMATS, materialize_format
        self.hf_dataset_dir = hf_dataset_dir
        self.ensure_models_done = ensure_models_done
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.split = split
        self.id_column = id_column
        self.audio_column = audio_column
        self.text_column = text_column
        self.sample_rate = sample_rate
        self.materialize_format = materialize_format
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.mfa_config_overrides = mfa_config_overrides or _DEFAULT_MFA_OVERRIDES
        self.skip_missing = skip_missing
        self.debug = debug
        # graph-only edge for concurrency capping, the path itself is never read at runtime
        self.after = after

        self.out_alignments = self.output_path("word_alignments.jsonl.gz")
        self.out_missing = self.output_path("missing.txt")
        self.out_num_seqs = self.output_var("num_seqs")
        self.out_num_words = self.output_var("num_words")
        self.out_align_log = self.output_path("align.log")

    def _scratch_name(self) -> str:
        return f"hf_{self.shard_index}"

    def _materialize(self, corpus_dir: SysPath) -> Dict[str, str]:
        """Stage this shard's audio + transcripts into ``corpus_dir``, returning {stem: seq_tag}."""
        from datasets import Audio, load_from_disk

        corpus_dir.mkdir(parents=True, exist_ok=True)
        obj = load_from_disk(self.hf_dataset_dir.get_path())
        ds = obj[self.split] if hasattr(obj, "keys") else obj
        # contiguous so the shard -> arrow mapping stays stable across jobs
        ds = ds.shard(
            num_shards=self.num_shards, index=self.shard_index, contiguous=True
        )

        fmt = self.materialize_format
        if fmt == "ogg":
            # decode=False writes the stored ogg bytes verbatim, no decode/re-encode round trip
            ds = ds.cast_column(self.audio_column, Audio(decode=False))

            def _write_audio(row, path: SysPath) -> None:
                path.write_bytes(row[self.audio_column]["bytes"])

        else:
            import soundfile as sf

            ds = ds.cast_column(
                self.audio_column, Audio(sampling_rate=self.sample_rate)
            )
            _subtype = "PCM_16" if fmt == "wav" else None

            def _write_audio(row, path: SysPath) -> None:
                sf.write(
                    str(path),
                    row[self.audio_column]["array"],
                    self.sample_rate,
                    subtype=_subtype,
                )

        seq_tag_map: Dict[str, str] = {}
        used: Set[str] = set()
        total = len(ds)
        for i, row in enumerate(ds):
            if i % 20_000 == 0:
                print(f"materialize: {i}/{total} seqs", flush=True)
            text = _normalize_orth(str(row[self.text_column]))
            if not text:
                # mfa can't align empty transcripts, skip so the consumer drops it
                continue
            seq_tag = str(row[self.id_column])
            stem = _unique_stem(seq_tag, used)
            _write_audio(row, corpus_dir / f"{stem}.{fmt}")
            lab_file = corpus_dir / f"{stem}.lab"
            lab_file.write_text(text + "\n", encoding="utf-8")
            seq_tag_map[stem] = seq_tag
        return seq_tag_map


def align_huggingface_dataset(
    hf_dir: tk.Path,
    *,
    shards_per_split: Dict[str, int],
    id_column: str = "id",
    audio_column: str = "audio",
    text_column: str = "text",
    sample_rate: int = 16000,
    materialize_format: str = "ogg",
    dictionary: str = "english_us_arpa",
    acoustic_model: str = "english_us_arpa",
    mfa_config_overrides: Optional[Dict[str, Any]] = None,
    max_concurrent: Optional[int] = 4,
    alias: Optional[str] = None,
) -> tk.Path:
    """Align a HuggingFace dataset, sharded and capped at ``max_concurrent`` in-flight shards.

    The cap chains shard i onto shard i - max_concurrent so staged shards don't blow through
    hpcwork's file-count quota.
    """
    ensure = EnsureMfaModelsJob(acoustic_model=acoustic_model, dictionary=dictionary)

    shard_outs: List[tk.Path] = []
    for split, num_shards in shards_per_split.items():
        split_outs: List[tk.Path] = []
        for i in range(num_shards):
            after = None
            if max_concurrent is not None and i >= max_concurrent:
                after = split_outs[i - max_concurrent]
            job = RunMfaAlignHuggingFaceShardJob(
                hf_dataset_dir=hf_dir,
                ensure_models_done=ensure.out_done,
                num_shards=num_shards,
                shard_index=i,
                split=split,
                id_column=id_column,
                audio_column=audio_column,
                text_column=text_column,
                sample_rate=sample_rate,
                materialize_format=materialize_format,
                dictionary=dictionary,
                acoustic_model=acoustic_model,
                mfa_config_overrides=mfa_config_overrides,
                after=after,
            )
            if alias:
                job.add_alias(
                    f"{alias}/align/{split}/shard_{i:04d}_of_{num_shards:04d}"
                )
            split_outs.append(job.out_alignments)
        shard_outs.extend(split_outs)

    merge = MergeWordAlignmentStoresJob(shard_outs)
    if alias:
        merge.add_alias(f"{alias}/merge")
        tk.register_output(f"{alias}/word_alignments.jsonl.gz", merge.out_store)
        tk.register_output(f"{alias}/num_seqs", merge.out_num_seqs)
    return merge.out_store
