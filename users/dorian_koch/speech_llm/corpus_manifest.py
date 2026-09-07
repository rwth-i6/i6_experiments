"""What a training corpus mix ACTUALLY contains (backlog F8).

``Corpus`` takes ``[(path, weight, window_sec)]`` and states nothing about the realised size, so the
arithmetic that matters -- how many hours, how many rows, how much of an epoch a run covers -- had to
be recomputed by hand from row counts and durations every time it was needed. The epoch figure behind
"collapse happens at 1.7% of an epoch" was derived that way. This job records it instead.

It is nearly free: our corpora carry a ``duration`` column, so the whole manifest is a read of one
float column per corpus (0.1-0.7 s for 60k rows) with no audio decoded. A corpus without that column
falls back to arrow's binary offsets -- ``pyarrow.compute.binary_length`` on the WAV bytes, which
reads the offsets buffer and not the data.

⚠ Never call ``combine_chunks()`` on the audio column to do that: our corpora exceed the 2 GB offset
limit of a single arrow array and it raises ``offset overflow while concatenating arrays``. Iterate
the chunks. (TurnBench's loader documents hitting the same wall from the ``load_dataset`` side.)

Registers ``corpus/<tag>/manifest.{json,txt}``; the txt is the one to read.
"""

import json
import os

import numpy as np
from sisyphus import Job, Task, tk


def _label(path: str) -> str:
    """A readable name for a corpus path.

    Every corpus lives at ``work/<...>/<Job>.<hash>/output/<name>``, so both a basename and a
    dirname come back as ``output`` -- which is what the first run of this job printed for two
    different corpora. Prefer the ``<Job>.<hash>`` component.
    """
    parts = [p for p in path.split("/") if p]
    for part in reversed(parts):
        if "." in part and not part.startswith("."):
            return part
    return "/".join(parts[-2:]) if len(parts) >= 2 else path


def _durations(table) -> "np.ndarray | None":
    """Per-row duration in seconds, or None if it cannot be established cheaply."""
    if "duration" in table.column_names:
        col = table.column("duration")
        chunks = col.chunks if col.num_chunks else [col]
        return np.concatenate([np.asarray(c, dtype="float64") for c in chunks])
    audio_col = next((c for c in ("audio_assistant", "audio") if c in table.column_names), None)
    if audio_col is None:
        return None
    import io
    import wave

    import pyarrow.compute as pc

    raw = table.column(audio_col)[0].as_py()["bytes"]
    with wave.open(io.BytesIO(raw)) as w:
        sr, ch, sw, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    header = len(raw) - frames * ch * sw  # WAV headers are not a fixed 44 bytes
    lens = []
    col = table.column(audio_col)
    for chunk in col.chunks if col.num_chunks else [col]:
        lens.append(np.asarray(pc.binary_length(chunk.field("bytes")), dtype="float64"))
    return (np.concatenate(lens) - header) / (sr * ch * sw)


class CorpusManifest(Job):
    """Row counts, hours and epoch arithmetic for one corpus mix. Login-node mini_task."""

    __sis_hash_exclude__ = {"note": ""}

    def __init__(self, *, tag, entries, duration_sec, batch_sequences=None, max_steps=None, note=""):
        """`entries`: list of (tk.Path, weight, window_sec|None) -- paths in VALUE position."""
        self.tag = tag
        self.entries = entries
        self.duration_sec = duration_sec
        self.batch_sequences = batch_sequences
        self.max_steps = max_steps
        self.note = note
        self.out_json = self.output_path("manifest.json")
        self.out_txt = self.output_path("manifest.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # Read the arrow table directly rather than through the lib's read_arrow_table: this is a
        # login-node mini_task in the MANAGER's venv, and the lib pulls torch.
        from datasets import load_from_disk

        total_weight = sum(float(w) for _, w, _ in self.entries) or 1.0
        corpora, lines = [], []
        for path, weight, window_sec in self.entries:
            p = path.get() if hasattr(path, "get") else str(path)
            table = load_from_disk(p).data.table
            d = _durations(table)
            share = float(weight) / total_weight
            rec = {
                "path": p,
                "label": _label(p),
                "weight": float(weight),
                "draw_share": share,
                "rows": table.num_rows,
                "window_sec": window_sec,
                "columns": list(table.column_names),
            }
            if d is not None and len(d):
                rec.update(
                    hours=float(d.sum() / 3600),
                    mean_sec=float(d.mean()),
                    median_sec=float(np.median(d)),
                    min_sec=float(d.min()),
                    max_sec=float(d.max()),
                )
                if window_sec:
                    # In-loader windowing: a draw yields `window_sec`, and window POSITION is the
                    # augmentation -- so how much slack a row has is the thing worth stating. A row
                    # only as long as the window is a fixed crop wearing a windowing config.
                    slack = d - float(window_sec)
                    rec["window_slack_median_sec"] = float(np.median(slack))
                    rec["rows_with_no_slack"] = int((slack <= 0).sum())
            corpora.append(rec)

        manifest = {
            "tag": self.tag,
            "note": self.note,
            "duration_sec": self.duration_sec,
            "batch_sequences": self.batch_sequences,
            "max_steps": self.max_steps,
            "corpora": corpora,
        }

        # Epoch arithmetic: what the RUN actually draws, against what the corpus holds.
        if self.batch_sequences and self.max_steps:
            per_step_h = self.batch_sequences * self.duration_sec / 3600
            drawn_h = per_step_h * self.max_steps
            manifest["run"] = {
                "audio_per_step_min": per_step_h * 60,
                "total_audio_drawn_h": drawn_h,
                "epochs_per_corpus": {
                    c["path"]: (drawn_h * c["draw_share"] / c["hours"]) if c.get("hours") else None for c in corpora
                },
            }

        lines.append(f"corpus manifest: {self.tag}" + (f"  ({self.note})" if self.note else ""))
        lines.append(f"  sequence length {self.duration_sec}s")
        for c in corpora:
            lines.append(f"  - {_label(c['path'])}  weight {c['weight']:g} (draw {100 * c['draw_share']:.0f}%)")
            if "hours" in c:
                lines.append(
                    f"      {c['rows']:,} rows, {c['hours']:.1f} h, "
                    f"mean {c['mean_sec']:.1f}s median {c['median_sec']:.1f}s "
                    f"range {c['min_sec']:.1f}-{c['max_sec']:.1f}s"
                )
            else:
                lines.append(f"      {c['rows']:,} rows, duration unavailable")
            if c.get("window_sec"):
                lines.append(
                    f"      windowed to {c['window_sec']:g}s on load; median slack "
                    f"{c.get('window_slack_median_sec', float('nan')):.1f}s, "
                    f"{c.get('rows_with_no_slack', 0):,} rows with none"
                )
        if "run" in manifest:
            r = manifest["run"]
            lines.append(
                f"  run: {r['audio_per_step_min']:.1f} min audio/step x {self.max_steps:,} steps "
                f"= {r['total_audio_drawn_h']:,.0f} h drawn"
            )
            for path, ep in r["epochs_per_corpus"].items():
                if ep is not None:
                    lines.append(f"      {_label(path)}: {ep:.2f} epochs")

        with open(self.out_json.get(), "w") as f:
            json.dump(manifest, f, indent=2)
        text = "\n".join(lines) + "\n"
        with open(self.out_txt.get(), "w") as f:
            f.write(text)
        print(text, flush=True)


def corpus_manifest_py(tag: str, corpus, *, batch_sequences=None, max_steps=None, note=""):
    """Register a manifest for a ``Corpus``. Returns the job.

    Accepts ``Corpus.mix`` in any of its shapes: a bare path, ``(path, weight)`` or
    ``(path, weight, window_sec)`` tuples.
    """
    mix = corpus.mix
    if not isinstance(mix, (list, tuple)):
        entries = [(mix, 1.0, None)]
    else:
        entries = [
            (e[0], e[1], e[2] if len(e) > 2 else None) if isinstance(e, (list, tuple)) else (e, 1.0, None) for e in mix
        ]
    job = CorpusManifest(
        tag=tag,
        entries=entries,
        duration_sec=corpus.duration_sec,
        batch_sequences=batch_sequences,
        max_steps=max_steps,
        note=note,
    )
    tk.register_output(f"corpus/{tag}/manifest.json", job.out_json)
    tk.register_output(f"corpus/{tag}/manifest.txt", job.out_txt)
    job.add_alias(f"corpus/{tag}")
    return job
