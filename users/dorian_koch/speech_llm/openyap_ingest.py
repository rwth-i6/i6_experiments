"""Open Yap 1K -> the PersonaPlex codes corpus, through the JRE pipeline's own downstream stages.

Open Yap (``TheAgenticDataCompany/open-yap-1k``, CC-BY-4.0 under the vendor's Data Use Agreement, 1,000 h,
48 kHz) is already TWO-CHANNEL: one mono FLAC per speaker, speaker labelled by construction, word timings
shipped (Deepgram Nova-3). So the JRE ingest's diarization, separation, permutation repair and ASR have
nothing to do here. Everything AFTER the episode rows is the JRE pipeline unchanged: ``PodcastDialogueSlice``
(DuplexChat's ``extract_valid_dialogues``), ``PodcastCodesTrainData`` windows, ``PodcastWindowContext``,
``PersonaPromptGen``, ``VoicePromptCodes``, ``AttachPersonaPrompts``, ``EpisodeEdgeRows``.

    OpenYapIndex          CPU, one streaming pass: per-conversation metadata + statistics (+ an audio sample)
    OpenYapShards         mini: filters, held-out conversations, balanced shards
    OpenYapEpisodeIngest  GPU per shard: decode 48 -> 24 kHz (our ffmpeg), loudness norm, OFFLINE mimi encode,
                          user-stream variants -> episode rows in ``PodcastEpisodeIngest``'s schema

The pure helpers live in ``moshi_family/openyap_common.py`` (loaded by path here and in the workers).
Decisions and measurements: projects/2026-01-speech-llm/audio_datasets.md, "Open Yap".
"""

from __future__ import annotations

import json
import os

from sisyphus import Job, Task, tk

from i6_experiments.users.dorian_koch.speech_llm.common import (
    job_progress_fraction,
    moshi_family_lib_parent,
    run_worker_script,
)
from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import _moshi_pythonpath
from i6_experiments.users.dorian_koch.speech_llm.tts import InstallFFmpeg


def openyap_common():
    """``moshi_family/openyap_common.py`` loaded by path (the package ``__init__`` needs the moshi stack)."""
    import importlib.util

    path = os.path.join(moshi_family_lib_parent(), "moshi_family", "openyap_common.py")
    spec = importlib.util.spec_from_file_location("openyap_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OpenYapIndex(Job):
    """One streaming pass over the archive: ``conversations.jsonl`` (vendor metadata + word-table checks + turn
    statistics at several segment gaps), ``audio_sample.jsonl`` (audio measurements on a seeded hash sample:
    loudness, bandwidth, cross-channel bleed, word-onset alignment) and ``summary.json``. Worker:
    ``moshi_family/openyap_index_main.py``. Nothing is extracted to disk."""

    def __init__(
        self,
        *,
        archive: tk.Path,
        venv_python,
        sample_frac: float = 0.03,
        seed: int = 0,
        gaps: tuple = (0.2, 0.3, 0.5, 0.8, 1.0),
        code_version: int = 1,
    ):
        self.archive = archive
        self.venv_python = venv_python
        self.sample_frac = float(sample_frac)
        self.seed = int(seed)
        self.gaps = tuple(float(g) for g in gaps)
        self.code_version = int(code_version)
        self.out_dir = self.output_path("index", directory=True)
        self.rqmt = {"cpu": 2, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        lib = moshi_family_lib_parent()
        run_worker_script(
            self.venv_python.get(),
            os.path.join(lib, "moshi_family", "openyap_index_main.py"),
            [
                "--archive",
                self.archive.get_path(),
                "--common",
                os.path.join(lib, "moshi_family", "openyap_common.py"),
                "--out_dir",
                self.out_dir.get_path(),
                "--sample_frac",
                self.sample_frac,
                "--seed",
                self.seed,
                "--gaps",
                ",".join(str(g) for g in self.gaps),
            ],
            log_label="Open Yap index",
            with_hf_home=False,
        )


class OpenYapShards(Job):
    """Filter the indexed conversations, set aside ``heldout_n`` seeded held-out conversations, and bin-pack the rest
    into ``num_shards`` duration-balanced shards (longest first into the lightest). Outputs ``shards/shard_<k>.jsonl``
    (``{id, duration_sec}`` per line, in ARCHIVE order, so a pilot over the first conversations of a shard stops
    reading the archive early), ``shards/shard_<num_shards>.jsonl`` = the held-out conversations, and
    ``summary.json`` (every filter's tally and hours, per-shard hours, the held-out ids).

    Filters (each hashed; None = off): ``min_bandwidth_hz`` on min(track a, track b) of the vendor's
    ``effective_bandwidth_hz`` (a conversation is kept or dropped whole, never half a duplex pair);
    ``min_duration_sec``; ``language``. Held-out draw: ``seeded_unit(heldout_seed, id)`` smallest ``heldout_n`` among
    the kept conversations."""

    def __init__(
        self,
        *,
        index_dir: tk.Path,
        num_shards: int,
        heldout_n: int = 32,
        heldout_seed: int = 0,
        min_bandwidth_hz: float | None = None,
        min_duration_sec: float | None = None,
        language: str | None = "en",
    ):
        self.index_dir = index_dir
        self.num_shards = int(num_shards)
        self.heldout_n = int(heldout_n)
        self.heldout_seed = int(heldout_seed)
        self.min_bandwidth_hz = None if min_bandwidth_hz is None else float(min_bandwidth_hz)
        self.min_duration_sec = None if min_duration_sec is None else float(min_duration_sec)
        self.language = language
        self.out_dir = self.output_path("shards", directory=True)
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        C = openyap_common()
        rows = [json.loads(line) for line in open(os.path.join(self.index_dir.get_path(), "conversations.jsonl"))]
        order = {r["id"]: i for i, r in enumerate(rows)}
        tally, kept = {}, []

        def drop(reason, r):
            t = tally.setdefault(reason, {"n": 0, "hours": 0.0})
            t["n"] += 1
            t["hours"] = round(t["hours"] + float(r["duration_sec"] or 0) / 3600.0, 3)

        for r in rows:
            bw = [r[t].get("bandwidth_hz") for t in ("a", "b")]
            if r.get("missing"):
                drop("missing_files", r)
            elif self.language is not None and r.get("language") != self.language:
                drop("language", r)
            elif self.min_duration_sec is not None and float(r["duration_sec"] or 0) < self.min_duration_sec:
                drop("short", r)
            elif self.min_bandwidth_hz is not None and (None in bw or min(bw) < self.min_bandwidth_hz):
                drop("bandwidth", r)
            else:
                kept.append(r)
        held = sorted(kept, key=lambda r: C.seeded_unit(self.heldout_seed, r["id"]))[: self.heldout_n]
        held_ids = {r["id"] for r in held}
        train = [r for r in kept if r["id"] not in held_ids]
        bins = C.bin_pack([(r["id"], float(r["duration_sec"] or 0)) for r in train], self.num_shards)
        dur = {r["id"]: float(r["duration_sec"] or 0) for r in rows}
        out = self.out_dir.get_path()
        for k, ids in enumerate([*bins, sorted(held_ids)]):
            with open(os.path.join(out, f"shard_{k:05d}.jsonl"), "w") as f:
                for cid in sorted(ids, key=order.__getitem__):
                    f.write(json.dumps({"id": cid, "duration_sec": dur[cid]}) + "\n")
        summary = {
            "n_indexed": len(rows),
            "hours_indexed": round(sum(dur.values()) / 3600.0, 2),
            "dropped": tally,
            "n_kept": len(kept),
            "hours_kept": round(sum(dur[r["id"]] for r in kept) / 3600.0, 2),
            "heldout": {
                "n": len(held_ids),
                "hours": round(sum(dur[c] for c in held_ids) / 3600.0, 2),
                "ids": sorted(held_ids),
            },
            "shard_hours": [round(sum(dur[c] for c in b) / 3600.0, 2) for b in bins],
            "heldout_shard": self.num_shards,
            "filters": {
                "min_bandwidth_hz": self.min_bandwidth_hz,
                "min_duration_sec": self.min_duration_sec,
                "language": self.language,
            },
        }
        with open(self.out_summary.get_path(), "w") as f:
            json.dump(summary, f, indent=1)


class OpenYapEpisodeIngest(Job):
    """One shard of Open Yap conversations -> whole-conversation code rows in ``PodcastEpisodeIngest``'s EPISODE
    schema (``podcast_mimi_encode.part_features("episode")``), so ``PodcastCodesIndex``, ``PodcastDialogueSlice``,
    ``VoicePromptCodes`` and ``EpisodeEdgeRows`` read them unchanged. Worker ``moshi_family/openyap_ingest_main.py``
    (module docstring: decode, word-derived segments, reference clips, loudness, offline encode, variants).

    Hashed: the archive, the shard file, ``gap`` (segment rule), ``loudness_norm``, ``user_augment`` (+
    ``noise_pool``), ``mimi_encode_mode`` (offline only: the streaming chunked encode is wrong, CLAUDE.md),
    ``max_items`` (a pilot is its own job), ``code_version``. Not hashed: ``env_ffmpeg_path`` (where our ffmpeg
    lives), ``convs_per_part`` (arrow layout), ``rqmt``. Resumable (``resume="run"``): never
    ``hpc-rerun --include-interrupted`` it."""

    __sis_hash_exclude__ = {"max_items": 0}

    def __init__(
        self,
        *,
        mimi_venv_python,
        archive: tk.Path,
        shards_dir: tk.Path,
        shard_idx: int,
        gap: float,
        loudness_norm: dict,
        user_augment: dict | None = None,
        noise_pool: tk.Path | None = None,
        mimi_encode_mode: str = "offline",
        max_items: int = 0,
        provenance: dict | None = None,
        code_version: int = 1,
        env_ffmpeg_path: tk.Path | None = None,
        convs_per_part: int = 32,
        rqmt: dict | None = None,
    ):
        assert mimi_encode_mode == "offline", "only the offline encode is exact (audio_datasets.md, Mimi encode bug)"
        assert loudness_norm.get("mode") == "per_track", loudness_norm
        draws_noise = user_augment is not None and float(user_augment["augment"].get("noise_prob", 0.0)) > 0.0
        if draws_noise != (noise_pool is not None):
            raise ValueError("pass noise_pool exactly when the user augmentation draws noise")
        self.mimi_venv_python = mimi_venv_python
        self.archive = archive
        self.shards_dir = shards_dir
        self.shard_idx = int(shard_idx)
        self.gap = float(gap)
        self.loudness_norm = dict(sorted(loudness_norm.items()))
        self.user_augment = user_augment
        self.noise_pool = noise_pool
        self.mimi_encode_mode = mimi_encode_mode
        self.max_items = int(max_items)
        self.provenance = dict(provenance or {})
        self.code_version = int(code_version)
        self.env_ffmpeg_path = env_ffmpeg_path
        self.convs_per_part = int(convs_per_part)
        self.out_dir = self.output_path("codes", directory=True)
        # host RAM: one conversation of 24 kHz stereo float32 (~0.35 GB/h) x a few copies + two waiting in the queue
        self.rqmt = rqmt or {"gpu": 1, "cpu": 6, "mem": 32, "time": 8, "gpu_mem_gb": 16}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(parsed_args)
        for k in ("env_ffmpeg_path", "convs_per_part", "rqmt"):
            d.pop(k, None)
        return super().hash(d)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        try:
            with open(os.path.join(self.out_dir.get(), "progress.json")) as f:
                p = json.load(f)
            return f"{p.get('ok', 0)}/{p.get('total', 0)} conversations ({p.get('failed', 0)} failed), {p.get('audio_hours', 0)} h"
        except Exception:  # noqa: BLE001
            return None

    def run(self):
        if self.env_ffmpeg_path is None:
            raise ValueError("env_ffmpeg_path is required: decode + resample go through OUR ffmpeg")
        lib = moshi_family_lib_parent()
        args = [
            "--archive",
            self.archive.get_path(),
            "--common",
            os.path.join(lib, "moshi_family", "openyap_common.py"),
            "--shard_jsonl",
            os.path.join(self.shards_dir.get_path(), f"shard_{self.shard_idx:05d}.jsonl"),
            "--out_dir",
            self.out_dir.get_path(),
            "--ffmpeg_dir",
            self.env_ffmpeg_path.get_path(),
            "--gap",
            self.gap,
            "--encode_mode",
            self.mimi_encode_mode,
            "--loudness_norm",
            json.dumps(self.loudness_norm, sort_keys=True),
            "--convs_per_part",
            self.convs_per_part,
            "--provenance",
            json.dumps({**self.provenance, "shard_idx": self.shard_idx}, sort_keys=True),
        ]
        if self.user_augment is not None:
            aug = dict(self.user_augment)
            if self.noise_pool is not None:
                aug["noise_pool"] = self.noise_pool.get_path()
            args += ["--user_augment", json.dumps(aug, sort_keys=True)]
        if self.max_items:
            args += ["--max_items", self.max_items]
        _ff = self.env_ffmpeg_path

        def env_hook(e, _ff=_ff):
            InstallFFmpeg.add_to_env(_ff, e)

        run_worker_script(
            self.mimi_venv_python.get(),
            os.path.join(lib, "moshi_family", "openyap_ingest_main.py"),
            args,
            log_label=f"Open Yap ingest shard {self.shard_idx}",
            with_hf_home=True,
            extra_env={"PYTHONPATH": _moshi_pythonpath()},
            env_hook=env_hook,
        )


def openyap_episode_codes(
    *,
    mimi_venv_python,
    archive: tk.Path,
    shards_dir: tk.Path,
    shard_indices: list[int],
    tag: str,
    register: bool = True,
    **ingest_kwargs,
):
    """``OpenYapEpisodeIngest`` per shard -> ``PodcastCodesIndex`` over them: ``(codes_index_dir, shard_dirs)``, the
    shape ``track_podcast.build_podcast`` hands downstream (outputs ``podcast_episodes/<tag>/codes_index``)."""
    from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import PodcastCodesIndex

    shard_dirs = [
        OpenYapEpisodeIngest(
            mimi_venv_python=mimi_venv_python,
            archive=archive,
            shards_dir=shards_dir,
            shard_idx=k,
            env_ffmpeg_path=InstallFFmpeg().out_path,
            **ingest_kwargs,
        ).out_dir
        for k in shard_indices
    ]
    index = PodcastCodesIndex(in_dirs=shard_dirs, require_all_shards=True)
    if register:
        tk.register_output(f"podcast_episodes/{tag}/codes_index", index.out_dir)
    return index.out_dir, shard_dirs
