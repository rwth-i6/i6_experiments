"""Podcast -> Mimi-codes ingestion: one normalised, sharded pipeline for labelled and unlabelled audio.

THE SHAPE. Three jobs, and the first one is what makes this general:

    PodcastWorkIndex   (mini_task, login node)  a source -> N balanced shard files of WORK ITEMS
    PodcastMimiIngest  (GPU, one per shard)     download + encode, concurrently, in one pass
    MergePodcastCodes  (mini_task)              shard parts -> one arrow dataset

Everything corpus-specific is confined to the index job, which lowers any source to the same work
item::

    {item_id, audio_url, source, spans | null, expected_duration_sec | null, feed, meta}

``spans = null`` means "the whole episode" (an unlabelled podcast: JRE). A span list means the source
already knows where the two-speaker dialogue is (DuplexChat, whose manifest carries
``episode_start_sec``/``episode_end_sec`` per dialogue). That one field is the entire difference
between the two corpora at this layer, which is why they can share a worker.

WHY DOWNLOAD AND ENCODE ARE ONE JOB. The staged shape -- download job writes waveforms, encode job
reads them, as ``seamless_jobs.py`` does -- cannot be used at the scale this exists for. DuplexChat
English is 282,634 h: ~16 TB as source mp3, ~49 TB as 24 kHz FLAC, against ~8 TB free on
``/hpcwork/p0023999`` (42 T of 50 T, 96% of its inode quota). The same audio as Mimi codes is
**1.44 MB/h => ~407 GB**. The intermediate waveform is the part that does not fit, so it is never
written; see the worker's docstring for the concurrency and the memory bound.

SHARDING, and why it is not a stride. Two properties are load-bearing:

1. **Group by episode.** A DuplexChat episode contributes many dialogue rows that all share one
   ``audio_url``. A naive ``i % num_shards`` would scatter them and re-download the same ~120 MB file
   in every shard. Items are grouped by ``audio_url`` and a group is never split.
2. **Balance by SECONDS, not item count.** Episodes vary more than 10x in length, so equal counts
   give wildly unequal runtimes. Groups are bin-packed longest-first into the lightest shard, which
   is what keeps "each job finishes in a reasonable time" true rather than aspirational.

``num_shards`` has to be fixed at graph-build time (it is the number of downstream jobs), so the
recipe helpers derive it from an estimated total and a target hours-per-shard instead of discovering
it at run time.

COST MODEL, measured, for sizing a shard. Mimi encode is ~2,119x realtime (28.32 ms per 60 s stereo
window, ``MimiAugmentationProbe`` code_version 3) => ~588 audio-h per GPU-h. Diarization is RTF
~0.007-0.010, i.e. ~100x realtime => ~100 audio-h per GPU-h. **So for an unlabelled podcast the
diarizer dominates the GPU cost by ~6x, not the codec.** At the default 60 audio-h per shard that is
~0.6 GPU-h of diarization plus ~0.1 of Mimi, and a few GB of download -- around an hour per job.
"""

from __future__ import annotations

import json
import os

from sisyphus import Job, Task, tk

from i6_experiments.users.dorian_koch.speech_llm.common import (
    job_progress_fraction,
    run_worker_script,
)
from i6_experiments.users.dorian_koch.speech_llm.tts import InstallFFmpeg

CHANNEL_MODES = ("diarize_mask", "stereo_passthrough", "mono_both", "dialogue_sidon")


def _moshi_family_lib_parent() -> str:
    """Absolute dir to put on PYTHONPATH so ``import moshi_family...`` resolves in a job venv.

    Walks up from this file to the recipe root rather than resolving symlinks: ``recipe/...`` paths
    are symlinks into the source repos, and resolving them lands outside the tree the worker expects.
    This is the same unresolved walk ``finetune.py`` uses; getting it wrong is a
    ``ModuleNotFoundError`` five seconds into a job that has already been allocated a GPU.
    """
    from pathlib import Path

    root = next((str(p) for p in Path(__file__).parents if (p / "i6_experiments").exists()), None)
    if not root:
        raise RuntimeError("could not locate the recipe root from podcast_ingest.py")
    return os.path.join(root, "speech_llm", "full_duplex")


class PodcastWorkIndex(Job):
    """Normalise a podcast source into ``num_shards`` episode-grouped, duration-balanced shards."""

    __sis_hash_exclude__ = {"rqmt": None, "user_agent": None}

    def __init__(
        self,
        *,
        source: str,
        num_shards: int,
        rss_urls: list[str] | None = None,
        manifest: tk.Path | None = None,
        max_episodes: int = 0,
        min_span_sec: float = 10.0,
        max_span_sec: float = 600.0,
        feed_allowlist: list[str] | None = None,
        default_episode_sec: float = 3600.0,
        rqmt: dict | None = None,
        user_agent: str | None = None,
    ):
        assert source in ("rss", "duplexchat"), f"unknown source {source!r}"
        if source == "rss":
            assert rss_urls, "source='rss' needs rss_urls"
        else:
            assert manifest is not None, "source='duplexchat' needs manifest"
        self.source = source
        self.num_shards = int(num_shards)
        self.rss_urls = list(rss_urls or [])
        self.manifest = manifest
        self.max_episodes = int(max_episodes)
        self.min_span_sec = float(min_span_sec)
        self.max_span_sec = float(max_span_sec)
        self.feed_allowlist = list(feed_allowlist or [])
        self.default_episode_sec = float(default_episode_sec)
        self.user_agent = user_agent
        self.out_dir = self.output_path("index", directory=True)
        self.rqmt = rqmt or {"cpu": 2, "mem": 8, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    # -- source adapters ---------------------------------------------------------------------
    def _from_rss(self) -> list[dict]:
        """Fetch each feed and lower its episodes to work items with ``spans=None``."""
        import urllib.request
        import xml.etree.ElementTree as ET

        ua = self.user_agent or "Mozilla/5.0 (compatible; i6-speechllm-research/1.0)"
        ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
        items: list[dict] = []
        for url in self.rss_urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": ua})
                with urllib.request.urlopen(req, timeout=120) as r:
                    xml = r.read()
                root = ET.fromstring(xml)
            except Exception as e:  # noqa: BLE001
                print(f"[index] FEED FAILED {url}: {type(e).__name__}: {e}", flush=True)
                continue
            chan = root.find("channel")
            feed_title = (chan.findtext("title") or url).strip() if chan is not None else url
            cats = []
            if chan is not None:
                for c in chan.findall("itunes:category", ns):
                    if c.get("text"):
                        cats.append(c.get("text"))
            n_feed = 0
            for ep in chan.findall("item") if chan is not None else []:
                enc = ep.find("enclosure")
                if enc is None or not enc.get("url"):
                    continue
                # itunes:duration is "SS", "MM:SS" or "HH:MM:SS" -- and is often absent or a lie,
                # so it is only ever used for BALANCING, never for the drift check.
                raw = (ep.findtext("itunes:duration", default="", namespaces=ns) or "").strip()
                est = None
                if raw:
                    try:
                        bits = [float(b) for b in raw.split(":")]
                        est = bits[-1]
                        if len(bits) >= 2:
                            est += bits[-2] * 60
                        if len(bits) >= 3:
                            est += bits[-3] * 3600
                    except ValueError:
                        est = None
                guid = (ep.findtext("guid") or enc.get("url")).strip()
                items.append(
                    {
                        "item_id": f"rss_{abs(hash(guid)) % (10**16):016d}",
                        "audio_url": enc.get("url"),
                        "source": "rss",
                        "spans": None,
                        "expected_duration_sec": None,  # RSS duration is not trustworthy; see above
                        "est_seconds": est or self.default_episode_sec,
                        "feed": feed_title,
                        "meta": {
                            "guid": guid,
                            "title": (ep.findtext("title") or "").strip(),
                            "rss_url": url,
                            "itunes_categories": cats,
                            "itunes_duration_raw": raw,
                        },
                    }
                )
                n_feed += 1
            print(f"[index] {feed_title}: {n_feed} episodes", flush=True)
        return items

    def _from_duplexchat(self) -> list[dict]:
        """Group DuplexChat manifest rows into one work item per episode, carrying its spans."""
        import gzip

        path = self.manifest.get()
        opener = gzip.open if path.endswith(".gz") else open
        by_ep: dict[str, dict] = {}
        n_rows = 0
        n_kept = 0
        with opener(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_rows += 1
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self.feed_allowlist and r.get("rss_url") not in self.feed_allowlist:
                    continue
                dur = float(r.get("duration_sec") or 0.0)
                if not (self.min_span_sec <= dur <= self.max_span_sec):
                    continue
                url = r.get("audio_url")
                if not url:
                    continue
                ep = by_ep.setdefault(
                    url,
                    {
                        "item_id": f"dc_{abs(hash(url)) % (10**16):016d}",
                        "audio_url": url,
                        "source": "duplexchat",
                        "spans": [],
                        # The manifest's own episode duration IS the drift reference -- this is the
                        # field that makes the mandatory drift check possible for this corpus.
                        "expected_duration_sec": float(r.get("episode_duration_sec") or 0.0) or None,
                        "est_seconds": 0.0,
                        "feed": r.get("rss_url", ""),
                        "meta": {"rss_url": r.get("rss_url", ""), "language": r.get("language", "")},
                    },
                )
                ep["spans"].append([float(r["episode_start_sec"]), float(r["episode_end_sec"])])
                ep["est_seconds"] += dur
                n_kept += 1
                if self.max_episodes and len(by_ep) > self.max_episodes:
                    by_ep.pop(url, None)
                    break
        print(f"[index] duplexchat: {n_rows} rows -> {n_kept} spans in {len(by_ep)} episodes", flush=True)
        return list(by_ep.values())

    def run(self):
        items = self._from_rss() if self.source == "rss" else self._from_duplexchat()
        if self.max_episodes:
            items = items[: self.max_episodes]
        if not items:
            raise SystemExit("index produced ZERO work items -- refusing to write empty shards")

        # Deterministic order before packing, so the same source always yields the same shards.
        items.sort(key=lambda it: it["item_id"])

        # Greedy longest-first bin packing by estimated seconds. Equal-count sharding would leave
        # one shard with the long episodes and finish hours after the rest.
        loads = [0.0] * self.num_shards
        shards: list[list[dict]] = [[] for _ in range(self.num_shards)]
        for it in sorted(items, key=lambda x: -float(x.get("est_seconds") or 0.0)):
            k = min(range(self.num_shards), key=lambda i: loads[i])
            shards[k].append(it)
            loads[k] += float(it.get("est_seconds") or 0.0)

        for k, sh in enumerate(shards):
            with open(os.path.join(self.out_dir.get(), f"shard_{k:05d}.jsonl"), "w") as f:
                for it in sorted(sh, key=lambda x: x["item_id"]):
                    f.write(json.dumps(it) + "\n")

        info = {
            "source": self.source,
            "num_shards": self.num_shards,
            "n_items": len(items),
            "total_est_hours": round(sum(loads) / 3600.0, 2),
            "per_shard_items": [len(s) for s in shards],
            "per_shard_est_hours": [round(v / 3600.0, 2) for v in loads],
        }
        with open(os.path.join(self.out_dir.get(), "index.json"), "w") as f:
            json.dump(info, f, indent=1)
        print("[index] " + json.dumps(info), flush=True)


class PodcastMimiIngest(Job):
    """Download + Mimi-encode one shard, concurrently, without ever storing the waveform."""

    __sis_hash_exclude__ = {"rqmt": None, "keep_audio": False, "max_items": 0}

    def __init__(
        self,
        *,
        venv_python_path,
        index_dir: tk.Path,
        shard_idx: int,
        hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        channel_mode: str = "diarize_mask",
        diarization_model: str = "pyannote/speaker-diarization-community-1",
        download_workers: int = 8,
        chunk_sec: float = 30.0,
        drift_tolerance_sec: float = 5.0,
        max_failure_frac: float = 0.25,
        keep_audio: bool = False,
        max_items: int = 0,
        code_version: int = 1,
        env_ffmpeg_path: tk.Path | None = None,
        rqmt: dict | None = None,
    ):
        assert channel_mode in CHANNEL_MODES, f"channel_mode must be one of {CHANNEL_MODES}"
        self.venv_python_path = venv_python_path
        self.index_dir = index_dir
        self.shard_idx = int(shard_idx)
        self.hf_repo = hf_repo
        self.channel_mode = channel_mode
        self.diarization_model = diarization_model
        self.download_workers = int(download_workers)
        self.chunk_sec = float(chunk_sec)
        self.drift_tolerance_sec = float(drift_tolerance_sec)
        self.max_failure_frac = float(max_failure_frac)
        self.keep_audio = bool(keep_audio)
        self.max_items = int(max_items)
        self.code_version = int(code_version)
        self.env_ffmpeg_path = env_ffmpeg_path
        self.out_dir = self.output_path("codes", directory=True)
        # `time` is generous on purpose: a shard is network bound and a slow CDN cannot be retried
        # around. `run` is resumable, so a wall-clock kill costs only the episodes in flight.
        self.rqmt = rqmt or {"gpu": 1, "cpu": 8, "mem": 48, "time": 8}

    @classmethod
    def hash(cls, parsed_args):
        # env_ffmpeg_path is WHERE our ffmpeg happens to live, not WHAT this job computes. If it
        # reached the hash, supplying it would re-run every already-ingested shard -- the same
        # reasoning (and the same guard) as knowledge_benchmark's env_ffmpeg_path.
        d = dict(parsed_args)
        d.pop("env_ffmpeg_path", None)
        return super().hash(d)

    def tasks(self):
        # resume=: a shard is hours of work and rebuilds `done` from its own flushed parts, so
        # Sisyphus reschedules an interrupted shard on its own. Per CLAUDE.md, such a task must NOT
        # be cleared with `hpc-rerun --include-interrupted`.
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        try:
            with open(os.path.join(self.out_dir.get(), "progress.json")) as f:
                p = json.load(f)
            return f"{p.get('ok', 0)} ok / {p.get('failed', 0)} failed, {p.get('audio_hours', 0)} audio-h"
        except Exception:  # noqa: BLE001
            return None

    def run(self):
        lib_parent = _moshi_family_lib_parent()
        script = os.path.join(lib_parent, "moshi_family", "podcast_ingest_main.py")
        shard = os.path.join(self.index_dir.get(), f"shard_{self.shard_idx:05d}.jsonl")
        if not os.path.exists(shard):
            raise FileNotFoundError(
                f"{shard} does not exist. shard_idx {self.shard_idx} is outside the num_shards the "
                "index was built with -- the two are set independently and must agree."
            )

        env: dict[str, str] = {"PYTHONPATH": lib_parent}
        ffmpeg_dir = ""
        if self.env_ffmpeg_path is not None:
            InstallFFmpeg.add_to_env(self.env_ffmpeg_path, env)
            ffmpeg_dir = self.env_ffmpeg_path.get()
        else:
            raise ValueError(
                "env_ffmpeg_path is required: decoding mp3/m4a and resampling to 24 kHz goes "
                "through OUR ffmpeg, never a system one. Pass env_ffmpeg_path=InstallFFmpeg()"
                ".out_path (hash-free)."
            )

        args = [
            "--shard_jsonl",
            shard,
            "--out_dir",
            self.out_dir.get(),
            "--hf_repo",
            self.hf_repo,
            "--channel_mode",
            self.channel_mode,
            "--diarization_model",
            self.diarization_model,
            "--ffmpeg_dir",
            ffmpeg_dir,
            "--download_workers",
            self.download_workers,
            "--chunk_sec",
            self.chunk_sec,
            "--drift_tolerance_sec",
            self.drift_tolerance_sec,
            "--max_failure_frac",
            self.max_failure_frac,
        ]
        if self.keep_audio:
            args.append("--keep_audio")
        if self.max_items:
            args += ["--max_items", self.max_items]

        run_worker_script(
            self.venv_python_path.get(),
            script,
            args,
            log_label=f"Podcast mimi ingest shard {self.shard_idx}",
            with_hf_home=True,
            extra_env=env,
        )


class MergePodcastCodes(Job):
    """Shard parts (jsonl + npz) -> ONE arrow dataset of Mimi codes."""

    def __init__(self, *, in_dirs: list[tk.Path], rqmt: dict | None = None):
        self.in_dirs = list(in_dirs)
        self.out_dir = self.output_path("codes_dataset", directory=True)
        self.rqmt = rqmt or {"cpu": 4, "mem": 32, "time": 4}

    def tasks(self):
        yield Task("merge", rqmt=self.rqmt)

    def merge(self):
        import numpy as np
        from datasets import Dataset

        rows: list[dict] = []
        n_shards_ok = 0
        for d in self.in_dirs:
            parts = os.path.join(d.get(), "parts")
            if not os.path.isdir(parts):
                print(f"[merge] SKIP {d.get()}: no parts/", flush=True)
                continue
            n_shards_ok += 1
            for name in sorted(os.listdir(parts)):
                if not name.endswith(".jsonl"):
                    continue
                meta_path = os.path.join(parts, name)
                npz_path = meta_path[: -len(".jsonl")] + ".npz"
                if not os.path.exists(npz_path):
                    print(f"[merge] SKIP {name}: no matching npz", flush=True)
                    continue
                with np.load(npz_path) as z, open(meta_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        c = z[r["item_id"]]  # [2, K, F] int16
                        # Flattened per channel, with n_codebooks/n_frames alongside, because a
                        # 3-D nested arrow list is awkward to read back and easy to get subtly
                        # wrong. A consumer reshapes to (n_codebooks, n_frames).
                        r["codes_a"] = c[0].reshape(-1).astype("int16").tolist()
                        r["codes_b"] = c[1].reshape(-1).astype("int16").tolist()
                        rows.append(r)

        if not rows:
            raise SystemExit(
                "merged ZERO rows. Every shard either failed or wrote no parts -- read each "
                "shard's summary.json / failures.jsonl rather than treating this as an empty corpus."
            )
        total_h = sum(r.get("duration_sec", 0.0) for r in rows) / 3600.0
        print(f"[merge] {len(rows)} rows from {n_shards_ok} shards, {total_h:.1f} audio-hours", flush=True)
        Dataset.from_list(rows).save_to_disk(self.out_dir.get())
        with open(os.path.join(self.out_dir.get(), "merge_summary.json"), "w") as f:
            json.dump({"n_rows": len(rows), "n_shards": n_shards_ok, "audio_hours": round(total_h, 2)}, f, indent=1)


# -------------------------------------------------------------------------------------------
# recipe helpers
# -------------------------------------------------------------------------------------------
def shards_for_hours(est_total_hours: float, target_hours_per_shard: float = 60.0, *, cap: int = 256) -> int:
    """num_shards from an estimated total, so a shard lands near ``target_hours_per_shard``.

    Must be decided at graph-build time -- it is the number of downstream jobs. 60 h/shard is
    ~0.6 GPU-h of diarization + ~0.1 of Mimi + a few GB of download, i.e. about an hour.
    """
    import math

    return max(1, min(cap, int(math.ceil(float(est_total_hours) / max(1e-9, target_hours_per_shard)))))


def podcast_codes(
    *,
    venv_python_path,
    tag: str,
    source: str,
    num_shards: int,
    rss_urls: list[str] | None = None,
    manifest: tk.Path | None = None,
    channel_mode: str = "diarize_mask",
    max_episodes: int = 0,
    register: bool = True,
    **ingest_kwargs,
):
    """Wire index -> N sharded ingests -> merge. Returns ``(merged_dir, index_dir, shard_dirs)``."""
    index = PodcastWorkIndex(
        source=source,
        num_shards=num_shards,
        rss_urls=rss_urls,
        manifest=manifest,
        max_episodes=max_episodes,
    )
    shard_dirs = []
    for k in range(num_shards):
        job = PodcastMimiIngest(
            venv_python_path=venv_python_path,
            index_dir=index.out_dir,
            shard_idx=k,
            channel_mode=channel_mode,
            env_ffmpeg_path=InstallFFmpeg().out_path,
            **ingest_kwargs,
        )
        shard_dirs.append(job.out_dir)
    merged = MergePodcastCodes(in_dirs=shard_dirs)
    if register:
        tk.register_output(f"podcast_codes/{tag}/index", index.out_dir)
        tk.register_output(f"podcast_codes/{tag}/dataset", merged.out_dir)
    return merged.out_dir, index.out_dir, shard_dirs
