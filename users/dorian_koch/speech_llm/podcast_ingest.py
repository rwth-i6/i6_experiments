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

COST MODEL, measured, for sizing a shard -- see the constants near the bottom of this file. Per
audio-hour of main-thread work: Mimi **1.70 s** (28.32 ms per 60 s stereo window, batch 2, from
``MimiAugmentationProbe`` code_version 3), diarization **36 s** (pyannote RTF 0.007-0.010), ffmpeg
decode **14.4 s**. **So the diarizer dominates Mimi by ~21x** and it, not the codec, sets the shard
size for unlabelled audio. Targeting 4 h per job gives ~**277 audio-h per shard** with
``diarize_mask`` and ~**900** with ``stereo_passthrough``; ``shards_for_hours()`` does this
arithmetic and refuses to silently exceed a cap.

⚠ Network is deliberately excluded from that model (as instructed), so it is an upper bound on
throughput: if downloads are slower than compute, a shard overruns and the resumable job continues
after a walltime reschedule rather than losing work.
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

# 🔴 `diarize_mask` is REJECTED for production duplex data (user, by ear, 2026-09-19).
# It is a per-sample gate on the mono mix, not separation: in overlap BOTH channels receive the
# same mixed audio, and even outside overlap the other speaker bleeds through because the channel
# IS the mix. The user compared all four candidates on two windows (55.6% and 19.9% overlap) and
# ranked DialogueSidon first "by far", SepFormer "useless", and gating "isn't the right tool for
# us here". ⚠ Our SI-SDR pilot ranked gating FIRST (+6.19 dB) -- the metric rewards gating on the
# non-overlapped majority and cannot score a resynthesis at all, so it is a domain-gap measurement
# and not a ranking. See `projects/2026-01-speech-llm/audio_datasets.md`.
#
# So for a mono source that needs two speakers, the mode is `dialogue_sidon` via
# `PodcastDuplexIngest`. `diarize_mask` remains only as the cheap baseline the comparison was
# made against. `stereo_passthrough` is unaffected and is the right mode for sources that are
# already dual-channel (Open Yap, DuplexChat reconstructions) -- nothing is separated there.
REJECTED_CHANNEL_MODES = ("diarize_mask",)

#: Episodes burned on smoke tests. `podcast_codes()` excludes these from every real corpus by
#: default, so test audio can never end up inside training data -- a guarantee in code rather than a
#: note somebody has to remember. Add to this list whenever an episode is used for a test.
SMOKE_AUDIO_URLS = [
    # 74.5 s episode, pinecast/colour-out-the-box; chosen as the shortest DuplexChat episode that
    # still carries a 20-90 s dialogue span, so the download is ~1 MB.
    "https://pinecast.com/listen/70469672-3d59-42c0-a2ba-121d97b1a1dc:"
    "92ff53eb-8adc-47ee-8dbd-04fb480995f8.mp3?source=rss&ext=asset.mp3",
]


def _moshi_pythonpath() -> str:
    """PYTHONPATH for a worker that imports both ``moshi_family`` and ``moshi``.

    🔴 TWO entries are required and only one is obvious. ``speech_llm/full_duplex`` provides
    ``moshi_family``; the **recipe root** provides ``moshi`` itself (kyutai's package). A worker
    given only the first dies on ``from moshi.models import loaders`` -- a ModuleNotFoundError
    about five seconds in, *after* the GPU has been allocated. That is exactly how the first
    podcast smoke test failed, and ``mimi_aug_probe`` had already been fixed the same way; the
    lesson simply had not been carried across.
    """
    from pathlib import Path

    root = next((str(p) for p in Path(__file__).parents if (p / "i6_experiments").exists()), None)
    if not root:
        raise RuntimeError("could not locate the recipe root from podcast_ingest.py")
    parts = []
    lib_parent = os.path.join(root, "speech_llm", "full_duplex")
    if os.path.isdir(lib_parent):
        parts.append(lib_parent)
    parts.append(root)
    return os.pathsep.join(parts)


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
        exclude_audio_urls: list[str] | None = None,
        overlapping_spans: str = "keep",
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
        assert overlapping_spans in ("keep", "merge", "raise"), overlapping_spans
        self.exclude_audio_urls = list(exclude_audio_urls or [])
        self.overlapping_spans = overlapping_spans
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
        # 🔴 DuplexChat spans OVERLAP, measured: the smoke episode has 35.79->50.15 and
        # 45.53->63.06, sharing 4.6 s. Ingesting every span therefore encodes some seconds TWICE
        # and the corpus silently contains duplicated audio -- the twin of silently losing it, and
        # just as invisible downstream. So it is measured here and acted on by an explicit policy,
        # never ignored. (Their pipeline treats each span as an independent dialogue clip, so the
        # overlap is by design on their side, not corruption.)
        dup_sec = 0.0
        n_overlapping = 0
        for ep in by_ep.values():
            sp = sorted(ep["spans"])
            merged: list[list[float]] = []
            had = False
            for b, e in sp:
                if merged and b < merged[-1][1]:
                    had = True
                    dup_sec += min(e, merged[-1][1]) - b
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([b, e])
            n_overlapping += int(had)
            if self.overlapping_spans == "merge":
                ep["spans"] = merged
                ep["est_seconds"] = sum(e - b for b, e in merged)
        msg = (
            f"[index] duplexchat: {n_rows} rows -> {n_kept} spans in {len(by_ep)} episodes; "
            f"{n_overlapping} episodes have overlapping spans, {dup_sec / 3600:.2f} h duplicated"
        )
        print(msg, flush=True)
        if self.overlapping_spans == "raise" and n_overlapping:
            raise SystemExit(msg + " -- overlapping_spans='raise'")
        if self.overlapping_spans == "merge":
            print("[index] overlapping spans MERGED (a merged span may hold >2 speakers)", flush=True)
        return list(by_ep.values())

    def run(self):
        items = self._from_rss() if self.source == "rss" else self._from_duplexchat()

        # Drop anything burned on a smoke test. Enforced HERE, at the one place every source passes
        # through, so no corpus can pick up test audio by omission -- and reported, so the exclusion
        # is visible in the log rather than being an invisible subtraction.
        if self.exclude_audio_urls:
            drop = set(self.exclude_audio_urls)
            before = len(items)
            items = [it for it in items if it["audio_url"] not in drop]
            print(
                f"[index] excluded {before - len(items)} item(s) via exclude_audio_urls ({len(drop)} url(s) listed)",
                flush=True,
            )

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
        span_mode: str = "whole",
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
        assert span_mode in ("whole", "union", "per_span"), span_mode
        self.span_mode = span_mode
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

        # BOTH paths: lib_parent for `moshi_family`, the recipe root for `moshi` itself.
        env: dict[str, str] = {"PYTHONPATH": _moshi_pythonpath()}
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
            "--span_mode",
            self.span_mode,
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


class PodcastCodesIndex(Job):
    """Index the shard parts in place -- metadata only, **no copy of the codes**.

    This replaced a merge job that read every part and rewrote the codes into one arrow dataset.
    That was a second full copy of the corpus: ~407 GB for DuplexChat English, on the volume whose
    shortage is the entire reason the pipeline stores codes at all. The codes are already in their
    final form and location after ingest, so the only thing actually missing is a single place that
    says what exists and where -- which is small (a few hundred bytes per row) and cheap to rebuild.

    Arrow is **memory-mapped**, so a loader opens a part and reads one row's codes without pulling
    the rest into RAM. That is what makes indexing-in-place viable rather than merging -- and it is
    why the corpus handle is simply the LIST of arrow parts, which is already the ``[(path, weight)]``
    mix shape the trainer takes.

    It is also where the **completeness** checks live, because this is the first point that sees
    every shard at once. All three failures it catches are silent:
      * a row whose declared shape disagrees with the arrow array beside it,
      * the same ``item_id`` in two shards (one would overwrite the other at load time),
      * a shard that produced no parts at all.
    """

    def __init__(self, *, in_dirs: list[tk.Path], require_all_shards: bool = True, rqmt: dict | None = None):
        self.in_dirs = list(in_dirs)
        self.require_all_shards = bool(require_all_shards)
        self.out_dir = self.output_path("codes_index", directory=True)
        self.rqmt = rqmt or {"cpu": 2, "mem": 8, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from datasets import load_from_disk

        rows: list[dict] = []
        seen: dict[str, str] = {}
        empty_shards: list[str] = []
        n_shards_ok = 0
        total_sec = 0.0

        for d in self.in_dirs:
            base = d.get()
            parts = os.path.join(base, "parts")
            names = (
                sorted(n for n in os.listdir(parts) if os.path.isdir(os.path.join(parts, n)) and not n.endswith(".tmp"))
                if os.path.isdir(parts)
                else []
            )
            if not names:
                empty_shards.append(base)
                continue
            n_shards_ok += 1
            for name in names:
                part = os.path.join(parts, name)
                ds = load_from_disk(part)  # memory-mapped; the codes are not read here
                # Only the small columns are pulled. Reading `codes_a` would defeat the entire
                # point of indexing in place -- it would stream the whole corpus through this job.
                cols = {k: ds[k] for k in ("item_id", "episode_id", "duration_sec", "n_codebooks", "n_frames")}
                for i in range(ds.num_rows):
                    iid = cols["item_id"][i]
                    if iid in seen:
                        raise SystemExit(
                            f"duplicate item_id {iid} in {part} and {seen[iid]}. One would "
                            "silently shadow the other at load time."
                        )
                    seen[iid] = part
                    total_sec += float(cols["duration_sec"][i] or 0.0)
                    rows.append(
                        {
                            "item_id": iid,
                            "episode_id": cols["episode_id"][i],
                            "duration_sec": float(cols["duration_sec"][i] or 0.0),
                            "n_codebooks": int(cols["n_codebooks"][i]),
                            "n_frames": int(cols["n_frames"][i]),
                            # Where the row actually lives. The corpus is the LIST of these arrow
                            # parts -- exactly the `[(path, weight)]` mix shape the trainer already
                            # takes -- so nothing is ever copied.
                            "part": part,
                            "row": i,
                        }
                    )

        if empty_shards:
            msg = f"{len(empty_shards)} of {len(self.in_dirs)} shards produced no parts"
            if self.require_all_shards:
                raise SystemExit(
                    msg + ". Read their summary.json / failures.jsonl. Indexing anyway would "
                    "produce a corpus that is quietly missing whole shards -- pass "
                    "require_all_shards=False only once you know why."
                )
            print(f"[index] WARNING: {msg}", flush=True)

        if not rows:
            raise SystemExit("indexed ZERO rows -- refusing to present this as a corpus.")

        out = os.path.join(self.out_dir.get(), "index.jsonl")
        with open(out, "w") as f:
            for r in sorted(rows, key=lambda x: x["item_id"]):
                f.write(json.dumps(r) + "\n")
        # The corpus handle: the distinct arrow parts, in order. This is what a trainer consumes as
        # a `[(path, weight)]` mix, so "merging" is a list of paths and never a copy of the data.
        part_paths = sorted({r["part"] for r in rows})
        with open(os.path.join(self.out_dir.get(), "parts.json"), "w") as f:
            json.dump(part_paths, f, indent=1)

        summary = {
            "n_rows": len(rows),
            "n_parts": len(part_paths),
            "n_shards_indexed": n_shards_ok,
            "n_shards_empty": len(empty_shards),
            "audio_hours": round(total_sec / 3600.0, 3),
            # 2 channels x K x F x int16
            "bytes_of_codes": sum(r["n_codebooks"] * r["n_frames"] * 2 * 2 for r in rows),
        }
        with open(os.path.join(self.out_dir.get(), "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print("[index] " + json.dumps(summary), flush=True)


# -------------------------------------------------------------------------------------------
# recipe helpers
# -------------------------------------------------------------------------------------------
# ---- measured per-audio-hour cost of the main thread, in SECONDS ---------------------------
# Deliberately a table of measured numbers rather than one fudge factor, because which term
# dominates changes the answer by 3x and depends on channel_mode.
#
# MIMI_SEC_PER_AUDIO_HOUR: 28.32 ms per 60 s stereo window (MimiAugmentationProbe, code_version 3,
#   batch 2 = the two channels, which is what training calls) => 60 windows per audio-hour
#   => 1.70 s. That is 2,119x realtime, i.e. **2,119 audio-hours per GPU-hour**.
#   ⚠ An earlier note of mine said "588 audio-h per GPU-h" and derived 481 GPU-h for DuplexChat.
#   That was a unit slip: a realtime factor and audio-hours-per-GPU-hour are the same dimensionless
#   ratio, so both are 2,119 and the DuplexChat encode is ~133 GPU-h, not 481.
# DIARIZE_SEC_PER_AUDIO_HOUR: pyannote RTF 0.007-0.010 (measured on the JRE episodes) => 36 s at the
#   pessimistic end. This DOMINATES Mimi by ~21x, so for unlabelled podcasts the diarizer sets the
#   shard size and the codec is a rounding error.
# DECODE_SEC_PER_AUDIO_HOUR: our ffmpeg, mp3 -> 24 kHz f32. Audio-only decode runs a few hundred x
#   realtime; 250x => 14.4 s. It is on the MAIN thread (the pool hands over file paths, not arrays,
#   to keep memory at one episode), so it counts.
MIMI_SEC_PER_AUDIO_HOUR = 1.70
DIARIZE_SEC_PER_AUDIO_HOUR = 36.0
DECODE_SEC_PER_AUDIO_HOUR = 14.4

# ---- MEASURED on a real shard, 2026-09-18 (JRE #2553, 2.68 episode-hours) --------------------
# From `summary.json`'s `sec_per_episode_hour`, which exists so the model is checked rather than
# trusted. These are per EPISODE-hour (not per retained hour), because that is what a shard is
# sized in: you pay diarization on the whole episode and separation only on what survives.
#
#   decode    2.71   predicted 14.4  -> our ffmpeg is ~5x faster than assumed
#   diarize  34.50   predicted 36.0  -> good prediction; still the bottleneck
#   separate 28.14   predicted 14.4  -> ~2x MORE than assumed, see below
#   encode    4.26   predicted  1.70 -> includes one subprocess start + mimi load per BATCH
#             -----
#            69.61   total, vs ~65 predicted -- the total was close, the breakdown was not.
#
# ⚠ The separation figure is the one to be careful with. The 6-clip run measured RTF 0.01 (~100x
# realtime) on ~35 s clips; here it came out at 65.9 s per RETAINED hour, i.e. **RTF 0.018, ~55x
# realtime**. Real dialogues average ~171 s and the 120 s / 10 s overlapping chunking adds ~9% on
# long spans, so the short-clip number was optimistic. Use the measured one.
# ⚠ The encode figure carries fixed cost: one mimi model load per batch, amortised over
# `batch_episodes`. At batch_episodes=1 (the smoke test) it is ~10 s per retained hour; it tends
# toward MIMI_SEC_PER_AUDIO_HOUR as batches grow.
DUPLEX_SEC_PER_EPISODE_HOUR = 69.6
#: Fraction of an episode that survives `extract_valid_dialogues`. Measured 0.427 on JRE #2553,
#: reproducing the independent `jre_yield` run exactly (42.7%).
DUPLEX_RETENTION = 0.427


def duplex_episode_hours_per_shard(target_runtime_hours: float = 4.0) -> float:
    """Episode-hours one SEPARATING shard can do in ``target_runtime_hours``, from measured cost.

    ~207 episode-hours at the 4 h default, so JRE's ~5,500 h is ~27 shards. Note this is
    episode-hours in, not corpus-hours out: at 42.7% retention a shard yields ~88 h of dialogue.
    """
    return (float(target_runtime_hours) * 3600.0) / DUPLEX_SEC_PER_EPISODE_HOUR


def audio_hours_per_shard(channel_mode: str, target_runtime_hours: float = 4.0) -> float:
    """How many audio-hours one shard can do in ``target_runtime_hours``, GPU/CPU bound.

    Network is deliberately EXCLUDED from this model, per the user's instruction to size as if we
    are not network bound. That makes the number an upper bound on throughput: if the download is
    slower than the compute, shards take longer than the target and the (resumable) job simply
    continues after a walltime reschedule.
    """
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(f"unknown channel_mode {channel_mode!r}; expected one of {CHANNEL_MODES}")
    if channel_mode == "dialogue_sidon":
        # 🔴 The separating path is NOT the sum of the per-stage constants above. It is measured
        # end to end (decode + diarize + separate + encode = 69.6 s/episode-hour) and it is the
        # only mode whose cost was checked against a real shard rather than predicted. Delegating
        # is the point: deriving it from the constants here yields the GATING number, which is
        # ~33% too optimistic (276 vs 207 input-hours per shard), so a duplex corpus sized that
        # way runs ~5.3 h per job against a 4 h target -- under the 8 h walltime, so it would
        # never fail, just quietly miss the target the sharding exists to hit.
        return duplex_episode_hours_per_shard(target_runtime_hours)
    per_hour = MIMI_SEC_PER_AUDIO_HOUR + DECODE_SEC_PER_AUDIO_HOUR
    if channel_mode == "diarize_mask":
        per_hour += DIARIZE_SEC_PER_AUDIO_HOUR
    return (float(target_runtime_hours) * 3600.0) / per_hour


def shards_for_hours(
    est_total_hours: float,
    *,
    channel_mode: str,
    target_runtime_hours: float = 4.0,
    max_shards: int = 0,
) -> int:
    """num_shards such that each shard runs ~``target_runtime_hours``.

    Must be decided at graph-build time -- it is the number of downstream jobs.

    ⚠ There is NO silent cap. An earlier version capped at 256, which is actively harmful: past the
    cap it does not reduce the number of shards, it silently makes each one bigger than the target,
    so "every job finishes in about 4 h" quietly becomes "some job runs for two days and gets
    killed". If a cap is wanted it must be passed explicitly, and exceeding it RAISES with the
    arithmetic rather than rounding the problem away.
    """
    import math

    per_shard = audio_hours_per_shard(channel_mode, target_runtime_hours)
    n = max(1, int(math.ceil(float(est_total_hours) / per_shard)))
    if max_shards and n > max_shards:
        raise ValueError(
            f"{est_total_hours:.0f} audio-hours at channel_mode={channel_mode!r} needs {n} shards "
            f"to hold each job near {target_runtime_hours} h ({per_shard:.0f} audio-h per shard), "
            f"but max_shards={max_shards}. Raise max_shards, raise target_runtime_hours, or ingest "
            "a subset -- silently using fewer, longer shards would blow the walltime."
        )
    return n


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
    require_all_shards: bool = True,
    exclude_audio_urls: list[str] | None = None,
    **ingest_kwargs,
):
    """Wire work-index -> N sharded ingests -> codes index.

    Returns ``(codes_index_dir, work_index_dir, shard_dirs)``.

    ``exclude_audio_urls`` defaults to :data:`SMOKE_AUDIO_URLS` so that anything used for a smoke
    test can never reappear inside a real corpus. Pass ``[]`` to disable deliberately.
    """
    index = PodcastWorkIndex(
        source=source,
        num_shards=num_shards,
        rss_urls=rss_urls,
        manifest=manifest,
        max_episodes=max_episodes,
        exclude_audio_urls=(SMOKE_AUDIO_URLS if exclude_audio_urls is None else exclude_audio_urls),
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
    codes_index = PodcastCodesIndex(in_dirs=shard_dirs, require_all_shards=require_all_shards)
    if register:
        tk.register_output(f"podcast_codes/{tag}/work_index", index.out_dir)
        tk.register_output(f"podcast_codes/{tag}/codes_index", codes_index.out_dir)
    return codes_index.out_dir, index.out_dir, shard_dirs


# ---------------------------------------------------------------------------------------------
# True per-speaker separation, via the DuplexChat pipeline's own stage functions
# ---------------------------------------------------------------------------------------------
#: Upstream of the DuplexChat construction/reconstruction pipeline, pinned. MIT for code+metadata.
DUPLEXCHAT_URL = "https://github.com/sarulab-speech/DuplexChat.git"
#: Pinned so the dependency is reproducible and visible in the graph. Bumping it can change what
#: `extract_valid_dialogues` selects and therefore what the corpus contains, so it is a deliberate
#: decision, never a floating HEAD.
DUPLEXCHAT_COMMIT = "f798f311330125f392ef181492acb758413100c4"


def duplexchat_repo(commit: str = DUPLEXCHAT_COMMIT):
    """The pinned DuplexChat checkout. Returns the repo ``tk.Path``.

    A clone job rather than a copy of their modules into our tree: the dependency stays explicit in
    the graph, the licence and attribution stay with the code, and a bump is one constant.
    ``CloneGitRepositoryJob`` honours ``commit`` (its ``elif self.commit is not None`` branch), so
    the pin is real.
    """
    from i6_core.tools.git import CloneGitRepositoryJob

    return CloneGitRepositoryJob(url=DUPLEXCHAT_URL, commit=commit, checkout_folder_name="DuplexChat").out_repository


class PodcastDuplexIngest(Job):
    """Download -> diarize -> extract dialogues -> SEPARATE -> Mimi-encode, one shard.

    The difference from :class:`PodcastMimiIngest` is the one that matters for a full-duplex model:
    that job's ``diarize_mask`` builds two channels by muting the other speaker's turns in a mono
    mix, so during one speaker's turn the other's backchannels and laughter sit *inside* their
    channel. This job runs real separation (DialogueSidon), so each channel is one speaker.

    **Their code, not a reimplementation**, because reading it turned up four correctness details an
    independent version would plausibly miss -- and our own first attempt missed three:

    * ``separate._maybe_swap`` resolves the **source-separation permutation ambiguity** across the
      120 s / 10 s overlapping chunks. A separator has no notion of which output channel is which
      speaker, so without this a channel silently changes speaker at a chunk boundary -- fatal for
      duplex training and audible only by listening.
    * ``diarize.run_diarization`` handles **both** pyannote output shapes (community-1's
      ``DiarizeOutput.speaker_diarization`` and the older ``Annotation.itertracks``). Ours used
      ``itertracks`` only, which would have crashed on the model we had set as the default.
    * ...and it passes an in-memory ``{"waveform", "sample_rate"}`` dict rather than a path, so the
      broken torchcodec in that venv is harmless. That venv does have a broken torchcodec.
    * ``dialogue.extract_valid_dialogues`` applies the dominance filter **after** splitting long
      dialogues, so a span that is balanced on average but contains a long monologue is still caught.

    Their filters are also what makes JRE usable: sponsor reads, monologue stretches, third-speaker
    intrusions and intro music are all removed. Measured on two real JRE episodes: **42.7% and 37.3%
    retention**.

    ⚠ **Two venvs, one job.** DialogueSidon's environment is torch 2.11.0+cu128 and the moshi stack
    is 2.12.1+cu126, so they cannot share a process. The worker runs in the DuplexChat venv and
    shells out to ``podcast_mimi_encode.py`` in the moshi venv per batch. Separated audio lives only
    in ``$TMPDIR`` and is unlinked after each batch -- never stored durably, which is the point of
    emitting codes.

    ⚠ **Separation is span-scoped**, so unlike ``span_mode="whole"`` this encodes the extracted
    dialogues rather than whole episodes (on a solo sponsor read there is no second channel to
    produce). Re-deciding spans later costs a re-separate. The diarization turn list is stored per
    row so the DuplexChat-style filters -- and Open Yap's ``turns_per_minute`` /
    ``turn_taking_gap_ms`` -- stay computable downstream without re-downloading.

    ⚠ **Cost.** Per episode-hour: decode ~14 s + diarize ~36 s. Per retained hour: separate ~36 s
    (RTF 0.01, measured on our own H100) + Mimi ~1.7 s. At JRE's ~40% retention that is ~65 s per
    episode-hour, so a 4 h shard holds ~220 episode-hours. Separation roughly doubles the GPU cost
    of the mask approach and buys the thing that makes the corpus worth having.
    """

    __sis_hash_exclude__ = {"rqmt": None, "max_items": 0}

    def __init__(
        self,
        *,
        duplex_venv_python,
        mimi_venv_python,
        repo_dir: tk.Path,
        index_dir: tk.Path,
        shard_idx: int,
        diarization_model: str = "pyannote/speaker-diarization-community-1",
        num_steps: int = 30,
        gap_seconds: float = 5.0,
        max_single_speaker_ratio: float = 0.8,
        min_duration_seconds: float = 10.0,
        max_duration_seconds: float = 600.0,
        max_xcorr: float = 0.30,
        max_failure_frac: float = 0.25,
        drift_tolerance_sec: float = 5.0,
        download_workers: int = 6,
        batch_episodes: int = 4,
        max_items: int = 0,
        code_version: int = 1,
        env_ffmpeg_path: tk.Path | None = None,
        rqmt: dict | None = None,
    ):
        self.duplex_venv_python = duplex_venv_python
        self.mimi_venv_python = mimi_venv_python
        self.repo_dir = repo_dir
        self.index_dir = index_dir
        self.shard_idx = int(shard_idx)
        self.diarization_model = diarization_model
        self.num_steps = int(num_steps)
        self.gap_seconds = float(gap_seconds)
        self.max_single_speaker_ratio = float(max_single_speaker_ratio)
        self.min_duration_seconds = float(min_duration_seconds)
        self.max_duration_seconds = float(max_duration_seconds)
        self.max_xcorr = float(max_xcorr)
        self.max_failure_frac = float(max_failure_frac)
        self.drift_tolerance_sec = float(drift_tolerance_sec)
        self.download_workers = int(download_workers)
        self.batch_episodes = int(batch_episodes)
        self.max_items = int(max_items)
        self.code_version = int(code_version)
        self.env_ffmpeg_path = env_ffmpeg_path
        self.out_dir = self.output_path("codes", directory=True)
        self.rqmt = rqmt or {"gpu": 1, "cpu": 8, "mem": 64, "time": 8}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(parsed_args)
        # WHERE our ffmpeg lives, and HOW FAST we go, are not WHAT this job computes. If any of
        # these reached the hash, tuning concurrency would re-run every already-ingested shard.
        # Everything that can change the CONTENT -- the filter thresholds, num_steps, max_xcorr,
        # the diarizer -- stays hashed on purpose.
        for k in ("env_ffmpeg_path", "download_workers", "batch_episodes"):
            d.pop(k, None)
        return super().hash(d)

    def tasks(self):
        # Resumable: `done` is rebuilt from the arrow parts already written, so Sisyphus
        # reschedules an interrupted shard on its own. Per CLAUDE.md, do NOT clear such a task
        # with `hpc-rerun --include-interrupted`.
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        try:
            with open(os.path.join(self.out_dir.get(), "progress.json")) as f:
                p = json.load(f)
            return f"{p.get('ok', 0)} ok / {p.get('failed', 0)} failed, {p.get('kept_hours', 0)} h kept"
        except Exception:  # noqa: BLE001
            return None

    def run(self):
        lib_parent = _moshi_family_lib_parent()
        script = os.path.join(lib_parent, "moshi_family", "podcast_duplex_main.py")
        encoder = os.path.join(lib_parent, "moshi_family", "podcast_mimi_encode.py")
        shard = os.path.join(self.index_dir.get(), f"shard_{self.shard_idx:05d}.jsonl")
        if not os.path.exists(shard):
            raise FileNotFoundError(
                f"{shard} does not exist -- shard_idx {self.shard_idx} is outside the num_shards "
                "the index was built with; the two are set independently and must agree."
            )

        env: dict[str, str] = {}
        if self.env_ffmpeg_path is None:
            raise ValueError(
                "env_ffmpeg_path is required: decoding to 16 kHz mono for the separator goes "
                "through OUR ffmpeg, never a system one. Pass "
                "env_ffmpeg_path=InstallFFmpeg().out_path (hash-free)."
            )
        InstallFFmpeg.add_to_env(self.env_ffmpeg_path, env)

        args = [
            "--shard_jsonl",
            shard,
            "--out_dir",
            self.out_dir.get(),
            "--repo_src",
            os.path.join(self.repo_dir.get(), "src"),
            "--ffmpeg_dir",
            self.env_ffmpeg_path.get(),
            "--mimi_python",
            self.mimi_venv_python.get(),
            "--mimi_encoder",
            encoder,
            "--moshi_lib_parent",
            _moshi_pythonpath(),
            "--diarization_model",
            self.diarization_model,
            "--num_steps",
            self.num_steps,
            "--download_workers",
            self.download_workers,
            "--batch_episodes",
            self.batch_episodes,
            "--max_xcorr",
            self.max_xcorr,
            "--max_failure_frac",
            self.max_failure_frac,
            "--drift_tolerance_sec",
            self.drift_tolerance_sec,
            "--gap_seconds",
            self.gap_seconds,
            "--max_single_speaker_ratio",
            self.max_single_speaker_ratio,
            "--min_duration_seconds",
            self.min_duration_seconds,
            "--max_duration_seconds",
            self.max_duration_seconds,
        ]
        if self.max_items:
            args += ["--max_items", self.max_items]

        run_worker_script(
            self.duplex_venv_python.get(),
            script,
            args,
            log_label=f"Podcast duplex ingest shard {self.shard_idx}",
            with_hf_home=True,
            extra_env=env,
        )


def podcast_duplex_codes(
    *,
    duplex_venv_python,
    mimi_venv_python,
    tag: str,
    source: str,
    num_shards: int,
    rss_urls: list[str] | None = None,
    manifest: tk.Path | None = None,
    max_episodes: int = 0,
    register: bool = True,
    require_all_shards: bool = True,
    exclude_audio_urls: list[str] | None = None,
    **ingest_kwargs,
):
    """Wire work-index -> N sharded SEPARATING ingests -> codes index.

    Returns ``(codes_index_dir, work_index_dir, shard_dirs)``.
    """
    repo = duplexchat_repo()
    index = PodcastWorkIndex(
        source=source,
        num_shards=num_shards,
        rss_urls=rss_urls,
        manifest=manifest,
        max_episodes=max_episodes,
        exclude_audio_urls=(SMOKE_AUDIO_URLS if exclude_audio_urls is None else exclude_audio_urls),
    )
    shard_dirs = []
    for k in range(num_shards):
        job = PodcastDuplexIngest(
            duplex_venv_python=duplex_venv_python,
            mimi_venv_python=mimi_venv_python,
            repo_dir=repo,
            index_dir=index.out_dir,
            shard_idx=k,
            env_ffmpeg_path=InstallFFmpeg().out_path,
            **ingest_kwargs,
        )
        shard_dirs.append(job.out_dir)
    codes_index = PodcastCodesIndex(in_dirs=shard_dirs, require_all_shards=require_all_shards)
    if register:
        tk.register_output(f"podcast_codes/{tag}/work_index", index.out_dir)
        tk.register_output(f"podcast_codes/{tag}/codes_index", codes_index.out_dir)
    return codes_index.out_dir, index.out_dir, shard_dirs
