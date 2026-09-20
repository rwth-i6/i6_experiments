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

CHANNEL_MODES = (
    "diarize_mask",
    "stereo_passthrough",
    "mono_both",
    "dialogue_sidon",
    "dialogue_sidon_whole",
)

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
    # JRE #2553 (9,958 s) and #2554 (9,494 s). These two are where every measured cost constant in
    # this file comes from, and they are the long-episode smoke for the whole-episode path -- the
    # short pinecast clip below is under one 120 s separation chunk, so it cannot exercise chunking,
    # the multi-GB encode allocation, or the permutation chain at all. Excluded here so test audio
    # can never reach training data: a guarantee in code rather than a note someone has to remember.
    "https://traffic.megaphone.fm/GLT4909015971.mp3",
    "https://traffic.megaphone.fm/GLT1673171329.mp3",
    # 74.5 s episode, pinecast/colour-out-the-box; chosen as the shortest DuplexChat episode that
    # still carries a 20-90 s dialogue span, so the download is ~1 MB.
    "https://pinecast.com/listen/70469672-3d59-42c0-a2ba-121d97b1a1dc:"
    "92ff53eb-8adc-47ee-8dbd-04fb480995f8.mp3?source=rss&ext=asset.mp3",
]


def _stable_id(key: str) -> str:
    """A 16-digit episode id that is the SAME in every process, for ever.

    🔴 This used to be `abs(hash(key)) % 10**16`, and Python's builtin `hash()` on a str is
    **salted per process** unless PYTHONHASHSEED is set. So every rebuild of the work index minted
    completely different ids for the same episodes -- which silently breaks the two things ids are
    for: resume (`episodes.json` records ids, so a rebuilt index would re-do a finished shard) and
    growing a corpus (an episode could never be recognised as already ingested). It was invisible
    because within one process the ids are perfectly consistent, and the index is normally built
    once.

    sha256 is stable across processes, machines and Python versions. Truncating to 16 digits leaves
    ~5e15 values; at JRE's ~2k episodes the collision probability is ~4e-13, and `PodcastCodesIndex`
    hard-fails on a duplicate `item_id` anyway, so a collision is loud rather than silent.
    """
    import hashlib

    return f"{int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:16], 16) % (10**16):016d}"


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
            # A local path (or file:// URL) reads straight off disk. This is the SUPPORTED way to
            # run the corpus, not a debugging convenience -- see `rss_urls` in __init__ for why the
            # live fetch cannot be trusted from a compute node here.
            xml = b""
            try:
                if not str(url).startswith(("http://", "https://")):
                    local = str(url)[7:] if str(url).startswith("file://") else str(url)
                    with open(local, "rb") as fh:
                        xml = fh.read()
                else:
                    req = urllib.request.Request(url, headers={"User-Agent": ua})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        xml = r.read()
                root = ET.fromstring(xml)
            except Exception as e:  # noqa: BLE001
                # FATAL, not `continue`. A feed that fails silently produces a SMALLER corpus that
                # looks entirely healthy -- fewer shards, every one of them green -- and the cause
                # is a single line in a log nobody reads. Measured 2026-09-19: the RWTH egress
                # truncated this 5.4 MB chunked response to 270 KB (2,753 items -> 71) and handed
                # back a scrambled tail, so the parse failed ~5% in. Had the XML happened to stay
                # well-formed at the cut, the run would have quietly ingested 2.6% of JRE.
                raise RuntimeError(
                    f"feed {url} could not be read: {type(e).__name__}: {e}\n"
                    f"  read {len(xml)} bytes.\n"
                    f"  If this is a truncated download rather than a bad feed, stage the XML and "
                    f"pass the FILE PATH in rss_urls -- see the note on that argument."
                ) from e
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
                        "item_id": f"rss_{_stable_id(guid)}",
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
                        "item_id": f"dc_{_stable_id(url)}",
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
            raise RuntimeError(msg + " -- overlapping_spans='raise'")
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
            raise RuntimeError("index produced ZERO work items -- refusing to write empty shards")

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
        env_hook = None
        if self.env_ffmpeg_path is not None:
            # Via env_hook, NOT extra_env: `add_to_env` PREPENDS to PATH/LD_LIBRARY_PATH, and
            # `run_worker_script` applies extra_env with `dict.update`. Mixing ffmpeg into a fresh
            # dict therefore produced `PATH=<ffmpeg>/bin:` and REPLACED the inherited PATH, which
            # breaks any squashfs-packed venv (its bin/python is a shell launcher needing
            # basename/dirname) with exit 127 before Python starts. env_hook runs on the full env.
            _ff = self.env_ffmpeg_path

            def env_hook(e, _ff=_ff):  # noqa: F811
                InstallFFmpeg.add_to_env(_ff, e)

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
            env_hook=env_hook,
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
                        raise RuntimeError(
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
                raise RuntimeError(
                    msg + ". Read their summary.json / failures.jsonl. Indexing anyway would "
                    "produce a corpus that is quietly missing whole shards -- pass "
                    "require_all_shards=False only once you know why."
                )
            print(f"[index] WARNING: {msg}", flush=True)

        if not rows:
            raise RuntimeError("indexed ZERO rows -- refusing to present this as a corpus.")

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


#: ✅ MEASURED on the PILOT SHARD through the manager (2026-09-19, SLURM 4263279, c25g): 3 episodes,
#: **7.817 episode-hours**, 0 failures. This is the number the whole "one shard first" gate exists to
#: produce, and it supersedes both the 113.1 prediction and the 107.55 from the 2-episode smoke.
#:
#:                  predicted   smoke (2.6 ep-h)   PILOT (7.8 ep-h)
#:    decode             2.71               6.16               3.60
#:    diarize           34.50              35.63              24.32
#:    separate          65.90              47.99              61.56   <- bottleneck, 63%
#:    repair                -               0.45               0.16
#:    encode             9.98              14.42               6.93
#:                     ------             ------             ------
#:                     113.09             107.55              97.88
#:
#: The trend down is fixed-cost amortisation: model loads (~60 s) and the mimi encoder's per-batch
#: startup are spread over more audio as the sample grows, and a real shard is ~130 episode-hours --
#: 17x the pilot -- so if anything this is still conservative. `separate` stays the bottleneck and is
#: the only term worth optimising.
#:
#: ⚠ Still n = 3 episodes, all 2.3-3.0 h. JRE episode length varies ~4x, and the cost is priced PER
#: EPISODE-HOUR, so the scaling should hold -- but the shard target is 4 h against an 8 h walltime,
#: i.e. 2x headroom, which is what absorbs the error. Each shard writes its own `summary.json`;
#: re-read them after the first few of the full fan-out rather than trusting this to the decimal.
#: ✅ UPDATED 2026-09-20 for the two stages added since: per-channel ASR and the speaker-embedding
#: channel scorer.
#:     97.9  separation + diarization + decode + encode (measured, pilot shard 4263279)
#:   + 22.3  ASR, faster-whisper/medium on both channels (measured: 265 and 415 audio-h/GPU-h)
#:   + ~10   speaker embeddings, 6 s windows at a 12 s hop, both channels. Measured 59.5 s/ep-h on
#:           CPU at 4 threads; this runs on the worker's GPU, so ~10 is a conservative allowance.
#:   ------
#:    130.2
#: ⚠ Each shard still writes its realised `sec_per_episode_hour`; re-read the first few of the
#: fan-out rather than trusting this to the decimal. The 2x rqmt headroom is what absorbs the error.
EPISODE_SEC_PER_EPISODE_HOUR = 130.2


def episode_hours_per_shard(target_runtime_hours: float = 4.0) -> float:
    """Episode-hours one WHOLE-EPISODE separating shard can do in ``target_runtime_hours``.

    ~147 episode-hours at the 4 h default, so JRE's ~5,500 h is ~38 shards (vs ~27 for the
    dialogue path, which only separates what survives filtering).
    """
    return (float(target_runtime_hours) * 3600.0) / EPISODE_SEC_PER_EPISODE_HOUR


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
    if channel_mode == "dialogue_sidon_whole":
        # Separation runs on 100% of the episode here, not the ~42.7% that survives filtering, so
        # this must NOT reuse the dialogue constant -- doing so under-shards by ~33%.
        return episode_hours_per_shard(target_runtime_hours)
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
        # Via env_hook, NOT extra_env: `add_to_env` PREPENDS to PATH/LD_LIBRARY_PATH, and
        # `run_worker_script` applies extra_env with `dict.update`. Mixing ffmpeg into a fresh
        # dict therefore produced `PATH=<ffmpeg>/bin:` and REPLACED the inherited PATH, which
        # breaks any squashfs-packed venv (its bin/python is a shell launcher needing
        # basename/dirname) with exit 127 before Python starts. env_hook runs on the full env.
        _ff = self.env_ffmpeg_path

        def env_hook(e, _ff=_ff):
            InstallFFmpeg.add_to_env(_ff, e)

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
            env_hook=env_hook,
        )


class PodcastEpisodeIngest(Job):
    """Download -> diarize -> SEPARATE THE WHOLE EPISODE -> repair permutation -> Mimi-encode.

    The difference from :class:`PodcastDuplexIngest` is *when* DuplexChat's editorial filter is
    applied. That job calls ``run_separation`` inside ``for dlg in extract_valid_dialogues(...)``,
    which bakes the filter into the artifact: 57% of JRE is discarded at ingest and cannot be
    recovered without paying for separation again. This job separates the whole episode and stores
    the **full** diarization beside the codes, so slicing becomes a separate, cheap, CPU-only job
    (:class:`PodcastDialogueSlice`) re-runnable at any parameters, for ever.

    That works because ``dialogue.py`` imports nothing but ``dataclasses`` and reads only
    ``seg["speaker"]/["start"]/["end"]`` -- ``extract_valid_dialogues`` is a **pure function of the
    diarization segments**. Store the segments and their slicing is reproducible bit-for-bit later.

    Whole-episode separation needs no change to their code: ``run_separation`` already chunks
    internally (120 s chunks, 10 s overlap, cross-fade stitch).

    **What this keeps that their pipeline discards**, none of it recoverable later without re-running
    the whole separation pass:

    * ``exclusive_speaker_diarization`` -- the same turns with overlaps removed. Its **difference**
      from ``speaker_diarization`` is the overlap map, free, and overlap is the full-duplex signal.
    * ``speaker_embeddings`` -- one centroid per speaker, stored with the label list because
      pyannote documents them as "sorted in ``speaker_diarization.labels()`` order" and without the
      labels the matrix is anonymous. Cross-episode identity, guest de-duplication and PersonaPlex
      voice prompts need this; nothing else provides it.
    * the per-window permutation evidence (see below).

    ⚠ **Permutation.** ``separate._maybe_swap`` decides each chunk's channel assignment against the
    previous chunk **as already stitched** -- a chain, with no margin threshold and the decision
    discarded. Over ~89 chunks of a 2.7 h episode, routed through ad breaks, one bad link propagates
    to the end. We leave their function alone and repair the stitched output afterwards against the
    episode-level diarization (``moshi_family.perm_repair``), which cannot drift because every
    window is scored against one episode-wide reference. **Measured (Test A, 2026-09-19, 18 episodes
    with ground truth from per-speaker mics): the chain preserves speaker identity 14/18, the
    anchored repair 18/18.** The failure mode is MUSIC interludes (3/6 chained), not solo ad reads
    (6/6) -- a solo read still has one consistent voice to correlate, music has none. The repair also
    fixes the global convention for free: channel 0 is always the longest-speaking speaker, the same
    way in every episode.

    ⚠ **Two venvs, one job.** DialogueSidon's environment is torch 2.11.0+cu128 and the moshi stack
    is 2.12.1+cu126, so they cannot share a process. The worker runs in the DuplexChat venv and
    shells out per episode to ``podcast_mimi_encode.py`` in the moshi venv. Separated audio lives
    only in ``$TMPDIR`` and is unlinked immediately -- never stored durably.

    ⚠ **One episode per encoder call, deliberately.** A 2.7 h episode is ~1.9 GB of float32 stereo
    crossing the venv boundary as a wav; batching multiplies that in ``$TMPDIR``. The mimi model load
    (~15 s) amortises fine over hours of audio, so there is nothing to win by batching here.

    ⚠ **Mimi is causal**, so slicing frames out of a whole-episode encode is NOT bit-identical to
    encoding that dialogue alone -- each window inherits left-context from before it. This is the
    more inference-faithful of the two (the model streams from the start) and is the deliberate
    trade; it means parity against :class:`PodcastDuplexIngest` is checked on *spans*, not on codes.

    ⚠ **Cost.** Separation now runs on 100% of the episode rather than the ~42.7% that survives
    filtering, so the whole pass is ~1.6x the dialogue path: measured ~69.6 s/episode-hour there,
    predicted ~110 here. **Derive the shard count from a measured shard's own
    ``sec_per_episode_hour``, not from that prediction** -- ``shards_for_hours`` is wired for it.
    """

    # ⚠ `asr_backend`/`asr_model` are excluded ONLY while None, which is the "no ASR" default --
    # exactly what `__sis_hash_exclude__` means. That is the right mechanism HERE and the wrong one
    # for `rqmt` (see `hash()` below), and the difference is the direction: a populated `rqmt` must
    # still be excluded, whereas a populated ASR backend must be HASHED, because it changes the word
    # alignments in the corpus. Excluding them unconditionally would let an ASR and a no-ASR corpus
    # hash identically -- and the enabling argument, `asr_venv_python`, is a PATH we must drop, so
    # these two are the only hashed record that ASR ran at all.
    __sis_hash_exclude__ = {
        "rqmt": None,
        "max_items": 0,
        "asr_backend": None,
        "asr_model": None,
        "perm_scorer": "energy",
    }

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
        seed: int = 1234,
        repair_permutation: bool = True,
        max_xcorr: float = 0.30,
        max_failure_frac: float = 0.25,
        drift_tolerance_sec: float = 5.0,
        download_workers: int = 6,
        max_items: int = 0,
        code_version: int = 1,
        duplexchat_commit: str = DUPLEXCHAT_COMMIT,
        env_ffmpeg_path: tk.Path | None = None,
        asr_venv_python=None,
        asr_backend: str | None = None,
        asr_model: str | None = None,
        asr_batch_size: int = 16,
        perm_scorer: str = "energy",
        rqmt: dict | None = None,
    ):
        self.duplex_venv_python = duplex_venv_python
        self.mimi_venv_python = mimi_venv_python
        self.repo_dir = repo_dir
        self.index_dir = index_dir
        self.shard_idx = int(shard_idx)
        self.diarization_model = diarization_model
        self.num_steps = int(num_steps)
        # Hashed on purpose: the diffusion is seeded from it, so it decides the audio the corpus is
        # built from. We deliberately do NOT store the separated waveform, so the seed plus this
        # code is the only thing that makes the audio reproducible at all.
        self.seed = int(seed)
        self.repair_permutation = bool(repair_permutation)
        self.max_xcorr = float(max_xcorr)
        self.max_failure_frac = float(max_failure_frac)
        self.drift_tolerance_sec = float(drift_tolerance_sec)
        self.download_workers = int(download_workers)
        self.max_items = int(max_items)
        self.code_version = int(code_version)
        self.duplexchat_commit = duplexchat_commit
        self.env_ffmpeg_path = env_ffmpeg_path
        # 🔴 Refuse a HALF-configured ASR. `asr_backend` is hashed once set, but the worker
        # only runs ASR when BOTH this and the venv are present -- so `asr_backend="faster_whisper"`
        # without `asr_venv_python` re-hashes every shard (a full ~247 GPU-h re-run) and produces a
        # corpus with `asr_json` NULL in every row: exactly the untrainable corpus this whole plan
        # exists to eliminate, with no error at graph build or at run time. Cheap to hit while
        # wiring a recipe, and invisible until a training run pad-collapses weeks later.
        if bool(asr_backend) != bool(asr_venv_python):
            raise ValueError(
                f"asr_backend={asr_backend!r} and asr_venv_python={asr_venv_python!r} must be set "
                "together: one without the other silently ingests a corpus with no alignments."
            )
        self.asr_venv_python = asr_venv_python
        self.asr_backend = asr_backend
        self.asr_model = asr_model
        self.asr_batch_size = int(asr_batch_size)
        # HASHED -- it changes which speaker each channel carries, i.e. the corpus content. Excluded
        # only at "energy" (the old behaviour) so existing shards keep their hash.
        assert perm_scorer in ("energy", "embedding"), perm_scorer
        self.perm_scorer = perm_scorer
        self.out_dir = self.output_path("codes", directory=True)
        # Host RAM, not GPU: a 2.7 h episode is ~0.6 GB as 16 kHz mono, ~1.9 GB separated at 24 kHz
        # stereo, and the repair holds a copy. 64 GB is comfortable; 16 would not be.
        self.rqmt = rqmt or {"gpu": 1, "cpu": 8, "mem": 64, "time": 8}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(parsed_args)
        # WHERE our ffmpeg lives and HOW FAST we go are not WHAT this job computes. If any reached
        # the hash, tuning concurrency would re-run every already-ingested shard. Everything that
        # can change the CONTENT -- num_steps, seed, max_xcorr, the diarizer, whether the
        # permutation is repaired -- stays hashed on purpose.
        # ⚠ `rqmt` is popped HERE, not left to `__sis_hash_exclude__`. That mechanism excludes an
        # argument only while it EQUALS the listed default, so a populated `rqmt` -- reachable via
        # `podcast_episode_codes(**ingest_kwargs)` -- would be hashed like any other kwarg and
        # orphan an already-completed ~173 GPU-h separation pass. This is precisely the `Compute`
        # post-mortem in CLAUDE.md, and the reason that fix needed a hash-excluded channel rather
        # than `__sis_hash_exclude__`.
        # `asr_venv_python` is WHERE the backend lives, not WHAT it computes -- the same
        # reasoning as env_ffmpeg_path. `asr_batch_size` is throughput: faster-whisper runs its VAD
        # and segments the audio BEFORE batching, so the batch size only sets how many of those
        # segments decode in parallel, not where they start or end.
        # ⚠ That is an assumption about someone else's library, and the kind that is silent when
        # wrong: the knob would change corpus content while every hash stayed put. So it was
        # MEASURED (2026-09-20, /hpcwork/tt201262/asr_bakeoff/batch_inv.py, 90 s of real separated
        # podcast audio at batch 4 vs 16): 229 words, 0 text mismatches, max onset delta 0.000
        # frames. Re-run it if the backend is ever changed.
        for k in (
            "env_ffmpeg_path",
            "download_workers",
            "rqmt",
            "asr_venv_python",
            "asr_batch_size",
        ):
            d.pop(k, None)
        return super().hash(d)

    def tasks(self):
        # Resumable: `done` is rebuilt from the arrow parts already written, so Sisyphus reschedules
        # an interrupted shard on its own. Per CLAUDE.md, do NOT clear such a task with
        # `hpc-rerun --include-interrupted`.
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        try:
            with open(os.path.join(self.out_dir.get(), "progress.json")) as f:
                p = json.load(f)
            return f"{p.get('ok', 0)} ok / {p.get('failed', 0)} failed, {p.get('episode_hours', 0)} ep-h"
        except Exception:  # noqa: BLE001
            return None

    def run(self):
        lib_parent = _moshi_family_lib_parent()
        script = os.path.join(lib_parent, "moshi_family", "podcast_episode_main.py")
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
                "through OUR ffmpeg, never a system one. It is also the TIME BASE a later slice "
                "job must reproduce. Pass env_ffmpeg_path=InstallFFmpeg().out_path (hash-free)."
            )
        # Via env_hook, NOT extra_env: `add_to_env` PREPENDS to PATH/LD_LIBRARY_PATH, and
        # `run_worker_script` applies extra_env with `dict.update`. Mixing ffmpeg into a fresh
        # dict therefore produced `PATH=<ffmpeg>/bin:` and REPLACED the inherited PATH, which
        # breaks any squashfs-packed venv (its bin/python is a shell launcher needing
        # basename/dirname) with exit 127 before Python starts. env_hook runs on the full env.
        _ff = self.env_ffmpeg_path

        def env_hook(e, _ff=_ff):
            InstallFFmpeg.add_to_env(_ff, e)

        args = [
            "--shard_jsonl",
            shard,
            "--out_dir",
            self.out_dir.get(),
            "--repo_src",
            os.path.join(self.repo_dir.get(), "src"),
            "--duplexchat_commit",
            self.duplexchat_commit,
            "--ffmpeg_dir",
            self.env_ffmpeg_path.get(),
            "--mimi_python",
            self.mimi_venv_python.get(),
            "--mimi_encoder",
            encoder,
            "--perm_repair_py",
            os.path.join(lib_parent, "moshi_family", "perm_repair.py"),
            "--moshi_lib_parent",
            _moshi_pythonpath(),
            "--diarization_model",
            self.diarization_model,
            "--num_steps",
            self.num_steps,
            "--seed",
            self.seed,
            "--repair_permutation",
            1 if self.repair_permutation else 0,
            "--download_workers",
            self.download_workers,
            "--max_xcorr",
            self.max_xcorr,
            "--max_failure_frac",
            self.max_failure_frac,
            "--drift_tolerance_sec",
            self.drift_tolerance_sec,
        ]
        if self.asr_venv_python and self.asr_backend:
            args += [
                "--asr_python",
                (self.asr_venv_python.get() if hasattr(self.asr_venv_python, "get") else self.asr_venv_python),
                "--asr_worker",
                os.path.join(lib_parent, "moshi_family", "podcast_asr.py"),
                "--asr_backend",
                self.asr_backend,
                "--asr_model",
                self.asr_model or "medium",
                "--asr_batch_size",
                self.asr_batch_size,
            ]
        if self.perm_scorer != "energy":
            args += ["--perm_scorer", self.perm_scorer]
        if self.max_items:
            args += ["--max_items", self.max_items]

        run_worker_script(
            self.duplex_venv_python.get(),
            script,
            args,
            log_label=f"Podcast episode ingest shard {self.shard_idx}",
            with_hf_home=True,
            # No PYTHONPATH: nothing in this venv may import `moshi_family` as a package (its
            # __init__ needs the moshi stack). The worker loads perm_repair by file path, and the
            # encoder subprocess sets its own PYTHONPATH from --moshi_lib_parent.
            extra_env=env,
            env_hook=env_hook,
        )


def podcast_episode_codes(
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
    shard_indices: list[int] | None = None,
    **ingest_kwargs,
):
    """Wire work-index -> N sharded WHOLE-EPISODE separating ingests -> codes index.

    Returns ``(codes_index_dir, work_index_dir, shard_dirs)``.

    The output is one row per EPISODE with the full diarization attached, which is an intermediate
    artifact rather than a training corpus: run :class:`PodcastDialogueSlice` over it to get
    training-shaped rows. That separation is the entire point -- the expensive, once-only work
    (download, diarize, separate, encode) lands here, and every editorial decision about what counts
    as a usable dialogue becomes a cheap CPU job downstream.

    ⚠ ``num_shards`` must come from ``shards_for_hours(..., channel_mode="dialogue_sidon")``, which
    prices the whole-episode path from its own measured cost. Sizing it with the gating model gives
    ~33% too few shards and jobs that quietly overrun the target runtime.
    """
    # ⚠ Forward the pin. `duplexchat_commit` is a hashed PodcastEpisodeIngest kwarg that is written
    # verbatim into `provenance_json`, so calling `duplexchat_repo()` with the module default here
    # would let a caller pass a different commit, re-hash every shard, record that commit as
    # provenance -- and actually run the DEFAULT checkout. Silently wrong provenance is worse than
    # a crash, because the corpus looks correctly labelled.
    repo = duplexchat_repo(ingest_kwargs.get("duplexchat_commit", DUPLEXCHAT_COMMIT))
    index = PodcastWorkIndex(
        source=source,
        num_shards=num_shards,
        rss_urls=rss_urls,
        manifest=manifest,
        max_episodes=max_episodes,
        exclude_audio_urls=(SMOKE_AUDIO_URLS if exclude_audio_urls is None else exclude_audio_urls),
    )
    shard_dirs = []
    # Build a SUBSET without changing `num_shards`. Shard->episode assignment is a global
    # bin-packing hashed on num_shards, so running "4 now, N later" as two different counts reuses
    # NOTHING: freeze the count at its final value and run only some of them.
    for k in shard_indices if shard_indices is not None else range(num_shards):
        job = PodcastEpisodeIngest(
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
        tk.register_output(f"podcast_episodes/{tag}/work_index", index.out_dir)
        tk.register_output(f"podcast_episodes/{tag}/codes_index", codes_index.out_dir)
    return codes_index.out_dir, index.out_dir, shard_dirs


class PodcastDialogueSlice(Job):
    """Slice whole-episode code rows into dialogue rows -- CPU only, no GPU, no audio.

    The cheap half of the whole-episode design. :class:`PodcastEpisodeIngest` pays once for
    download, diarization, separation and Mimi encoding; this decides what counts as a usable
    dialogue, and can be re-run at any parameters for as long as the episode corpus exists.

    It reproduces DuplexChat's slicing **exactly**, by importing and calling their
    ``extract_valid_dialogues`` from the pinned clone rather than re-implementing it. That is
    possible at all because ``dialogue.py`` imports nothing but ``dataclasses`` and reads only
    ``seg["speaker"]/["start"]/["end"]`` -- it is a pure function of the diarization segments, and
    the segments are stored on every episode row.

    ⚠ **Never re-implement their rules here.** The four filters interact in ways that are easy to get
    subtly wrong: dominance is applied *after* long dialogues are chunked (so a span that is balanced
    on average but contains a long monologue is still caught), single-speaker runs are dropped by
    never being *emitted* rather than by a filter, and a gap of exactly ``gap_seconds`` splits.

    **The four filter thresholds and the episode gate are hashed constructor args**, so a different
    retention policy is a new cheap CPU job rather than a re-separation. That is the entire payoff of
    the design: at DuplexChat's defaults JRE retains **42.7%**, and changing that number used to cost
    the whole ~173 GPU-h separation pass.

    ⚠ **Frame mapping.** An episode row's frame 0 is episode t=0 (``span_start_sec == 0.0``, asserted
    at run time), so a dialogue at [start, end) seconds becomes frames
    ``[floor(start*fr), ceil(end*fr))`` -- floor/ceil rather than round, matching the existing
    ``spans_json`` convention, so a slice is a superset of the dialogue and never clips its edges.
    The emitted row's ``span_*`` describe the frames actually taken; the requested seconds are kept
    in ``channel_info``.

    ⚠ **The codes layout is codebook-major** (``k * n_frames + f``), so slicing frames is a slice of
    the second axis after reshaping. A contiguous slice of the flat array would take a band of
    *codebooks* instead -- valid-looking output, completely wrong audio.

    ⚠ Output is a second copy of the (much smaller) retained codes rather than an index into the
    episode rows. At ~1.44 MB per stereo hour that is ~3.4 GB for all of JRE, which is cheap enough
    that plugging into every existing consumer unchanged is worth more than avoiding the copy.
    """

    __sis_hash_exclude__ = {"rqmt": None, "rows_per_part": 2000}

    def __init__(
        self,
        *,
        mimi_venv_python,
        repo_dir: tk.Path,
        in_dir: tk.Path,
        gap_seconds: float = 5.0,
        max_single_speaker_ratio: float = 0.8,
        min_duration_seconds: float = 10.0,
        max_duration_seconds: float = 600.0,
        min_dialogues_per_episode: int = 4,
        rows_per_part: int = 2000,
        code_version: int = 1,
        rqmt: dict | None = None,
    ):
        self.mimi_venv_python = mimi_venv_python
        self.repo_dir = repo_dir
        self.in_dir = in_dir
        self.gap_seconds = float(gap_seconds)
        self.max_single_speaker_ratio = float(max_single_speaker_ratio)
        self.min_duration_seconds = float(min_duration_seconds)
        self.max_duration_seconds = float(max_duration_seconds)
        self.min_dialogues_per_episode = int(min_dialogues_per_episode)
        self.rows_per_part = int(rows_per_part)
        self.code_version = int(code_version)
        self.out_dir = self.output_path("codes", directory=True)
        # No GPU: this only reads arrow and calls a pure-Python filter. Memory is one episode's
        # codes at a time (~4 MB for 2.7 h) plus the rows accumulating toward one part.
        self.rqmt = rqmt or {"cpu": 4, "mem": 16, "time": 4}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(parsed_args)
        # Same reasoning as PodcastEpisodeIngest: `__sis_hash_exclude__` drops these only while they
        # equal their defaults, so tuning either would re-hash and re-run a whole re-slice. Neither
        # changes WHAT is produced -- `rqmt` is scheduling, and `rows_per_part` is arrow layout that
        # `PodcastCodesIndex` reads through regardless.
        for k in ("rqmt", "rows_per_part"):
            d.pop(k, None)
        return super().hash(d)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        try:
            with open(os.path.join(self.out_dir.get(), "summary.json")) as f:
                s = json.load(f)
            return f"{s.get('n_dialogues', 0)} dlg, {s.get('kept_hours', 0)} h, retention {s.get('retention', 0):.3f}"
        except Exception:  # noqa: BLE001
            return None

    def run(self):
        lib_parent = _moshi_family_lib_parent()
        script = os.path.join(lib_parent, "moshi_family", "podcast_slice_main.py")
        run_worker_script(
            self.mimi_venv_python.get(),
            script,
            [
                "--in_dir",
                self.in_dir.get(),
                "--out_dir",
                self.out_dir.get(),
                "--repo_src",
                os.path.join(self.repo_dir.get(), "src"),
                "--gap_seconds",
                self.gap_seconds,
                "--max_single_speaker_ratio",
                self.max_single_speaker_ratio,
                "--min_duration_seconds",
                self.min_duration_seconds,
                "--max_duration_seconds",
                self.max_duration_seconds,
                "--min_dialogues_per_episode",
                self.min_dialogues_per_episode,
                "--rows_per_part",
                self.rows_per_part,
            ],
            log_label="Podcast dialogue slice",
            extra_env={"PYTHONPATH": _moshi_pythonpath()},
        )


def podcast_sliced_codes(
    *,
    mimi_venv_python,
    tag: str,
    episode_shard_dirs: list,
    register: bool = True,
    require_all_shards: bool = True,
    **slice_kwargs,
):
    """One :class:`PodcastDialogueSlice` per episode shard -> a codes index over the results.

    Returns ``(codes_index_dir, slice_dirs)``. One slice job per ingest shard rather than one over
    everything, so the fan-out matches the ingest and a re-slice parallelises the same way.
    """
    repo = duplexchat_repo()
    slice_dirs = []
    for d in episode_shard_dirs:
        job = PodcastDialogueSlice(mimi_venv_python=mimi_venv_python, repo_dir=repo, in_dir=d, **slice_kwargs)
        slice_dirs.append(job.out_dir)
    codes_index = PodcastCodesIndex(in_dirs=slice_dirs, require_all_shards=require_all_shards)
    if register:
        # Its own namespace: `podcast_duplex_codes` already registers
        # `podcast_codes/<tag>/codes_index`, and building both paths for one podcast -- the natural
        # way to compare them -- would double-register that alias.
        tk.register_output(f"podcast_dialogues/{tag}/codes_index", codes_index.out_dir)
    return codes_index.out_dir, slice_dirs


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


#: Cosine above which two episode centroids are taken to be the same person. MEASURED on the pilot,
#: not guessed: the recurring host scores **0.959-0.970** across episode pairs while different
#: speakers sit at -0.03..+0.75, and the SAME episode through two independent ingest runs scores
#: 1.000 (pyannote's embeddings are deterministic). 0.85 sits in the empty band between those.
HOST_MATCH_COSINE = 0.85
#: A speaker needs this much speech in an episode to be a host candidate; below it the label is
#: usually a diarization artefact.
HOST_MIN_SPEECH_SEC = 300.0


def _episode_speakers(row):
    """(labels, unit-norm centroids, seconds-per-label, channel->label) for one episode row."""
    import json as _json

    import numpy as _np

    labels = _json.loads(row["speaker_labels_json"])
    dim = int(row["speaker_embedding_dim"] or 0)
    emb = _np.asarray(row["speaker_embeddings"] or [], dtype=_np.float64)
    if not dim or emb.size != len(labels) * dim:
        return None
    E = emb.reshape(len(labels), dim)
    E = E / (_np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sec = {}
    for t in _json.loads(row["diar_turns_json"] or "[]"):
        k = str(t["speaker"])
        sec[k] = sec.get(k, 0.0) + float(t["end"]) - float(t["start"])
    chan = _json.loads(row["channel_speakers_json"]) if row.get("channel_speakers_json") else None
    return labels, E, sec, chan


def find_host(episodes):
    """The identity present in the most episodes, as a unit-norm centroid.

    On an interview podcast that is the host -- and choosing the assistant by IDENTITY rather than
    by position is what gives the corpus ONE assistant voice across thousands of hours instead of a
    different person per episode. A positional rule ("channel 0", i.e. whoever spoke most) picks the
    guest whenever the guest dominates, which on JRE is common.

    Returns ``(centroid, n_episodes_supporting)`` or ``(None, 0)``.
    """
    import numpy as _np

    cands = []  # (episode_index, centroid)
    for ei, ep in enumerate(episodes):
        labels, E, sec, _ = ep
        for i, lab in enumerate(labels):
            if sec.get(lab, 0.0) >= HOST_MIN_SPEECH_SEC:
                cands.append((ei, E[i]))
    if not cands:
        return None, 0
    best, best_n = None, 0
    for ei, v in cands:
        # Count DISTINCT other episodes containing this identity. Counting candidates instead would
        # let one episode with several matching labels outvote genuine recurrence.
        hits = {ej for ej, w in cands if ej != ei and float(v @ w) >= HOST_MATCH_COSINE}
        if len(hits) > best_n:
            best, best_n = v, len(hits)
    if best is None or best_n == 0:
        return None, 0
    # Refine: average every centroid that matches, so the reference is not one episode's noise.
    members = [w for _, w in cands if float(best @ w) >= HOST_MATCH_COSINE]
    c = _np.mean(members, axis=0)
    return c / (_np.linalg.norm(c) + 1e-9), best_n


def host_channel(ep, host_centroid):
    """Which channel ('a'/'b') carries the host in this episode, or None."""
    labels, E, _sec, chan = ep
    if host_centroid is None or not chan or len(chan) < 2:
        return None
    idx = {lab: i for i, lab in enumerate(labels)}
    sims = []
    for ci, lab in enumerate(chan[:2]):
        i = idx.get(str(lab))
        sims.append(float(E[i] @ host_centroid) if i is not None else -1.0)
    if max(sims) < HOST_MATCH_COSINE:
        return None  # the host is not in this episode at all
    return "a" if sims[0] >= sims[1] else "b"


class PodcastCodesTrainData(Job):
    """Dialogue code rows -> the canonical codes-training schema the loader reads.

    This is a pure column mapping: CPU only, no GPU, no audio. It exists as its own job precisely
    because it is cheap -- the expensive separation/ASR pass stays untouched while the one genuinely
    editorial decision in it, *which speaker is the assistant*, becomes a hashed knob that can be
    re-decided for the price of a few CPU minutes.

    ⚠ Normalising HERE rather than teaching the loader the podcast's internal `codes_a`/`words_a`
    schema is deliberate: the loader should know ONE codes schema, not one per producer.

    Output columns are exactly what `train_data_common.decode_codes_row` reads:
    ``codes_assistant``, ``codes_user``, ``n_codebooks``, ``n_frames``, ``frame_rate``,
    ``alignments`` -- plus ``id``/``duration`` for the manifest.

    🔴 **Only the assistant's words become `alignments`.** The text row is Moshi's inner monologue,
    i.e. what the model itself says; putting the other speaker's words there trains it to speak its
    interlocutor's lines. `interleave_text` places every alignment it is given and IGNORES the
    speaker field, so this selection is the only thing preventing that.
    """

    __sis_hash_exclude__ = {"rqmt": None}

    def __init__(
        self,
        *,
        shard_dirs: list[tk.Path],
        episode_shard_dirs: list[tk.Path] | None = None,
        assistant_channel: str = "a",
        selection_seed: int = 0,
        min_assistant_words: int = 8,
        rqmt: dict | None = None,
    ):
        # "host": choose by speaker IDENTITY, so the assistant is the same person in every
        # episode. Measured feasible on the pilot -- the recurring speaker scores 0.959-0.970 across
        # episodes against <=0.75 for anyone else -- and it needs the EPISODE rows, which carry the
        # centroids; the dialogue rows do not.
        # "random": one row per dialogue, channel picked by a SEEDED hash of the item id --
        # deterministic and reproducible, never `random.random()`. For a first arm we want a
        # speaker who talks, host or guest; identity-consistency ("host") is a later concern.
        assert assistant_channel in ("a", "b", "both", "host", "random"), assistant_channel
        if assistant_channel == "host" and not episode_shard_dirs:
            raise ValueError("assistant_channel='host' needs episode_shard_dirs for the embeddings")
        self.shard_dirs = list(shard_dirs)
        self.episode_shard_dirs = list(episode_shard_dirs or [])
        self.assistant_channel = assistant_channel
        # Hashed: it decides which speaker each row trains as the assistant, so it is corpus
        # CONTENT. Changing it must yield a different corpus, not silently reuse this one.
        self.selection_seed = int(selection_seed)
        self.min_assistant_words = int(min_assistant_words)
        self.rqmt = rqmt or {"cpu": 4, "mem": 16, "time": 4}
        self.out_dir = self.output_path("dataset", directory=True)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import hashlib as _hashlib
        import json as _json

        import numpy as _np
        from datasets import Dataset, Features, Sequence, Value, load_from_disk

        feats = Features(
            {
                "id": Value("string"),
                "duration": Value("float32"),
                "n_codebooks": Value("int32"),
                "n_frames": Value("int32"),
                "frame_rate": Value("float32"),
                "codes_assistant": Sequence(Value("int16")),
                "codes_user": Sequence(Value("int16")),
                "alignments": Sequence(
                    {
                        "text": Value("string"),
                        "start": Value("float32"),
                        "end": Value("float32"),
                        "speaker": Value("string"),
                    }
                ),
            }
        )

        def _parts(d):
            root = d.get_path()
            return sorted(
                os.path.join(root, "parts", n)
                for n in os.listdir(os.path.join(root, "parts"))
                if not n.endswith(".tmp")
            )

        # ---- identity, if asked for ---------------------------------------------------------
        host_by_episode = {}
        if self.assistant_channel == "host":
            eps, keys = [], []
            for d in self.episode_shard_dirs:
                for p in _parts(d):
                    for r in load_from_disk(p):
                        info = _episode_speakers(r)
                        if info is not None:
                            eps.append(info)
                            keys.append(str(r["item_id"]))
            centroid, n_sup = find_host(eps)
            if centroid is None:
                raise RuntimeError(
                    f"assistant_channel='host' but no recurring speaker was found across "
                    f"{len(eps)} episodes. Refusing to silently fall back to a positional rule: "
                    "that would give a corpus whose assistant is a different person per episode, "
                    "which is the opposite of what this mode is for."
                )
            for k, ep in zip(keys, eps):
                ch = host_channel(ep, centroid)
                if ch:
                    host_by_episode[k] = ch
            print(
                f"[train_data] host identified in {len(host_by_episode)}/{len(eps)} episodes "
                f"(supported by {n_sup} episodes at cos >= {HOST_MATCH_COSINE})",
                flush=True,
            )

        want = ["a", "b"] if self.assistant_channel == "both" else [self.assistant_channel]
        cols = {k: [] for k in feats}
        n_in = n_out = n_nowords = n_short = n_nohost = 0

        for d in self.shard_dirs:
            for p in _parts(d):
                ds = load_from_disk(p)
                for r in ds:
                    n_in += 1
                    if self.assistant_channel == "random":
                        # Seeded per row, so the corpus is reproducible and a re-run is identical.
                        d8 = _hashlib.sha256(f"{self.selection_seed}:{r['item_id']}".encode()).digest()
                        want = ["a" if d8[0] % 2 == 0 else "b"]
                    elif self.assistant_channel == "host":
                        ch_sel = host_by_episode.get(str(r["episode_id"]))
                        if ch_sel is None:
                            n_nohost += 1
                            continue
                        want = [ch_sel]
                    for ch in want:
                        wjson = r.get(f"words_{ch}")
                        if not wjson:
                            n_nowords += 1
                            continue
                        words = _json.loads(wjson)
                        if len(words) < self.min_assistant_words:
                            # A row whose assistant barely speaks is a row whose text stream is
                            # almost entirely PAD -- it teaches silence, which is the failure this
                            # corpus exists to fix. Drop it rather than dilute with it.
                            n_short += 1
                            continue
                        other = "b" if ch == "a" else "a"
                        cols["id"].append(f"{r['item_id']}@{ch}")
                        cols["duration"].append(float(r["duration_sec"]))
                        cols["n_codebooks"].append(int(r["n_codebooks"]))
                        cols["n_frames"].append(int(r["n_frames"]))
                        cols["frame_rate"].append(float(r["frame_rate"]))
                        # Flat `k * n_frames + f`, copied as-is -- both sides use the same layout,
                        # so this is a rename, never a reshape.
                        cols["codes_assistant"].append(_np.asarray(r[f"codes_{ch}"], dtype=_np.int16))
                        cols["codes_user"].append(_np.asarray(r[f"codes_{other}"], dtype=_np.int16))
                        cols["alignments"].append(
                            [
                                {
                                    "text": w["text"],
                                    "start": float(w["start"]),
                                    "end": float(w["end"]),
                                    # Rebound to the role vocabulary the loader uses. The diarized
                                    # label is kept in the dialogue row, not here.
                                    "speaker": "assistant",
                                }
                                for w in words
                            ]
                        )
                        n_out += 1

        if not n_out:
            raise RuntimeError(
                f"produced ZERO training rows from {n_in} dialogue rows "
                f"({n_nowords} had no words_*, {n_short} had < {self.min_assistant_words} words). "
                "An empty corpus trains a model to never speak instead of failing."
            )
        print(
            f"[train_data] {n_in} dialogue rows -> {n_out} training rows "
            f"(assistant={self.assistant_channel}; dropped {n_nowords} wordless, "
            f"{n_short} short, {n_nohost} no-host)",
            flush=True,
        )
        Dataset.from_dict(cols, features=feats).save_to_disk(self.out_dir.get_path())


def podcast_train_data(
    *,
    tag: str,
    shard_dirs: list[tk.Path],
    assistant_channel: str = "a",
    register: bool = True,
    **kwargs,
):
    """Wire the converter. Returns the dataset path."""
    job = PodcastCodesTrainData(shard_dirs=shard_dirs, assistant_channel=assistant_channel, **kwargs)
    if register:
        tk.register_output(f"podcast_train/{tag}", job.out_dir)
    return job.out_dir
