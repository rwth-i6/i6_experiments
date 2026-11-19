from pathlib import Path
from sisyphus import Job, Task, tk
import os
import subprocess
from .common import (
    merge_hf_parts,
    run_worker_script_per_gpu,
    add_torchcodec_to_env,
    job_progress_fraction,
    run_worker_script,
)
import json
from i6_experiments.users.dorian_koch.jobs.hf import HfMergeShards


# cluster lmod is behaving wierd, just install ffmpeg ourselves...
#: Build flags that make our FFmpeg **portable across nodes**.
#:
#: FFmpeg's configure AUTODETECTS optional system libraries on whatever node the build lands on and
#: links against them. Building on c23g therefore produced a binary needing libva/libva-drm (VA-API),
#: X11/XCB, ALSA and libbz2 -- none of which we use, and several of which are absent on c25g's
#: Rocky9/epyc nodes and on the login node. That is the real reason "c25g has no FFmpeg": the system
#: was never the problem, our own build was non-hermetic. (Symptom: `torchcodec` import dies with
#: `OSError: libva-drm.so.2: cannot open shared object file`.)
#:
#: `--disable-autodetect` turns that off: nothing links unless explicitly enabled. We only ever
#: decode/encode audio, and the codecs we need (PCM, WAV, FLAC, MP3, Opus/Vorbis) are all NATIVE to
#: FFmpeg -- so the portable build needs no external libraries at all beyond libc/libm/zlib.
PORTABLE_FFMPEG_OPTIONS = [
    "--disable-autodetect",  # the fix: never link a library just because the build node has it
    "--disable-doc",
    # We are an audio pipeline; dropping the display/device layers removes the X11/ALSA/VA-API
    # dependency surface entirely rather than relying on autodetect to have missed it.
    "--disable-vaapi",
    "--disable-vdpau",
    "--disable-xlib",
    "--disable-libxcb",
    "--disable-alsa",
    "--disable-sdl2",
]


class InstallFFmpeg(Job):
    """Build FFmpeg from source into a job output, so no cluster/system FFmpeg is needed.

    Consumers put it on PATH/LD_LIBRARY_PATH via :meth:`add_to_env`; that is also what lets
    `torchcodec` find libavcodec/libavutil, so jobs using it are node-independent and do NOT need a
    partition pinned for them.

    ``hash_overwrite`` exists so the build flags can be changed WITHOUT changing this job's Sisyphus
    hash -- the whole TTS/annotate chain (hundreds of GB) consumes this job's output, and a hash
    change would re-run all of it for what is only a link-time fix. To apply new flags:
    ``hpc-rerun.py <this job dir> --finished`` then restart the manager.
    """

    #: FFmpeg release built by default. NOTE 8.x has bitten us twice: its nistsphere demuxer rejects
    #: Fisher's `ulaw,embedded-shorten-v2.00` (hence InstallSph2pipe), and `torchcodec` supports only
    #: FFmpeg 4-7 -- it dlopen's `libtorchcodec_core{4..7}.so` against libavcodec 58-61, whereas 8.1
    #: ships libavcodec.so.62, so torchcodec cannot load against it. Build `version="7.1"` for any
    #: consumer that needs torchcodec.
    DEFAULT_VERSION = "8.1"

    def __init__(self, additional_options: list[str] | None = None, version: str = DEFAULT_VERSION):
        self.out_path = self.output_path("out", directory=True)
        self.additional_options = additional_options if additional_options is not None else []
        self.version = version
        self.rqmt = {
            "cpu": 8,
            "mem": 8,
            "time": 1,
        }

    @classmethod
    def hash(cls, parsed_args):
        # `version` is excluded from the hash while it equals the default, so adding the parameter
        # does not change the existing job's hash (which the whole TTS/annotate chain depends on).
        # Passing a non-default version yields a distinct job, as it should.
        args = dict(parsed_args)
        if args.get("version") == cls.DEFAULT_VERSION:
            args.pop("version")
        return super().hash(args)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def add_to_env(out_path: tk.Path, env: dict[str, str]):
        env["PATH"] = f"{out_path.get()}/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = f"{out_path.get()}/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["PKG_CONFIG_PATH"] = f"{out_path.get()}/lib/pkgconfig:" + env.get("PKG_CONFIG_PATH", "")

    def run(self):
        # wget https://ffmpeg.org/releases/ffmpeg-8.1.tar.xz
        tarball = f"ffmpeg-{self.version}.tar.xz"
        subprocess.run(
            [
                "wget",
                f"https://ffmpeg.org/releases/{tarball}",
            ],
            check=True,
        )
        # tar -xf ffmpeg-8.1.tar.xz
        subprocess.run(
            [
                "tar",
                "-xf",
                tarball,
            ],
            check=True,
        )
        os.chdir(f"ffmpeg-{self.version}")
        # ./configure --prefix=/path/to/output --enable-shared --disable-static --enable-pic --disable-x86asm
        subprocess.run(
            [
                "./configure",
                f"--prefix={self.out_path.get()}",
                "--enable-shared",
                "--disable-static",
                "--enable-pic",
                "--disable-x86asm",
                # Deliberately NOT a constructor argument: these are a build detail, not a semantic
                # input, so keeping them out of the Sisyphus hash means the flags can be changed and
                # the binary rebuilt (`hpc-rerun.py <job dir> --finished`) without re-running the
                # hundreds of GB of TTS/annotate jobs downstream that consume this output.
                *PORTABLE_FFMPEG_OPTIONS,
                *self.additional_options,
            ],
            check=True,
        )
        # make -j$(nproc)
        subprocess.run(
            [
                "make",
                "-j",
                str(self.rqmt["cpu"]),
            ],
            check=True,
        )
        # make install
        subprocess.run(
            [
                "make",
                "install",
            ],
            check=True,
        )
        # Fail loud if the build is not actually portable -- the whole point of
        # PORTABLE_FFMPEG_OPTIONS. `ldd` must show nothing beyond libc/libm/libz and our own libav*,
        # otherwise this binary will die on some other node exactly as the pre-2026-07-23 build did.
        ldd = subprocess.run(
            ["ldd", f"{self.out_path.get()}/bin/ffmpeg"],
            capture_output=True,
            text=True,
            env={**os.environ, "LD_LIBRARY_PATH": f"{self.out_path.get()}/lib"},
        ).stdout
        allowed = ("linux-vdso", "ld-linux", "libc.so", "libm.so", "libz.so", "libdl", "librt", "libpthread")
        stray = [
            line.strip()
            for line in ldd.splitlines()
            if line.strip() and self.out_path.get() not in line and not any(a in line for a in allowed)
        ]
        assert not stray, (
            f"FFmpeg build is not portable -- unexpected external deps: {stray}. "
            f"Something slipped past --disable-autodetect; add an explicit --disable-* for it."
        )
        print(f"FFmpeg {self.version} installed to {self.out_path.get()} (portable: {len(stray)} stray deps)")


class ChatterboxInference(Job):
    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        in_json: tk.Path | None = None,
        in_hf: tk.Path | None = None,
        speaker_dir: tk.Path,
        speaker_alias: dict[str, str] | None = None,
        with_audio_output: bool = False,
        ffmpeg_path: tk.Path | None = None,
        shard: int | None = None,
        num_shards: int | None = None,
        keep_columns: list[str] | None = None,
        id_prefix: str | None = None,
        rqmt: dict[str, int] | None = None,
        legacy_per_utterance: bool = False,
        language: str | None = None,
    ):
        """``legacy_per_utterance``: synthesize with upstream ``model.generate`` one utterance at a time (Perth
        watermark on, ~25x slower) instead of the batched default (``chatterbox_fast``: continuous batching,
        batched S3Gen, no watermark; same Parakeet WER and durations vs upstream, not bit-reproducible). Never part
        of the hash: a job's hash names its inputs, and both paths write the same kind of corpus from them; every
        existing hash runs the batched path from now on (guard ``check_chatterbox_fast_path``). As the hash is the same,
        Sisyphus hands every later construction of that hash the first-built object: within one graph the first
        construction decides the path.

        ``language``: a Chatterbox language id (``de``, ``zh``, ... the 23 of ``chatterbox.mtl_tts.SUPPORTED_LANGUAGES``)
        synthesizes with Chatterbox Multilingual V3 (``chatterbox_fast_mtl``, batched; ~21x upstream's one-at-a-time
        speed with the same Parakeet WER on German). Default None = Turbo (English); hash-excluded at None."""
        self.in_json = in_json
        self.in_hf = in_hf
        assert (in_json is not None) ^ (in_hf is not None), "Must provide exactly one of in_json or in_hf"
        self.venv_python_path = venv_python_path
        self.speaker_dir = speaker_dir
        self.speaker_alias = speaker_alias if speaker_alias is not None else {}
        self.ffmpeg_path = ffmpeg_path
        self.shard = shard
        self.num_shards = num_shards
        # Input columns to carry through TTS to the output (e.g. RAG ``reference_text``). Hash-excluded
        # at the default (None) so existing TTS hashes are unchanged; only an opt-in caller re-runs.
        self.keep_columns = keep_columns
        # Prepended to every output id (``f"{id_prefix}dialogue_{i}"``): ``sharded(ids_at_source=True)`` passes
        # ``shard_{s}_`` so the merge needs no id rewrite and can link instead of copy (check_zero_copy_merge).
        # Hash-excluded at the default (None): existing TTS hashes are unchanged.
        self.id_prefix = id_prefix
        self.legacy_per_utterance = legacy_per_utterance
        assert language is None or not legacy_per_utterance, "legacy_per_utterance is Turbo only"
        self.language = language
        if with_audio_output:
            self.out_dir = self.output_path("chatterbox_output", directory=True)
        else:
            self.out_dir = None
        self.out_hf = self.output_path("out_hf", directory=True)
        self.rqmt = (
            rqmt
            if rqmt is not None
            else {
                "gpu": 1,
                "cpu": 4,
                "mem": 16,
                "time": 4,
                # No `requires: ["system_ffmpeg"]` (dropped 2026-09-16). torchcodec needs three
                # things this job now ships itself: our own FFmpeg build, CUDA NPP from the venv,
                # and the base interpreter's libpython -- see the env_hook below. VERIFIED on a real
                # c25g node with neither system FFmpeg nor system NPP: torchcodec imports and a
                # `datasets` Audio() column round trips, while the same import WITHOUT those paths
                # fails on the same node. Keeping the tag pinned this to c23g, where the queue ran
                # ~1.5 days against c25g's ~2 h.
            }
        )

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 3
        if not d.get("keep_columns"):
            d.pop("keep_columns", None)  # exclude at default so pre-existing hashes are unchanged
        if not d.get("id_prefix"):
            d.pop("id_prefix", None)  # same: only ids_at_source shards hash it
        d.pop("legacy_per_utterance", None)  # the synthesis path, never the hash (see __init__)
        if not d.get("language"):
            d.pop("language", None)  # Turbo at the default: existing hashes unchanged
        return super().hash(d)

    @staticmethod
    def sharded(*, num_shards: int, ids_at_source: bool = False, **kwargs) -> tk.Path:
        """``ids_at_source`` (2026-10-16; use it for every NEW corpus): each shard writes ``shard_{s}_dialogue_{i}``
        itself, so ``HfMergeShards`` needs no id rewrite and LINKS the shards instead of copying them. The merged
        dataset is identical to the default path's (``add_shard_to_id`` makes the same ``shard_{s}_{id}``; guard
        check_zero_copy_merge). Default False keeps every existing corpus's hashes and its copying merge."""
        assert "in_hf" in kwargs, "Sharding only supported for HF input"
        if num_shards == 1:
            return ChatterboxInference(**kwargs).out_hf
        assert num_shards > 1
        shards = []
        for shard in range(num_shards):
            prefix = {"id_prefix": f"shard_{shard}_"} if ids_at_source else {}
            shards.append(ChatterboxInference(shard=shard, num_shards=num_shards, **prefix, **kwargs))
        return HfMergeShards(shard_paths=[s.out_hf for s in shards], add_shard_to_id=not ids_at_source).out_hf

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        # chatterbox_inference.py writes progress.json into the job work dir.
        return job_progress_fraction(self)

    def run(self):
        tts_script_path = Path(__file__).resolve().parent / "chatterbox_inference.py"

        work_dir = os.path.join(os.getcwd(), "chatterbox_inference_workdir")
        os.makedirs(work_dir, exist_ok=True)

        args = [
            "--out_hf",
            self.out_hf.get(),
            "--speaker_dir",
            self.speaker_dir.get(),
            "--speaker_alias",
            json.dumps(self.speaker_alias),
        ]
        if self.in_json is not None:
            args += ["--in_jsonl", self.in_json.get()]
        elif self.in_hf is not None:
            args += ["--in_hf", self.in_hf.get()]
        args += ["--out_dir", self.out_dir.get() if self.out_dir is not None else work_dir]
        if self.shard is not None and self.num_shards is not None:
            args += ["--in_hf_shard", self.shard, "--in_hf_num_shards", self.num_shards]
        if self.keep_columns:
            args += ["--keep_columns", *self.keep_columns]
        if self.id_prefix:
            args += ["--id_prefix", self.id_prefix]
        # getattr: a job pickled before this parameter existed has no such attribute (and runs batched)
        if getattr(self, "legacy_per_utterance", False):
            args += ["--legacy_per_utterance"]
        if getattr(self, "language", None):
            args += ["--language", self.language]

        # torchcodec needs BOTH our FFmpeg libs and CUDA NPP on LD_LIBRARY_PATH. Supplying both is
        # what makes this job node-independent; c23g merely happened to provide them system-wide.
        def env_hook(env):
            add_torchcodec_to_env(
                env,
                ffmpeg_path=self.ffmpeg_path,
                npp_venv_python=self.venv_python_path.get(),
                venv_python=self.venv_python_path.get(),
                who="ChatterboxInference",
            )

        # Per-GPU fan-out: each worker takes a contiguous sub-shard of this job's
        # shard and writes its own out_hf part; the parts are concatenated in order, so the merged
        # dataset is row-for-row what one worker would have written. The legacy per-dialogue
        # out_dir layout indexes dialogues by their number, which the sub-shards would collide
        # on, so a job that registers out_dir stays single-worker.
        out_hf = self.out_hf.get()

        def args_for(k, n):
            a = list(args)
            if n > 1:
                a += ["--sub_shard", k, "--sub_num_shards", n]
                a[a.index("--out_hf") + 1] = f"{out_hf}.part{k}"
                a[a.index("--out_dir") + 1] = os.path.join(work_dir, f"gpu{k}")
            return a

        n = run_worker_script_per_gpu(
            self.venv_python_path.get(),
            tts_script_path,
            args_for,
            log_label="Chatterbox inference",
            env_hook=env_hook,
            max_gpus=None if self.out_dir is None else 1,
        )
        if n > 1:
            merge_hf_parts(out_hf, [f"{out_hf}.part{k}" for k in range(n)])


class ChatterboxInferenceV2(Job):
    """Dialogue TTS for SFT data, any Chatterbox language: turns that switch language, an ASR check that regenerates
    bad segments, and word timings, all in one pass on one GPU (worker ``chatterbox_v2_inference.py``; the model-free
    logic in ``chatterbox_v2_core``).

    Input ``in_hf`` with a ``dialogue`` column (JSON): a list of turns ``{"speaker", "text" | "segments", "lang"?,
    "exaggeration"?, "pre_silence"?}`` or ``{"turns", "lang"?, "voices"?: {speaker: voice name in speaker_dir}}``. A
    turn's text may switch language inline (``Das heißt <en>to commute</en>.``); the output's ``turns.text`` has no
    tags, the languages are in ``segments``. Synthesis: Turbo for English-only dialogues (``english_model="turbo"``),
    Multilingual V3 for every other dialogue as a whole (one model per voice per dialogue), per segment in its language.

    ``annotators`` (default ``chatterbox_v2_core.DEFAULT_ANNOTATORS``: Parakeet v3, Qwen3 for zh / ja / yue): per
    language the ``check`` backend (its transcript, normalised as ``speech_text_norm``, against the segment's text:
    ``metric``: ``cer`` (default) or ``wer``) and the ``words`` backend; the job also caps the TTS process at the card minus
    16 GB for the annotators and halves its decoder batches on CUDA out of memory instead of failing. A segment above ``max_err`` (per language, ``*`` default) is
    synthesized again with a new seed, up to ``max_attempts`` in all, and the best attempt kept (attempt 0 = the seed a
    run without regeneration uses); a segment shorter than ``min_check_tokens`` reference tokens is scored but never
    regenerated (default ``chatterbox_v2_core.DEFAULT_MIN_CHECK_TOKENS``). ``pace``: ``{speaker: {lang | "*": [lo,
    hi]}}`` speaking-rate bands (words / s, characters / s for zh / ja / yue, over the sounding span); a segment of at
    least ``min_pace_units`` outside its band is regenerated like an error. ``exaggeration``: ``{speaker | "*": value or
    [lo, hi]}``, a range drawn once per speaker and dialogue (a turn's own ``exaggeration`` wins); ``mtl_decode``:
    Multilingual V3's sampling (``cfg_weight``, ``temperature``, ``min_p``, ``top_p``, ``repetition_penalty``; upstream
    defaults 0.5, 0.8, 0.05, 1.0, 1.2). Every segment's edge silence is trimmed, so the pause before a turn is the
    sampled gap only; ``stress_python_path`` (``venvs.ru_stress_venv``): Russian TTS input gets stress marks as
    upstream's tokenizer would add them (``ru_stress_worker.py``); ``cross_lang_cfg``: guidance weight for a segment
    whose language differs from its voice prompt's (upstream's advice against accent transfer: 0) (``gaps``: ``{next speaker | "*": [mean, std, lo, hi]}``, default
    ``chatterbox_v2_core.DEFAULT_GAPS``: the user's pause varies widely, the assistant's little; a turn with ``<cut/>`` is interrupted there: cut after
    its last word before the marker, the next turn overlapping it by ``cut_overlap`` seconds); every turn is levelled to ``loudness_dbfs`` (speech RMS; None: as synthesized). Output columns: ``chatterbox_v2_inference``'s docstring; ``out_stats``: per
    language error rates, regeneration shares, audio hours, wall times.

    ``tts_model="qwen3"``: every dialogue through Qwen3-TTS instead (``qwen3_tts_worker.py`` over ``qwen3_tts_fast``
    in ``qwen3_tts_python_path``, ``venvs.qwen3_tts_venv``; voice cloning from each voice wav plus its transcript
    ``<name>.txt`` beside it); ``code_switch`` ``single_pass`` (a mixed turn as one call in its main language, or with
    no language token: ``qwen3_mixed_lang="auto"``) or ``segments``; ``qwen3_slots``: its decode batch (not hashed).
    Exaggeration and the Multilingual V3 settings do not apply.

    ``parakeet_python_path``: the NeMo venv (``fdb.fdb_asr_venv()``), needed when a backend is ``parakeet``;
    ``annotate_python_path``: the annotate venv (``qwen-asr``, ``num2words``, ``cn2an``, ``whisper_normalizer``;
    ``venvs.speech_annotate_venv``), always (it scores). The config is resolved at construction, so the hash pins it
    (a later change of a module default never changes what an existing hash means). ``rqmt`` and ``chunk`` (dialogues
    per batched call) are not hashed."""

    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        annotate_python_path: tk.AbstractPath,
        in_hf: tk.Path,
        speaker_dir: tk.Path,
        parakeet_python_path: tk.AbstractPath | None = None,
        speaker_alias: dict[str, str] | None = None,
        ffmpeg_path: tk.Path | None = None,
        shard: int | None = None,
        num_shards: int | None = None,
        id_prefix: str = "",
        keep_columns: list[str] | None = None,
        default_lang: str = "en",
        english_model: str = "turbo",
        code_switch: str = "segments",
        annotators: dict | None = None,
        max_err: dict | None = None,
        metric: str = "cer",
        max_attempts: int = 3,
        min_check_tokens: int | None = None,
        segment_gap: float = 0.02,
        qwen3_asr_model: str = "Qwen/Qwen3-ASR-0.6B",
        qwen3_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        exaggeration: dict | None = None,
        mtl_decode: dict | None = None,
        pace: dict | None = None,
        min_pace_units: int | None = None,
        loudness_dbfs: float | None = -23.0,
        gaps: dict | None = None,
        cut_overlap: list | None = None,
        stress_python_path: tk.AbstractPath | None = None,
        cross_lang_cfg: float | None = None,
        tts_model: str = "chatterbox",
        qwen3_tts_python_path: tk.AbstractPath | None = None,
        qwen3_mixed_lang: str | None = None,
        qwen3_slots: int = 64,
        chunk: int = 192,
        rqmt: dict | None = None,
    ):
        from . import chatterbox_v2_core as core

        assert english_model in ("turbo", "mtl"), english_model
        assert code_switch in ("segments", "single_pass"), code_switch
        assert tts_model in ("chatterbox", "qwen3"), tts_model
        assert tts_model == "chatterbox" or qwen3_tts_python_path is not None, "tts_model qwen3 needs qwen3_tts_python_path"
        annotators = annotators if annotators is not None else core.DEFAULT_ANNOTATORS
        core.check_annotators(annotators)
        uses = {b for r in annotators.values() for b in r.values()}
        assert "parakeet" not in uses or parakeet_python_path is not None, "a parakeet annotator needs parakeet_python_path"
        max_err = max_err if max_err is not None else core.DEFAULT_MAX_ERR
        assert "*" in max_err and max_attempts >= 1 and metric in ("cer", "wer")
        self.venv_python_path = venv_python_path
        self.annotate_python_path = annotate_python_path
        self.parakeet_python_path = parakeet_python_path
        self.stress_python_path = stress_python_path
        self.in_hf = in_hf
        self.speaker_dir = speaker_dir
        self.speaker_alias = speaker_alias or {}
        self.ffmpeg_path = ffmpeg_path
        self.shard, self.num_shards = shard, num_shards
        self.id_prefix = id_prefix
        self.keep_columns = keep_columns or []
        self.config = {
            "default_lang": default_lang,
            "english_model": english_model,
            "code_switch": code_switch,
            "annotators": {k: dict(v) for k, v in sorted(annotators.items())},
            "max_err": dict(sorted(max_err.items())),
            "metric": metric,
            "max_attempts": int(max_attempts),
            "min_check_tokens": int(core.DEFAULT_MIN_CHECK_TOKENS if min_check_tokens is None else min_check_tokens),
            "segment_gap": float(segment_gap),
            "qwen3_asr_model": qwen3_asr_model,
            "qwen3_aligner_model": qwen3_aligner_model,
            "exaggeration": dict(sorted((exaggeration or {"*": 0.5}).items())),
            "mtl_decode": dict(sorted((mtl_decode or {}).items())),
            "pace": {r: dict(sorted(b.items())) for r, b in sorted((pace or core.DEFAULT_PACE).items())},
            "min_pace_units": int(core.DEFAULT_MIN_PACE_UNITS if min_pace_units is None else min_pace_units),
            "loudness_dbfs": None if loudness_dbfs is None else float(loudness_dbfs),
            "gaps": {k: [float(x) for x in v] for k, v in sorted((gaps or core.DEFAULT_GAPS).items())},
            "cut_overlap": [float(x) for x in (cut_overlap or core.DEFAULT_CUT_OVERLAP)],
            "ru_stress": stress_python_path is not None,
            "cross_lang_cfg": None if cross_lang_cfg is None else float(cross_lang_cfg),
        }
        self.qwen3_tts_python_path = qwen3_tts_python_path
        self.qwen3_slots = qwen3_slots
        if tts_model == "qwen3":  # keys only for Qwen3 runs: every Chatterbox job keeps its hash
            self.config["tts_model"] = "qwen3"
            self.config["qwen3_mixed_lang"] = qwen3_mixed_lang
        self.chunk = chunk
        self.out_hf = self.output_path("out_hf", directory=True)
        self.out_stats = self.output_path("stats.json")
        self.rqmt = rqmt if rqmt is not None else {"gpu": 1, "cpu": 4, "mem": 32, "time": 4}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d.pop("rqmt", None)
        d.pop("chunk", None)
        d.pop("qwen3_slots", None)
        for k in ("tts_model", "qwen3_tts_python_path", "qwen3_mixed_lang"):  # in the config / absent: old hashes
            if d.get(k) in (None, "chatterbox"):
                d.pop(k, None)
        return super().hash(d)

    @staticmethod
    def sharded(*, num_shards: int, **kwargs) -> tk.Path:
        """``num_shards`` jobs (ids ``shard_{s}_dialogue_{i}`` written at the source), merged by linking."""
        if num_shards == 1:
            return ChatterboxInferenceV2(**kwargs).out_hf
        shards = [ChatterboxInferenceV2(shard=s, num_shards=num_shards, id_prefix=f"shard_{s}_", **kwargs)
                  for s in range(num_shards)]  # fmt: skip
        return HfMergeShards(shard_paths=[s.out_hf for s in shards], add_shard_to_id=False).out_hf

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def run(self):
        import shutil
        import tempfile

        # the dataset builder's cache: node-local disk (SLURM's per-job $TMPDIR), never the job dir (audio goes
        # straight into the arrow file; no per-dialogue files anywhere)
        scratch = tempfile.mkdtemp(prefix="cbx_v2_", dir=os.environ.get("TMPDIR"))
        try:
            self._run(scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _run(self, scratch: str):
        import importlib.util

        from . import chatterbox_v2_core as core

        here = Path(__file__).resolve().parent
        cfg = {**self.config, "chunk": self.chunk, "qwen3_slots": self.qwen3_slots}
        args = ["--in_hf", self.in_hf.get(), "--out_hf", self.out_hf.get(), "--scratch_dir", scratch,
                "--stats_json", f"{self.out_stats.get()}.part0", "--speaker_directory", self.speaker_dir.get(),
                "--speaker_alias", json.dumps(self.speaker_alias), "--config", json.dumps(cfg),
                "--annotate_python", self.annotate_python_path.get(),
                "--annotate_worker", str(here / "speech_annotate_worker.py")]  # fmt: skip
        if self.parakeet_python_path is not None:
            root = importlib.util.find_spec("speech_llm").submodule_search_locations[0]  # imports nothing
            worker = os.path.join(root, "full_duplex", "moshi_family", "rl", "parakeet_worker.py")
            assert os.path.exists(worker), worker
            args += ["--parakeet_python", self.parakeet_python_path.get(), "--parakeet_worker", worker]
        if self.stress_python_path is not None:
            args += ["--stress_python", self.stress_python_path.get(), "--stress_worker", str(here / "ru_stress_worker.py")]
        if self.qwen3_tts_python_path is not None:
            args += ["--qwen3_tts_python", self.qwen3_tts_python_path.get(),
                     "--qwen3_tts_worker", str(here / "qwen3_tts_worker.py")]  # fmt: skip
        if self.shard is not None:
            args += ["--in_hf_shard", self.shard, "--in_hf_num_shards", self.num_shards]
        if self.keep_columns:
            args += ["--keep_columns", *self.keep_columns]
        if self.id_prefix:
            args += ["--id_prefix", self.id_prefix]

        def env_hook(env):
            add_torchcodec_to_env(env, ffmpeg_path=self.ffmpeg_path, npp_venv_python=self.venv_python_path.get(),
                                  venv_python=self.venv_python_path.get(), who="ChatterboxInferenceV2")  # fmt: skip

        out_hf, stats = self.out_hf.get(), self.out_stats.get()

        def args_for(k, n):
            a = list(args)
            a[a.index("--stats_json") + 1] = f"{stats}.part{k}"
            if n > 1:
                a += ["--sub_shard", k, "--sub_num_shards", n]
                a[a.index("--out_hf") + 1] = f"{out_hf}.part{k}"
            return a

        n = run_worker_script_per_gpu(self.venv_python_path.get(), here / "chatterbox_v2_inference.py", args_for,
                                      log_label="ChatterboxInferenceV2", env_hook=env_hook)  # fmt: skip
        if n > 1:
            merge_hf_parts(out_hf, [f"{out_hf}.part{k}" for k in range(n)])
        parts = [json.load(open(f"{stats}.part{k}")) for k in range(n)]
        raw = core.stats_merge([p["raw"] for p in parts])
        with open(stats, "w") as f:
            json.dump({"summary": core.stats_summary(raw), "raw": raw, "workers": n,
                       "peak_tts_gpu_gb": max(p["peak_tts_gpu_gb"] for p in parts)}, f, indent=1)  # fmt: skip


class ParlerTTSInference(Job):
    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        prompt: str | None = None,
        voices_per_prompt: int = 5,
        voice_descriptions: list[str] | None = None,
    ):
        self.venv_python_path = venv_python_path
        self.out_dir = self.output_path("parlertts_output", directory=True)
        self.voices_per_prompt = voices_per_prompt
        self.prompt = prompt
        self.voice_descriptions = (
            voice_descriptions
            if voice_descriptions is not None
            else [
                "A male speaker delivers a slightly expressive and animated speech with a moderate speed and deep pitch. "
                "The recording is of very high quality, with the speaker's voice sounding clear and very close up, "
                "recorded in a soundproof studio with zero background noise."
            ]
        )
        self.rqmt = {
            "gpu": 1,
            "cpu": 4,
            "mem": 16,
            "time": 2,
        }

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 1
        return super().hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def get_filenames(self):
        for prompt_idx in range(len(self.voice_descriptions)):
            for i in range(self.voices_per_prompt):
                output_filename = f"prompt_{prompt_idx}_voice_{i}.wav"
                yield output_filename

    def run(self):
        tts_script_path = Path(__file__).resolve().parent / "parlertts_inference.py"

        args = ["--voices_per_prompt", self.voices_per_prompt, "--out_dir", self.out_dir]
        for desc in self.voice_descriptions:
            args += ["--voice_description", desc]
        if self.prompt is not None:
            args += ["--text", self.prompt]

        run_worker_script(
            self.venv_python_path.get(),
            tts_script_path,
            args,
            log_label="ParlerTTS inference",
        )
