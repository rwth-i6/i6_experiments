from pathlib import Path
from sisyphus import Job, Task, tk
import os
import subprocess
from .common import job_progress_fraction, run_worker_script
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
        rqmt: dict[str, int] | None = None,
    ):
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
                # Decodes audio through torchcodec, which needs system FFmpeg/VA libraries
                # (libva-drm.so.2). Declared as a capability, not a partition name, so the recipe
                # stays cluster-agnostic -- settings.py owns the mapping.
                "requires": ["system_ffmpeg"],
            }
        )

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 3
        if not d.get("keep_columns"):
            d.pop("keep_columns", None)  # exclude at default so pre-existing hashes are unchanged
        return super().hash(d)

    @staticmethod
    def sharded(*, num_shards: int, **kwargs) -> tk.Path:
        assert "in_hf" in kwargs, "Sharding only supported for HF input"
        if num_shards == 1:
            return ChatterboxInference(**kwargs).out_hf
        assert num_shards > 1
        shards = []
        for shard in range(num_shards):
            shards.append(ChatterboxInference(shard=shard, num_shards=num_shards, **kwargs))
        return HfMergeShards(shard_paths=[s.out_hf for s in shards], add_shard_to_id=True).out_hf

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

        env_hook = None
        if self.ffmpeg_path is not None:
            print(f"Adding FFmpeg from {self.ffmpeg_path.get()} to environment")

            def env_hook(env):
                InstallFFmpeg.add_to_env(self.ffmpeg_path, env)

        run_worker_script(
            self.venv_python_path.get(),
            tts_script_path,
            args,
            log_label="Chatterbox inference",
            env_hook=env_hook,
        )


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
