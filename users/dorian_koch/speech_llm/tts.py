from pathlib import Path
from sisyphus import Job, Task, tk
import os
import subprocess
from .common import HF_CACHE_DIR
import json
from i6_experiments.users.dorian_koch.jobs.hf import HfMergeShards


# cluster lmod is behaving wierd, just install ffmpeg ourselves...
class InstallFFmpeg(Job):
    def __init__(self, additional_options: list[str] | None = None):
        self.out_path = self.output_path("out", directory=True)
        self.additional_options = (
            additional_options if additional_options is not None else []
        )
        self.rqmt = {
            "cpu": 8,
            "mem": 8,
            "time": 1,
        }

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def add_to_env(out_path: tk.Path, env: dict[str, str]):
        env["PATH"] = f"{out_path.get()}/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = f"{out_path.get()}/lib:" + env.get(
            "LD_LIBRARY_PATH", ""
        )
        env["PKG_CONFIG_PATH"] = f"{out_path.get()}/lib/pkgconfig:" + env.get(
            "PKG_CONFIG_PATH", ""
        )

    def run(self):
        # wget https://ffmpeg.org/releases/ffmpeg-8.1.tar.xz
        subprocess.run(
            [
                "wget",
                "https://ffmpeg.org/releases/ffmpeg-8.1.tar.xz",
            ],
            check=True,
        )
        # tar -xf ffmpeg-8.1.tar.xz
        subprocess.run(
            [
                "tar",
                "-xf",
                "ffmpeg-8.1.tar.xz",
            ],
            check=True,
        )
        # cd ffmpeg-8.1
        os.chdir("ffmpeg-8.1")
        # ./configure --prefix=/path/to/output --enable-shared --disable-static --enable-pic --disable-x86asm
        subprocess.run(
            [
                "./configure",
                f"--prefix={self.out_path.get()}",
                "--enable-shared",
                "--disable-static",
                "--enable-pic",
                "--disable-x86asm",
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
        print(f"FFmpeg installed to {self.out_path.get()}")


class ChatterboxInference(Job):
    def __init__(
        self,
        *,
        venv_python_path: tk.Path,
        in_json: tk.Path | None = None,
        in_hf: tk.Path | None = None,
        speaker_dir: tk.Path,
        speaker_alias: dict[str, str] | None = None,
        with_audio_output: bool = False,
        ffmpeg_path: tk.Path | None = None,
        shard: int | None = None,
        num_shards: int | None = None,
        rqmt: dict[str, int] | None = None,
    ):
        self.in_json = in_json
        self.in_hf = in_hf
        assert (in_json is not None) ^ (in_hf is not None), (
            "Must provide exactly one of in_json or in_hf"
        )
        self.venv_python_path = venv_python_path
        self.speaker_dir = speaker_dir
        self.speaker_alias = speaker_alias if speaker_alias is not None else {}
        self.ffmpeg_path = ffmpeg_path
        self.shard = shard
        self.num_shards = num_shards
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
            }
        )

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 3
        return super().hash(d)

    @staticmethod
    def sharded(*, num_shards: int, **kwargs) -> tk.Path:
        assert "in_hf" in kwargs, "Sharding only supported for HF input"
        if num_shards == 1:
            return ChatterboxInference(**kwargs).out_hf
        assert num_shards > 1
        shards = []
        for shard in range(num_shards):
            shards.append(
                ChatterboxInference(shard=shard, num_shards=num_shards, **kwargs)
            )
        return HfMergeShards(
            shard_paths=[s.out_hf for s in shards], add_shard_to_id=True
        ).out_hf

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        this_file_path = Path(__file__).resolve()
        tts_script_path = this_file_path.parent / "chatterbox_inference.py"

        work_dir = os.path.join(os.getcwd(), "chatterbox_inference_workdir")
        os.makedirs(work_dir, exist_ok=True)

        command = [
            self.venv_python_path.get(),
            str(tts_script_path),
            "--out_hf",
            str(self.out_hf.get()),
            "--speaker_dir",
            str(self.speaker_dir.get()),
            "--speaker_alias",
            json.dumps(self.speaker_alias),
        ]
        if self.in_json is not None:
            command += ["--in_jsonl", str(self.in_json.get())]
        elif self.in_hf is not None:
            command += ["--in_hf", str(self.in_hf.get())]
        if self.out_dir is not None:
            command += ["--out_dir", str(self.out_dir.get())]
        else:
            command += ["--out_dir", str(work_dir)]
        if self.shard is not None and self.num_shards is not None:
            command += [
                "--in_hf_shard",
                str(self.shard),
                "--in_hf_num_shards",
                str(self.num_shards),
            ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HOME"] = HF_CACHE_DIR.get()
        if self.ffmpeg_path is not None:
            print(f"Adding FFmpeg from {self.ffmpeg_path.get()} to environment")
            InstallFFmpeg.add_to_env(self.ffmpeg_path, env)

        print(f"Env: {env}")

        print(
            f"Running Chatterbox inference with command: {' '.join(command)}",
            flush=True,
        )
        print(f"Using HF cache directory: {HF_CACHE_DIR}")
        subprocess.run(command, env=env, check=True)


class ParlerTTSInference(Job):
    def __init__(
        self,
        *,
        venv_python_path: tk.Path,
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

        this_file_path = Path(__file__).resolve()
        tts_script_path = this_file_path.parent / "parlertts_inference.py"

        command = [
            self.venv_python_path.get(),
            str(tts_script_path),
            "--voices_per_prompt",
            str(self.voices_per_prompt),
            "--out_dir",
            str(self.out_dir),
            *[
                item
                for desc in self.voice_descriptions
                for item in ["--voice_description", desc]
            ],
        ]
        if self.prompt is not None:
            command += ["--text", self.prompt]

        env = os.environ.copy()
        env["HF_HOME"] = HF_CACHE_DIR.get()
        env["PYTHONUNBUFFERED"] = "1"

        print(f"Running ParlerTTS inference with command: {' '.join(command)}")
        print(f"Using HF cache directory: {HF_CACHE_DIR}")
        subprocess.run(command, env=env, check=True)
