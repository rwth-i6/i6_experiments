from pathlib import Path
from sisyphus import Job, Task, tk
import os
import subprocess
from .common import HF_CACHE_DIR


class ChatterboxInference(Job):
    def __init__(
        self,
        *,
        venv_python_path: tk.Path,
        in_text: tk.Path,
        speaker_dir: tk.Path,
    ):
        self.in_text = in_text
        self.venv_python_path = venv_python_path
        self.speaker_dir = speaker_dir
        self.out_dir = self.output_path("chatterbox_output", directory=True)
        self.rqmt = {
            "gpu": 1,
            "cpu": 2,
            "mem": 16,
            "time": 2,
        }

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 3
        return super().hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):

        this_file_path = Path(__file__).resolve()
        tts_script_path = this_file_path.parent / "chatterbox_inference.py"

        command = [
            self.venv_python_path.get(),
            str(tts_script_path),
            "--in_text",
            str(self.in_text.get()),
            "--out_dir",
            str(self.out_dir),
            "--speaker_dir",
            str(self.speaker_dir.get()),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HOME"] = HF_CACHE_DIR.get()

        print(f"Running Chatterbox inference with command: {' '.join(command)}")
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
            "cpu": 2,
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
