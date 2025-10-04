from __future__ import annotations
from typing import TYPE_CHECKING, Union, Any, Sequence, Dict
from functools import partial
from sisyphus import tk, Path
from i6_core.datasets.huggingface import TransformAndMapHuggingFaceDatasetJob
from i6_experiments.users.zeyer import tools_paths

if TYPE_CHECKING:
    from datasets import DatasetDict


def py():
    d = get_loquacious_hf_ogg()
    tk.register_output("datasets/loquacious_hf_ogg", d)


def get_loquacious_hf_ogg() -> Path:
    ffmpeg_binary = tools_paths.get_ffmpeg_binary()

    job = TransformAndMapHuggingFaceDatasetJob(
        "speechbrain/LoquaciousSet",
        "large",
        transform=_transform_rename_columns,
        map_func=partial(_map_func_wav_to_ogg, ffmpeg_binary=ffmpeg_binary, quality_opts=["-q", "4"]),
        map_opts=_map_opts,
        non_hashed_map_opts={"num_proc": 64},
    )
    job.rqmt.update({"cpu": 32, "time": 24, "mem": 32})
    return job.out_dir


def _transform_rename_columns(ds: DatasetDict) -> DatasetDict:
    return ds.rename_columns({"ID": "id", "wav": "audio"})


def _map_func_wav_to_ogg(
    data: Dict[str, Any], *, ffmpeg_binary: Union[str, Path], quality_opts: Sequence[str]
) -> Dict[str, Any]:
    import subprocess

    proc_res = subprocess.run(
        [
            ffmpeg_binary.get_path() if isinstance(ffmpeg_binary, Path) else ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-c:a",
            "libvorbis",
            *quality_opts,
            "-f",
            "ogg",
            "-",
        ],
        input=data["audio"]["bytes"],
        stdout=subprocess.PIPE,
        check=True,
    )
    data["audio"]["bytes"] = proc_res.stdout
    return data


def _map_opts(ds: DatasetDict) -> Dict[str, Any]:
    from datasets import Audio

    features = ds["train"].features.copy()
    audio_feat = features["audio"]
    assert isinstance(audio_feat, Audio)
    audio_feat.decode = True
    return {"features": features}
