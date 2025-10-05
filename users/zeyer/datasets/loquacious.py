from __future__ import annotations
from typing import TYPE_CHECKING, Union, Any, Sequence, Dict
from functools import partial, cache
from sisyphus import tk, Path
from i6_core.datasets.huggingface import TransformAndMapHuggingFaceDatasetJob, ExtractTextFromHuggingFaceDatasetJob
from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
from i6_experiments.users.zeyer import tools_paths
from .utils.spm import SentencePieceModel

if TYPE_CHECKING:
    from datasets import DatasetDict


_alias_prefix = "datasets/Loquacious/"


def py():
    for name in ["small", "medium", "large"]:
        for q in [3, 4]:
            get_loquacious_hf_ogg(name, quality=q)
    get_train_corpus_text()
    get_spm_vocab(dim=10_240)


@cache
def get_loquacious_hf_ogg(name: str = "large", *, quality: int = 3) -> Path:
    ffmpeg_binary = tools_paths.get_ffmpeg_binary()

    job = TransformAndMapHuggingFaceDatasetJob(
        "speechbrain/LoquaciousSet",
        name,
        transform=_transform_rename_columns,
        map_func=partial(_map_func_wav_to_ogg, ffmpeg_binary=ffmpeg_binary, quality_opts=["-q", str(quality)]),
        map_opts=_map_opts,
    )
    job.rqmt.update({"cpu": 32, "time": 24, "mem": 32})
    job.add_alias(_alias_prefix + f"dataset_hf_{name}_q{quality}_ogg")
    tk.register_output(f"{_alias_prefix}dataset_hf_{name}_q{quality}_ogg", job.out_dir)
    return job.out_dir


@cache
def get_train_corpus_text() -> Path:
    job = ExtractTextFromHuggingFaceDatasetJob("speechbrain/LoquaciousSet", "large", split="train", column_name="text")
    job.add_alias(f"{_alias_prefix}train_corpus.txt.extract")
    tk.register_output(f"{_alias_prefix}train_corpus.txt.gz", job.out_text)
    return job.out_text


@cache
def get_spm_vocab(
    *, dim: Union[int, str], model_type: SentencePieceType = SentencePieceType.UNIGRAM
) -> SentencePieceModel:
    """
    Get a SentencePiece model and vocab of given dimension trained on the Loquacious transcriptions.
    """
    dim_str = str(dim)
    if isinstance(dim, str):
        # Not sure if power-of-two or just multiple-of-64, but 10240 has more 2s in it (2048*5) than 10048.
        dim = {"20k": 20_480, "10k": 10_240, "5k": 5_120, "4k": 4_096, "1k": 1_024, "512": 512, "128": 128, "64": 64}[
            dim
        ]
    assert isinstance(dim, int) and dim >= 10

    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    _spm_train_job = TrainSentencePieceJob(
        training_text=get_train_corpus_text(),
        vocab_size=dim,
        model_type=model_type,
        additional_options={
            "split_digits": True,
            "unk_id": 2,  # default is 0
            "bos_id": 1,  # default is 1
            "eos_id": 0,  # default is 2
            "train_extremely_large_corpus": True,
            "shuffle_input_sentence": True,
            "input_sentence_size": 10_000_000,
        },
    )
    _spm_train_job.rqmt.update({"time": 12, "mem": 126})  # needs much more mem, maybe little longer
    _spm_train_job.add_alias(f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train")
    tk.register_output(f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train.model", _spm_train_job.out_model)
    tk.register_output(
        f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train.vocab",
        ExtractSentencePieceVocabJob(_spm_train_job.out_model).out_vocab,
    )
    spm = SentencePieceModel(
        dim=dim,
        model_file=_spm_train_job.out_model,
        unknown_label="<unk>",
        bos_idx=1,
        eos_idx=0,
    )
    return spm


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
