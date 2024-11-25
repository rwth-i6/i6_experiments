from dataclasses import dataclass, field
from random import Random
from typing import Dict, Iterator

from i6_core.util import uopen
import i6_experiments.common.datasets.librispeech as lbs_dataset
import numpy as np
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe_v2
from i6_experiments.common.datasets.util import CorpusObject
from i6_experiments.common.setups.serialization import Collection, Import
from sisyphus import Job, Task, tk

from ..tools import returnn_python_exe, returnn_root


def speed_perturbation(audio: np.ndarray, sample_rate: int, random_state: Random) -> np.ndarray:
    import librosa

    new_sample_rate = int(sample_rate * (1 + random_state.randint(-1, 2) * 0.1))
    if new_sample_rate != sample_rate:
        audio = librosa.core.resample(y=audio, orig_sr=sample_rate, target_sr=new_sample_rate, res_type="kaiser_fast")
    return audio


@dataclass
class TrainData:
    train_config_dict: dict
    dev_config_dict: dict
    extra_serializers: Collection = field(
        default_factory=lambda: Collection(
            [
                Import(
                    f"{__package__}.bpe.speed_perturbation",
                    use_for_hash=False,
                )
            ]
        )
    )


def get_train_data(bpe_size: int) -> TrainData:
    corpus_object_dict: Dict[str, CorpusObject] = lbs_dataset.get_corpus_object_dict(
        audio_format="wav", output_prefix="corpora"
    )

    train_corpus_file = corpus_object_dict["train-other-960"].corpus_file
    dev_clean_corpus_file = corpus_object_dict["dev-clean"].corpus_file
    dev_other_corpus_file = corpus_object_dict["dev-other"].corpus_file

    train_oggzip_job = BlissToOggZipJob(
        bliss_corpus=train_corpus_file,
        segments=SegmentCorpusJob(bliss_corpus=train_corpus_file, num_segments=200).out_segment_path,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
    )
    train_oggzip_job.rqmt = {"cpu": 1, "mem": 4, "time": 1}  # type: ignore
    train_oggzip_job.merge_rqmt = {"cpu": 1, "mem": 16, "time": 24}
    train_oggzip_file = train_oggzip_job.out_ogg_zip

    dev_clean_oggzip_file = BlissToOggZipJob(
        bliss_corpus=dev_clean_corpus_file,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
    ).out_ogg_zip

    dev_other_oggzip_file = BlissToOggZipJob(
        bliss_corpus=dev_other_corpus_file,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
    ).out_ogg_zip

    bpe_settings = get_subword_nmt_bpe_v2(corpus_key="train-other-960", bpe_size=bpe_size)
    bpe_config = {
        "class": "BytePairEncoding",
        "unknown_label": None,
        "bpe_file": bpe_settings.bpe_codes,
        "vocab_file": bpe_settings.bpe_vocab,
    }

    dev_audio_config = {
        "features": "raw",
        "peak_normalization": True,
        "preemphasis": 0.97,
    }

    train_audio_config = {
        **dev_audio_config,
        "pre_process": CodeWrapper("speed_perturbation"),
    }

    train_config_dict = {
        "class": "MetaDataset",
        "datasets": {
            "data": {
                "class": "OggZipDataset",
                "use_cache_manager": True,
                "path": [train_oggzip_file],
                "audio": train_audio_config,
                "targets": bpe_config,
                "partition_epoch": 10,
                "seq_ordering": "laplace:.1000",
            }
        },
        "data_map": {"data": ("data", "data"), "classes": ("data", "classes")},
        "seq_order_control_dataset": "data",
    }

    dev_config_dict = {
        "class": "MetaDataset",
        "datasets": {
            "data": {
                "class": "OggZipDataset",
                "use_cache_manager": True,
                "path": [dev_clean_oggzip_file, dev_other_oggzip_file],
                "audio": dev_audio_config,
                "targets": bpe_config,
                "partition_epoch": 1,
                "seq_ordering": "sorted",
            }
        },
        "data_map": {"data": ("data", "data"), "classes": ("data", "classes")},
        "seq_order_control_dataset": "data",
    }

    return TrainData(train_config_dict=train_config_dict, dev_config_dict=dev_config_dict)


def get_recog_data(corpus_name: str) -> dict:
    corpus_object_dict: Dict[str, CorpusObject] = lbs_dataset.get_corpus_object_dict(
        audio_format="wav", output_prefix="corpora"
    )

    corpus_file = corpus_object_dict[corpus_name].corpus_file

    oggzip_file = BlissToOggZipJob(
        bliss_corpus=corpus_file,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
    ).out_ogg_zip

    return {
        "class": "MetaDataset",
        "datasets": {
            "data": {
                "class": "OggZipDataset",
                "use_cache_manager": True,
                "path": [oggzip_file],
                "audio": {
                    "features": "raw",
                    "peak_normalization": True,
                    "preemphasis": 0.97,
                },
                "targets": None,
            }
        },
        "data_map": {"data": ("data", "data")},
        "seq_order_control_dataset": "data",
    }


class BPEVocabFileConversionJob(Job):
    def __init__(self, bpe_vocab_file: tk.Path) -> None:
        self.bpe_vocab_file = bpe_vocab_file
        self.out_vocab_file = self.output_path("vocab.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        with uopen(self.bpe_vocab_file) as f:
            vocab_dict = eval(f.read())
        inverse_dict = {val: key for key, val in vocab_dict.items()}
        with open(self.out_vocab_file.get(), "w") as f:
            for val in inverse_dict.values():
                f.write(f"{val}\n")

            f.write("<blank>")


def get_bpe_vocab_file(bpe_size: int) -> tk.Path:
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key="train-other-960", bpe_size=bpe_size)
    return BPEVocabFileConversionJob(bpe_vocab_file=bpe_settings.bpe_vocab).out_vocab_file
