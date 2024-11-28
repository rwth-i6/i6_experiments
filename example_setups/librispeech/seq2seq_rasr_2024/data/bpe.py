from dataclasses import dataclass
from random import Random
from typing import Dict, Iterator, List, Literal, Optional

import i6_experiments.common.datasets.librispeech as lbs_dataset
import numpy as np
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.util import uopen
from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe_v2
from i6_experiments.common.datasets.util import CorpusObject
from i6_experiments.common.setups.serialization import Import
from sisyphus import Job, Task, tk

from ..tools import returnn_python_exe, returnn_root


def speed_perturbation(audio: np.ndarray, sample_rate: int, random_state: Random) -> np.ndarray:
    import librosa

    new_sample_rate = int(sample_rate * (1 + random_state.randint(-1, 2) * 0.1))
    if new_sample_rate != sample_rate:
        audio = librosa.core.resample(y=audio, orig_sr=sample_rate, target_sr=new_sample_rate, res_type="kaiser_fast")
    return audio


@dataclass
class DataConfig:
    dataset_type: Literal["train", "dev", "forward_data"]
    corpus_names: List[
        Literal[
            "train-other-960",
            "train-other-500",
            "train-clean-460",
            "train-clean-360",
            "train-clean-100",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
        ]
    ]
    speed_perturbation: bool
    ogg_segments: int
    partition_epoch: int
    seq_ordering: str
    bpe_size: Optional[int] = None
    preemphasis: float = 0.97

    def get_returnn_data(self) -> ReturnnConfig:
        corpus_object_dict: Dict[str, CorpusObject] = lbs_dataset.get_corpus_object_dict(
            audio_format="wav", output_prefix="corpora"
        )

        oggzip_files = []

        for corpus_name in self.corpus_names:
            corpus_file = corpus_object_dict[corpus_name].corpus_file

            oggzip_job = BlissToOggZipJob(
                bliss_corpus=corpus_file,
                segments=SegmentCorpusJob(bliss_corpus=corpus_file, num_segments=self.ogg_segments).out_segment_path
                if self.ogg_segments > 1
                else None,
                returnn_root=returnn_root,
                returnn_python_exe=returnn_python_exe,
            )
            oggzip_job.rqmt = {"cpu": 1, "mem": 4, "time": 1}  # type: ignore
            oggzip_job.merge_rqmt = {"cpu": 1, "mem": 16, "time": 24}
            oggzip_files.append(oggzip_job.out_ogg_zip)

        if self.bpe_size is not None:
            bpe_settings = get_subword_nmt_bpe_v2(corpus_key="train-other-960", bpe_size=self.bpe_size)
            target_config = {
                "class": "BytePairEncoding",
                "unknown_label": None,
                "bpe_file": bpe_settings.bpe_codes,
                "vocab_file": bpe_settings.bpe_vocab,
            }
        else:
            target_config = None

        audio_config = {
            "features": "raw",
            "peak_normalization": True,
            "preemphasis": self.preemphasis,
        }

        if self.speed_perturbation:
            audio_config["pre_process"] = CodeWrapper("speed_perturbation")

        dataset_config_dict = {
            "class": "MetaDataset",
            "datasets": {
                "data": {
                    "class": "OggZipDataset",
                    "use_cache_manager": True,
                    "path": oggzip_files,
                    "audio": audio_config,
                    "targets": target_config,
                    "partition_epoch": self.partition_epoch,
                    "seq_ordering": self.seq_ordering,
                }
            },
            "data_map": {"data": ("data", "data")},
            "seq_order_control_dataset": "data",
        }
        if target_config is not None:
            dataset_config_dict["data_map"]["classes"] = ("data", "classes")

        return ReturnnConfig(
            config={self.dataset_type: dataset_config_dict},
            python_prolog=Import(f"{__package__}.bpe.speed_perturbation", use_for_hash=False)
            if self.speed_perturbation
            else None,
            sort_config=False,
        )


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
