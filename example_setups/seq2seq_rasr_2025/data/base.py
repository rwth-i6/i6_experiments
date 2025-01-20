from dataclasses import dataclass
from random import Random
import textwrap
from typing import Iterator, List, Literal, Optional, Protocol

import numpy as np
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.util import uopen
from i6_experiments.common.setups.serialization import Import
from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedFormat

from ..tools import returnn_python_exe, returnn_root


def speed_perturbation(audio: np.ndarray, sample_rate: int, random_state: Random) -> np.ndarray:
    import librosa

    new_sample_rate = int(sample_rate * (1 + random_state.randint(-1, 2) * 0.1))
    if new_sample_rate != sample_rate:
        audio = librosa.core.resample(y=audio, orig_sr=sample_rate, target_sr=new_sample_rate, res_type="kaiser_fast")
    return audio


class DataConfig(Protocol):
    def get_returnn_data(self, dataset_type: Literal["train", "dev", "forward_data"]) -> ReturnnConfig: ...


@dataclass
class OggZipDataConfig:
    bliss_corpus_files: List[tk.Path]
    speed_perturbation: bool = False
    ogg_segments: int = 1
    partition_epoch: int = 1
    seq_ordering: str = "sorted"
    target_config: Optional[dict] = None
    segment_file: Optional[tk.Path] = None

    def get_returnn_data(self, dataset_type: Literal["train", "dev", "forward_data"]) -> ReturnnConfig:
        oggzip_files = []

        for corpus_file in self.bliss_corpus_files:
            oggzip_job = BlissToOggZipJob(
                bliss_corpus=corpus_file,
                segments=(
                    SegmentCorpusJob(bliss_corpus=corpus_file, num_segments=self.ogg_segments).out_segment_path
                    if self.ogg_segments > 1
                    else None
                ),
                returnn_root=returnn_root,
                returnn_python_exe=returnn_python_exe,
            )
            oggzip_job.rqmt = {"cpu": 1, "mem": 4, "time": 1}  # type: ignore
            oggzip_job.merge_rqmt = {"cpu": 1, "mem": 16, "time": 24}
            oggzip_files.append(oggzip_job.out_ogg_zip)

        audio_config = {
            "features": "raw",
            "peak_normalization": True,
            "preemphasis": 0.97,
        }

        if self.speed_perturbation:
            audio_config["pre_process"] = CodeWrapper("speed_perturbation")

        dataset_config_dict = {
            "class": "OggZipDataset",
            "use_cache_manager": True,
            "path": oggzip_files,
            "audio": audio_config,
            "targets": self.target_config,
            "partition_epoch": self.partition_epoch,
            "seq_ordering": self.seq_ordering,
        }
        if self.segment_file is not None:
            dataset_config_dict["seq_list_filter_file"] = self.segment_file

        return ReturnnConfig(
            config={dataset_type: dataset_config_dict},
            python_prolog=(
                Import(f"{__package__}.base.speed_perturbation", use_for_hash=False)
                if self.speed_perturbation
                else None
            ),
            sort_config=False,
        )


@dataclass
class MetaOggZipDataConfig(OggZipDataConfig):
    def get_returnn_data(self, dataset_type: Literal["train", "dev", "forward_data"]) -> ReturnnConfig:
        returnn_data = super().get_returnn_data(dataset_type)

        returnn_data.config[dataset_type] = {
            "class": "MetaDataset",
            "datasets": {"data": returnn_data.config[dataset_type]},
            "seq_order_control_dataset": "data",
            "data_map": {
                "data": ("data", "data"),
            },
        }
        if self.target_config:
            returnn_data.config[dataset_type]["data_map"]["classes"] = ("data", "classes")

        return returnn_data


@dataclass
class LmDataConfig:
    corpus_file: tk.Path
    vocab_file: tk.Path
    partition_epoch: int
    seq_ordering: str

    def get_returnn_data(self, dataset_type: Literal["train", "dev", "forward_data"]) -> ReturnnConfig:
        dataset_config_dict = {
            "class": "LmDataset",
            "corpus_file": CodeWrapper(DelayedFormat('lambda: cf("{}")', self.corpus_file)),
            "orth_symbols_map_file": self.vocab_file,
            "orth_replace_map_file": "",
            "word_based": True,
            "seq_end_symbol": "</s>",
            "auto_replace_unknown_symbol": False,
            "unknown_symbol": "<unk>",
            "add_delayed_seq_data": True,
            "delayed_seq_data_start_symbol": "<s>",
            "partition_epoch": self.partition_epoch,
            "seq_ordering": self.seq_ordering,
        }
        return ReturnnConfig(
            python_prolog=textwrap.dedent(
                """\
                import os
                
                _cf_cache = {}

                def cf(filename):
                    "Cache manager"
                    from subprocess import check_output, CalledProcessError
                    if filename in _cf_cache:
                        return _cf_cache[filename]
                    if int(os.environ.get("RETURNN_DEBUG", "0")):
                        print("use local file: %s" % filename)
                        return filename  # for debugging
                    try:
                        cached_fn = check_output(["cf", filename]).strip().decode("utf8")
                    except CalledProcessError:
                        print("Cache manager: Error occurred, using local file")
                        return filename
                    assert os.path.exists(cached_fn)
                    _cf_cache[filename] = cached_fn
                    return cached_fn
                """
            ),
            config={dataset_type: dataset_config_dict},
            sort_config=False,
        )


class BPEVocabToTextFileConversionJob(Job):
    def __init__(self, bpe_vocab_file: tk.Path, extra_tokens: Optional[List[str]] = None) -> None:
        self.bpe_vocab_file = bpe_vocab_file
        self.extra_tokens = extra_tokens or []
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

            for token in self.extra_tokens:
                f.write(f"{token}\n")
