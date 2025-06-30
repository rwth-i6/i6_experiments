from dataclasses import dataclass
from functools import lru_cache
from random import Random
import textwrap
from typing import Dict, Iterator, List, Literal, Optional, Protocol, Set

import numpy as np
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.lib.lexicon import Lexicon
from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.returnn.hdf import get_returnn_simple_hdf_writer
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.util import uopen, write_xml
from i6_core.lib.corpus import Corpus
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
class HdfDataConfig:
    files: List[tk.Path]

    def get_returnn_data(self, dataset_type: Literal["train", "dev", "forward_data"]) -> ReturnnConfig:
        dataset_config_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": self.files,
        }

        return ReturnnConfig(
            config={dataset_type: dataset_config_dict},
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
class MetaOggZipHdfTargetDataConfig(DataConfig):
    oggzip_config: OggZipDataConfig
    oggzip_target_name: Optional[str]
    hdf_config: HdfDataConfig
    hdf_target_name: str

    def get_returnn_data(self, dataset_type: Literal["train", "dev", "forward_data"]) -> ReturnnConfig:
        oggzip_data = self.oggzip_config.get_returnn_data(dataset_type)
        hdf_data_dict = self.hdf_config.get_returnn_data(dataset_type).config[dataset_type]

        oggzip_data.config[dataset_type] = {
            "class": "MetaDataset",
            "datasets": {
                "data": oggzip_data.config[dataset_type],
                self.hdf_target_name: hdf_data_dict,
            },
            "seq_order_control_dataset": "data",
            "data_map": {
                "data": ("data", "data"),
                self.hdf_target_name: (self.hdf_target_name, "data"),
            },
        }
        if self.oggzip_target_name is not None:
            oggzip_data.config[dataset_type]["data_map"][self.oggzip_target_name] = ("data", "classes")

        return oggzip_data


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
            "auto_replace_unknown_symbol": True,
            "unknown_symbol": "<UNK>",
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


class RemoveWordsFromTranscriptionsJob(Job):
    def __init__(self, corpus_file: tk.Path, remove_words: List[str]) -> None:
        self.corpus_file = corpus_file
        self.remove_words = remove_words

        self.out_corpus_file = self.output_path("corpus.xml.gz")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        corpus = Corpus()
        corpus.load(self.corpus_file.get())

        def remove_words(c: Corpus) -> None:
            for rec in c.recordings:
                for seg in rec.segments:
                    seg.orth = " ".join(
                        filter(lambda symbol: symbol not in self.remove_words, (seg.orth or "").split())
                    )
            for subcorpus in c.subcorpora:
                remove_words(subcorpus)

        remove_words(corpus)

        corpus.dump(self.out_corpus_file.get())


class RemoveSpecialLemmasFromLexiconJob(Job):
    def __init__(self, lexicon_file: tk.Path) -> None:
        self.lexicon_file = lexicon_file
        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self) -> None:
        lexicon = Lexicon()
        lexicon.load(self.lexicon_file.get_path())

        remaining_lemmas = []
        for lemma in lexicon.lemmata:
            if lemma.special is None:
                remaining_lemmas.append(lemma)

        lexicon.lemmata = remaining_lemmas

        write_xml(self.out_lexicon.get_path(), lexicon.to_xml())


class BlissCorpusToTargetHdfJob(Job):
    """
    Use a bliss lexicon to convert all words in a bliss corpus into their phoneme representation
    and write these targets to an HDF file.

    Currently only supports picking the first phoneme.
    """

    def __init__(
        self,
        bliss_corpus: tk.Path,
        bliss_lexicon: tk.Path,
        returnn_root: tk.Path,
        segment_file: Optional[tk.Path] = None,
        word_separation_orth: Optional[str] = None,
        dim: Optional[int] = None,
    ):
        """
        :param bliss_corpus: path to a bliss corpus xml
        :param bliss_lexicon: path to a bliss lexicon file
        :param str|None word_separation_orth: a default word separation lemma orth. The corresponding phoneme
            (or phonemes in some special cases) are inserted between each word.
            Usually it makes sense to use something like "[SILENCE]" or "[space]" or so).
        """
        self.bliss_corpus = bliss_corpus
        self.bliss_lexicon = bliss_lexicon
        self.word_separation_orth = word_separation_orth
        self.segment_file = segment_file
        self.dim = dim

        self.returnn_root = returnn_root

        self.out_hdf = self.output_path("targets.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    @staticmethod
    def _create_target_lookup_dict(lex: Lexicon) -> Dict[str, List[int]]:
        # Mapping from phoneme symbol to target index. E.g. {"[SILENCE]": 0, "a": 1, "b": 2, ...}
        phoneme_indices = dict(zip(lex.phonemes.keys(), range(len(lex.phonemes))))

        # build lookup dict of word to target sequence
        lookup_dict: Dict[str, List[int]] = {}
        for lemma in lex.lemmata:
            for orth in lemma.orth:
                if not orth:
                    continue
                if not lemma.phon:
                    continue
                phon = lemma.phon[0]
                lookup_dict[orth] = [phoneme_indices[p] for p in phon.split()]

        return lookup_dict

    @staticmethod
    def _get_unknown_lemma(lex: Lexicon) -> str:
        for lemma in lex.lemmata:
            if lemma.special == "unknown":
                return lemma.orth[0]
        return ""

    @lru_cache
    def _get_segment_whitelist(self) -> Optional[Set[bytes]]:
        # Create whitelist of allowed segments
        if self.segment_file is None:
            return None
        with uopen(self.segment_file, "rt") as f:
            segments_whitelist = set(line.strip() for line in f.readlines() if len(line.strip()) > 0)
        return segments_whitelist

    def _segment_allowed(self, segment_name: str) -> bool:
        whitelist = self._get_segment_whitelist()
        if whitelist is None:
            return True
        return segment_name in whitelist

    def run(self):
        lex = Lexicon()
        lex.load(self.bliss_lexicon.get_path())

        lookup_dict = self._create_target_lookup_dict(lex)
        unknown_lemma = self._get_unknown_lemma(lex)

        if self.word_separation_orth is not None:
            word_separation_targets = lookup_dict[self.word_separation_orth]
            print(
                f"using word separation symbol: {self.word_separation_orth} mapped to targets {word_separation_targets}"
            )
        else:
            word_separation_targets = []

        # Create hdf writer
        out_hdf_writer = get_returnn_simple_hdf_writer(self.returnn_root.get())(
            filename=self.out_hdf, dim=self.dim, ndim=1
        )

        # Load corpus
        c = Corpus()
        c.load(self.bliss_corpus.get_path())

        # Iterate over corpus segments
        for segment in c.segments():
            # Skip disallowed segments
            if not self._segment_allowed(segment.fullname()):
                continue

            assert segment.orth is not None

            # Create list of targets for each word in the orth
            word_targets = []
            for word in segment.orth.split():
                if word in lookup_dict:
                    word_targets.append(lookup_dict[word])
                else:
                    word_targets.append(lookup_dict[unknown_lemma])
            assert len(word_targets) > 0

            # Concatenate all word target lists with the separator targets inserted in between
            segment_targets: List[int] = []
            for word in word_targets[:-1]:
                segment_targets.extend(word)
                segment_targets.extend(word_separation_targets)
            segment_targets.extend(word_targets[-1])

            # Write target sequence into hdf
            out_hdf_writer.insert_batch(
                inputs=np.array(segment_targets).reshape((1, -1)),
                seq_len=[len(segment_targets)],
                seq_tag=[segment.fullname()],
            )
        out_hdf_writer.close()
