"""
Which vocab entries does a dataset actually use?

:class:`ExtractVocabUsageJob` reproduces
``returnn/tools/vocab_usage.py --type null --key text --endseq -1 --dump_vocab_usage <name>``
as it was driven by
``~/experiments/2025_11_06_speech_llm/dataset_stats/vocab_usage/loquacious/dump_vocab_usage.sh``,
i.e.:

- iterate every sequence of the dataset (``--endseq -1``),
- apply the config's ``max_seq_length`` / ``min_seq_length``, and SKIP filtered sequences
  (the reference tool only reaches ``vocab_usage[data] = True`` in the non-filtered branch,
  so its numbers are conditioned on the audio-length filter -- for Loquacious that is
  ``max_seq_length = {"audio": 312000.0}``, the same 19.5s filter the trainings use),
- mark every id occurring in ``data[key]`` as used,
- report ``sum(used)`` out of ``vocab_size``.

Differences from the reference tool, all additive:

- ``vocab_size`` is a parameter instead of the hard-coded 151646 (the tool has a
  ``# TODO: do not hard code`` there).
- Per-id COUNTS are collected, not just a boolean mask. The mask is recoverable as
  ``counts > 0``, and counts additionally allow a frequency cutoff when building a restricted
  vocab (a token seen 3 times in 9.5M utterances is arguably noise, not vocabulary).
- The used ids are written out sorted, plus scalar Variables, so downstream jobs can consume the
  result without loading numpy arrays.

MOTIVATION: for Loquacious + the Qwen2 tokenizer, only a small part of the 151646-entry LLM vocab
is ever produced by the ASR transcripts. Training an AED with the full vocab would pay for a
151646 x 1024 embedding and output projection (~155M parameters each, i.e. more than half of the
whole 561M model) that is mostly dead weight.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sisyphus import Job, Task, tk

# Module level ON PURPOSE, do not move into create_files(). Sisyphus chdir's into <job>/work
# before running a task (sisyphus/task.py:177), and the recipe import paths are RELATIVE
# (gs.IMPORT_PATHS = ["config", "recipe", "recipe/"]), resolved at import time. A lazy import of
# this inside create_files therefore fails with "No module named 'i6_core'" whenever i6_core is
# not already in sys.modules -- which is exactly what killed the dev/test jobs (the train jobs
# happened to survive it, i.e. it is state-dependent and flaky, not deterministic).
# At module level this runs at graph-build time, while cwd is still the setup root.
from i6_experiments.users.zeyer.serialization_v2 import ReturnnConfigWithNewSerialization


class ExtractVocabUsageJob(Job):
    """
    Count, per vocab id, how often it occurs in ``dataset[key]``.

    :param dataset: RETURNN dataset dict. Its ``key`` stream must be sparse (label ids).
        May contain arbitrary Python objects (functions, ``functools.partial``) -- see the
        serialization note below.
    :param vocab_size: size of the vocab, i.e. the length of the output arrays.
    :param key: which data key to inspect (Loquacious: ``"text"``).
    :param extra_config: extra RETURNN global config entries. ``max_seq_length`` /
        ``min_seq_length`` belong here if the usage should be conditioned on them (see the module
        docstring).
    :param post_dataset: merged into ``dataset`` but excluded from the hash, for things like
        worker counts that must not change the result.
    :param returnn_root: inserted into ``sys.path`` for the RETURNN import.

    SERIALIZATION: the config is written with
    :class:`i6_experiments.users.zeyer.serialization_v2.ReturnnConfigWithNewSerialization`, NOT
    plain :class:`ReturnnConfig`. Plain ReturnnConfig ``repr()``s a callable as
    ``<function ... at 0x...>`` (invalid Python -- black then fails to parse the config) or, if
    the surrounding value is "unreadable", routes it through ``json.dumps``, which raises
    ``TypeError: Object of type function is not JSON serializable``.
    Both matter here: the Loquacious train dataset is a ``DistributeFilesDataset`` whose ``files``
    and ``get_sub_epoch_dataset`` are ``functools.partial``s (unavoidable -- the train split is
    396 GB over 856 shards, so a plain HuggingFaceDataset with file caching is not an option),
    and the lowercasing hook is a function too. The v2 serializer emits real ``import`` statements
    for such objects, which is exactly how the reference vocab-usage configs were produced.
    """

    __sis_version__ = 2

    def __init__(
        self,
        *,
        dataset: Dict[str, Any],
        vocab_size: int,
        key: str = "text",
        extra_config: Optional[Dict[str, Any]] = None,
        post_dataset: Optional[Dict[str, Any]] = None,
        returnn_root: Optional[tk.Path] = None,
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.post_dataset = post_dataset
        self.vocab_size = vocab_size
        self.key = key
        self.extra_config = extra_config
        self.returnn_root = returnn_root

        self.out_returnn_config_file = self.output_path("returnn.config")
        # per-id occurrence counts, int64 [vocab_size]; the boolean "used" mask is counts > 0
        self.out_counts = self.output_path("vocab_counts.npy")
        # bool [vocab_size], the exact equivalent of the reference tool's dump
        self.out_usage_mask = self.output_path("vocab_usage.npy")
        # sorted used ids, one per line -- the input for building a restricted vocab
        self.out_used_ids = self.output_path("used_ids.txt")
        self.out_num_used = self.output_var("num_used")
        self.out_num_seqs = self.output_var("num_seqs")
        self.out_num_seqs_filtered = self.output_var("num_seqs_filtered")
        self.out_num_tokens = self.output_var("num_tokens")
        self.out_stats = self.output_path("stats.txt")

        # Defaults sized for a small split. The train split needs much more (9.5M seqs, ogg
        # decoding); callers override via rqmt.
        self.rqmt = {"gpu": 0, "cpu": 2, "mem": 8, "time": 8}
        if rqmt:
            self.rqmt.update(rqmt)

    @classmethod
    def hash(cls, parsed_args):
        """hash"""
        parsed_args = parsed_args.copy()
        parsed_args.pop("post_dataset")
        parsed_args.pop("rqmt")  # resources cannot change the counts
        return super().hash(parsed_args)

    def tasks(self):
        """tasks"""
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        """create files"""
        config_dict = dict(self.extra_config or {})
        assert "dataset" not in config_dict
        dataset_dict = self.dataset.copy()
        if self.post_dataset:
            # Not part of the hash anymore, so merge only now.
            dataset_dict.update(self.post_dataset)
        config_dict["dataset"] = dataset_dict
        # See the SERIALIZATION note in the class docstring for why this is not plain ReturnnConfig.
        ReturnnConfigWithNewSerialization(config_dict).write(self.out_returnn_config_file.get_path())

    def run(self):
        """run"""
        import sys
        import time

        if self.returnn_root is not None:
            sys.path.insert(0, self.returnn_root.get_path())

        import numpy

        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset
        from returnn.log import log
        from returnn.util.basic import NumbersDict, hms

        config = Config()
        config.load_file(self.out_returnn_config_file.get_path())
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 4
        log.init_by_config(config)

        dataset_dict = config.typed_value("dataset")
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)
        dataset.init_seq_order(epoch=1)

        # Same filter semantics as the reference tool: a sequence violating any bound is skipped
        # entirely, so it contributes nothing to the usage counts.
        max_seq_length = NumbersDict(config.typed_value("max_seq_length", None) or config.int("max_seq_length", 0))
        min_seq_length = NumbersDict(config.typed_value("min_seq_length", None) or config.int("min_seq_length", 0))

        counts = numpy.zeros((self.vocab_size,), dtype=numpy.int64)
        num_seqs = 0
        num_seqs_filtered = 0
        start_time = time.time()

        seq_idx = 0
        while dataset.is_less_than_num_seqs(seq_idx):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            seq_len = dataset.get_seq_length(seq_idx)

            filtered = False
            if max_seq_length or min_seq_length:
                for k in dataset.get_data_keys():
                    if max_seq_length.has_value_for(k) and seq_len[k] > max_seq_length[k]:
                        filtered = True
                    if min_seq_length.has_value_for(k) and seq_len[k] < min_seq_length[k]:
                        filtered = True

            if filtered:
                num_seqs_filtered += 1
            else:
                data = dataset.get_data(seq_idx, self.key).reshape(-1)
                assert data.min() >= 0 and data.max() < self.vocab_size, (
                    f"{self}: id out of range in seq {dataset.get_tag(seq_idx)}:"
                    f" [{data.min()}, {data.max()}] vs vocab_size {self.vocab_size}"
                )
                # numpy.add.at is the scatter-add that handles repeated ids correctly
                # (counts[data] += 1 would count a repeated id only once).
                numpy.add.at(counts, data, 1)
                num_seqs += 1

            seq_idx += 1
            if seq_idx % 100_000 == 0:
                print(
                    f"seq {seq_idx}, {int((counts > 0).sum())} vocab entries used so far,"
                    f" elapsed {hms(time.time() - start_time)}",
                    file=log.v3,
                )

        usage_mask = counts > 0
        num_used = int(usage_mask.sum())
        num_tokens = int(counts.sum())

        print(f"vocab usage: {num_used} out of {self.vocab_size}", file=log.v1)

        numpy.save(self.out_counts.get_path(), counts)
        numpy.save(self.out_usage_mask.get_path(), usage_mask)
        used_ids = numpy.flatnonzero(usage_mask)
        with open(self.out_used_ids.get_path(), "w") as f:
            for i in used_ids:
                f.write(f"{int(i)}\n")

        self.out_num_used.set(num_used)
        self.out_num_seqs.set(num_seqs)
        self.out_num_seqs_filtered.set(num_seqs_filtered)
        self.out_num_tokens.set(num_tokens)

        lines = [
            f"vocab_size: {self.vocab_size}",
            f"used: {num_used} ({100.0 * num_used / self.vocab_size:.3f}%)",
            f"unused: {self.vocab_size - num_used}",
            f"num_seqs (counted): {num_seqs}",
            f"num_seqs (filtered out): {num_seqs_filtered}",
            f"num_tokens: {num_tokens}",
            f"tokens/seq: {num_tokens / max(num_seqs, 1):.3f}",
        ]
        # A frequency cutoff is the natural knob if the tail turns out to be junk; report it so the
        # decision can be made from the job output instead of by loading the arrays.
        nonzero = counts[usage_mask]
        for thresh in (1, 2, 5, 10, 100, 1_000):
            lines.append(f"ids with count >= {thresh}: {int((nonzero >= thresh).sum())}")
        with open(self.out_stats.get_path(), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines), file=log.v1)
