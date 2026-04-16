__all__ = [
    "TextFileToTargetHdfJob",
]

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from i6_core.lib import lexicon
from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.util import uopen
from sisyphus import Job, Task, setup_path, tk

assert __package__ is not None
Path = setup_path(__package__)


class TextFileToTargetHdfJob(Job):
    """
    Convert a plain text file (one sequence per line) into multiple target HDF files using a bliss lexicon.

    * Input: arbitrary text file, one sequence per line, whitespace-separated tokens
    * Lexicon parsing is identical to BlissCorpusToTargetHdfJob._create_target_lookup_dict
    * Each line is mapped to a sequence of phoneme indices
    * Lines that contain unknown tokens are skipped and written to a separate text file
    * Work is sharded across `num_splits` subtasks, each writing its own HDF + unknown-lines file
    * seq_tag is the (1-based) line number in the original file, as string
    """

    def __init__(
        self,
        text_file: tk.Path,
        bliss_lexicon: tk.Path,
        returnn_root: tk.Path,
        word_separation_orth: Optional[str] = None,
        dim: Optional[int] = None,
        num_splits: int = 1,
    ):
        """
        :param text_file: path to a plain text file, one sequence per line
        :param bliss_lexicon: path to a bliss lexicon file
        :param word_separation_orth: same semantics as in BlissCorpusToTargetHdfJob
        :param dim: target dimension for HDF writer (usually len(lex.phonemes))
        :param num_splits: number of HDF shards / parallel subtasks
        """
        self.text_file = text_file
        self.bliss_lexicon = bliss_lexicon
        self.returnn_root = returnn_root
        self.word_separation_orth = word_separation_orth
        self.dim = dim

        assert num_splits >= 1
        self.num_splits = num_splits
        self.rqmt = {"mem": 8, "time": 2}

        # Outputs: one HDF + one unknown-lines file per shard
        if self.num_splits == 1:
            # Backwards-compatible naming if you just want a single file
            self.out_hdf = self.output_path("targets.hdf")
            self.out_unknown_txt = self.output_path("unknown_lines.txt")
            self.out_hdfs: List[tk.Path] = [self.out_hdf]
            self.out_unknown_txts: List[tk.Path] = [self.out_unknown_txt]
        else:
            self.out_hdfs = [
                self.output_path(f"targets.{i}.hdf") for i in range(self.num_splits)
            ]
            self.out_unknown_txts = [
                self.output_path(f"unknown_lines.{i}.txt") for i in range(self.num_splits)
            ]
            # for convenience if someone still uses .out_hdf when num_splits>1
            self.out_hdf = self.out_hdfs[0]
            self.out_unknown_txt = self.out_unknown_txts[0]

    def tasks(self):
        if self.num_splits == 1:
            # simple, single-process case
            yield Task("run_single", resume="run_single", mini_task=False, rqmt=self.rqmt)
        else:
            # `args` will be passed to run_shard(shard_idx)
            # parallel=len(args) => as many parallel subtasks as shards
            shard_indices = list(range(self.num_splits))
            yield Task(
                "run_shard",
                resume="run_shard",
                args=shard_indices,
                mini_task=False,
                rqmt=self.rqmt,
            )

    # ---- shared helpers -------------------------------------------------

    @staticmethod
    def _create_target_lookup_dict(lex: lexicon.Lexicon) -> Dict[str, List[int]]:
        """
        Identical lexicon parsing as in BlissCorpusToTargetHdfJob.
        """
        phoneme_indices = dict(zip(lex.phonemes.keys(), range(len(lex.phonemes))))

        lookup_dict: Dict[str, List[int]] = {}
        for lemma in lex.lemmata:
            for orth in lemma.orth:
                if not orth:
                    continue
                if len(lemma.phon) > 0:
                    phon = lemma.phon[0]
                else:
                    phon = ""
                lookup_dict[orth] = [phoneme_indices[p] for p in phon.split()]

        return lookup_dict

    def _load_lexicon_and_lookup(self):
        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())
        lookup_dict = self._create_target_lookup_dict(lex)

        if self.word_separation_orth is not None:
            if self.word_separation_orth not in lookup_dict:
                raise KeyError(
                    f"word_separation_orth={self.word_separation_orth!r} "
                    f"not found in lexicon orthography"
                )
            word_separation_targets = lookup_dict[self.word_separation_orth]
            print(
                f"using word separation symbol: {self.word_separation_orth} "
                f"mapped to targets {word_separation_targets}"
            )
        else:
            word_separation_targets = []

        return lex, lookup_dict, word_separation_targets

    # ---- execution ------------------------------------------------------

    def run_single(self):
        """
        Fallback: no parallelism, just reuse the same shard logic with num_splits=1.
        """
        self._process_shard(
            shard_idx=0,
            num_splits=1,
            out_hdf_path=self.out_hdf,
            unknown_txt_path=self.out_unknown_txt,
        )

    def run_shard(self, shard_idx: int):
        """
        Parallel shard entry point.

        :param shard_idx: integer in [0, num_splits-1]
        """
        assert 0 <= shard_idx < self.num_splits
        out_hdf_path = self.out_hdfs[shard_idx]
        unknown_txt_path = self.out_unknown_txts[shard_idx]

        self._process_shard(
            shard_idx=shard_idx,
            num_splits=self.num_splits,
            out_hdf_path=out_hdf_path,
            unknown_txt_path=unknown_txt_path,
        )

    def _process_shard(
        self,
        shard_idx: int,
        num_splits: int,
        out_hdf_path: tk.Path,
        unknown_txt_path: tk.Path,
    ):
        """
        Actual implementation: processes the full text file, but only keeps
        lines whose (1-based) line index satisfies (line_idx - 1) % num_splits == shard_idx.
        """

        print(f"[TextFileToTargetHdfJob] Starting shard {shard_idx+1}/{num_splits}")

        # Load lexicon and lookup
        lex, lookup_dict, word_separation_targets = self._load_lexicon_and_lookup()

        # Create HDF writer for this shard
        out_hdf_writer = get_returnn_simple_hdf_writer(self.returnn_root.get())(
            filename=out_hdf_path,
            dim=self.dim,
            ndim=1,
        )

        # Open unknown-lines file for this shard
        with uopen(unknown_txt_path, "wt") as unknown_f, uopen(self.text_file, "rt") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # skip empty lines consistently

                # Shard assignment: simple round-robin by line index
                # (line_idx-1) so shard 0 gets lines 1, 1+num_splits, ...
                if (line_idx - 1) % num_splits != shard_idx:
                    continue

                tokens = line.split()
                if not tokens:
                    continue

                # Map tokens to target sequences; if any token unknown, skip line
                word_targets: List[List[int]] = []
                unknown = False
                for w in tokens:
                    t = lookup_dict.get(w)
                    if t is None:
                        unknown = True
                        break
                    word_targets.append(t)

                if unknown or not word_targets:
                    unknown_f.write(line + "\n")
                    continue

                # Concatenate with separator targets
                segment_targets: List[int] = []
                for word in word_targets[:-1]:
                    segment_targets.extend(word)
                    segment_targets.extend(word_separation_targets)
                segment_targets.extend(word_targets[-1])

                if not segment_targets:
                    # Should not happen, but be robust
                    unknown_f.write(line + "\n")
                    continue

                # Write to HDF; seq_tag is the global (1-based) line number
                out_hdf_writer.insert_batch(
                    inputs=np.array(segment_targets, dtype="int32").reshape((1, -1)),
                    seq_len=[len(segment_targets)],
                    seq_tag=[str(line_idx)],
                )

        out_hdf_writer.close()

        print(
            f"[TextFileToTargetHdfJob] Finished shard {shard_idx+1}/{num_splits}, "
            f"wrote HDF to {out_hdf_path} and unknown lines to {unknown_txt_path}"
        )

