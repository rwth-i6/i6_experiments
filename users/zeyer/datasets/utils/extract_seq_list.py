"""
Extract/generate seq lists.
"""

from typing import Optional, Union, Any, Dict
from sisyphus import tk, Task, Job
from i6_core.util import uopen


class ExtractSeqListJob(Job):
    """
    Takes any dataset dict, and extracts all seq tags from it.
    """

    def __init__(
        self,
        *,
        returnn_dataset: Dict[str, Any],  # to get all seq tags
        returnn_dataset_ext_non_hashed: Optional[Dict[str, Any]] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param returnn_dataset: dict, the dataset dict, as used in RETURNN.
        :param returnn_dataset_ext_non_hashed: optional addition to the dataset dict but non-hashed
        :param returnn_root: path, optional, the RETURNN root dir.
        """
        self.returnn_dataset = returnn_dataset
        self.returnn_dataset_ext_non_hashed = returnn_dataset_ext_non_hashed
        self.returnn_root = returnn_root

        self.out_seq_list = self.output_path("out_seq_list.txt")

        self.rqmt = {"cpu": 1, "mem": 4, "time": 1, "gpu": 0}

    @classmethod
    def hash(cls, parsed_args):
        parsed_args = parsed_args.copy()
        parsed_args.pop("returnn_dataset_ext_non_hashed")
        return super().hash(parsed_args)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sys
        import os
        import time
        import tempfile
        import shutil
        import i6_experiments
        from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(1, returnn_root.get_path())

        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset
        from returnn.log import log
        from returnn.util.basic import hms

        config = Config()
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 4
        log.init_by_config(config)

        dataset_dict = self.returnn_dataset
        dataset_dict = dict_update_deep(dataset_dict, self.returnn_dataset_ext_non_hashed)
        dataset_dict = util.instanciate_delayed(dataset_dict)
        print("RETURNN dataset dict:", dataset_dict)
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)

        # noinspection PyBroadException
        try:
            num_seqs = dataset.get_total_num_seqs()
        except Exception:  # might not work for all datasets
            num_seqs = None
        start_time = time.monotonic()

        with tempfile.NamedTemporaryFile(suffix="." + os.path.basename(self.out_seq_list.get_path())) as tmp_file:
            print("Using temp file:", tmp_file.name)
            with util.uopen(tmp_file.name, "wt") as f:
                for seq_idx, seq_tag in enumerate(dataset.get_all_tags()):
                    print(seq_tag, file=f)

                    if seq_idx % 100 == 0:
                        info = [f"seq idx {seq_idx}"]
                        if num_seqs is not None:
                            start_elapsed = time.monotonic() - start_time
                            complete = (seq_idx + 1) / num_seqs
                            assert 1 >= complete >= 0, f"{seq_idx} seq idx, {num_seqs} num seqs"
                            total_time_estimated = start_elapsed / complete
                            remaining_estimated = total_time_estimated - start_elapsed
                            info += [
                                f"num seqs {num_seqs}",
                                f"exp. remaining {hms(remaining_estimated)}",
                                f"complete {complete:.2%}",
                            ]
                        else:
                            info += ["num seqs unknown"]
                        print(", ".join(info))

            print("Copying to final file:", self.out_seq_list.get_path())
            shutil.copy(tmp_file.name, self.out_seq_list.get_path())


class ExtractNumLinesFromTextFileJob(Job):
    """
    Extracts the number of lines from a (potentially gzipped) text file.
    """

    def __init__(self, *, text_file: tk.Path, skip_empty_lines: bool = True):
        self.text_file = text_file
        self.skip_empty_lines = skip_empty_lines
        self.out_num_lines = self.output_var("out_num_lines.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        num_lines = 0
        with uopen(self.text_file.get_path(), "rt") as f:
            for line in f:
                if self.skip_empty_lines and not line.strip():
                    continue
                num_lines += 1
        self.out_num_lines.set(num_lines)


class WriteLmDatasetSeqListJob(Job):
    """
    LmDataset has simple seq tags like "line-{line_nr}".
    So to write out the seq list, we just need to know the number of lines.
    """

    def __init__(self, *, num_lines: Union[int, tk.Variable]):
        super().__init__()
        assert isinstance(num_lines, (int, tk.Variable))
        self.num_lines = num_lines
        self.out_seq_list = self.output_path("out_seq_list.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        num_lines = self.num_lines
        if isinstance(num_lines, tk.Variable):
            num_lines = num_lines.get()
        assert isinstance(num_lines, int)
        with open(self.out_seq_list.get_path(), "w") as f:
            for line_nr in range(num_lines):
                print(f"line-{line_nr}", file=f)


class WriteSeqListFromShuffledJob(Job):
    """
    Assuming that some dataset was split and shuffled using
    :class:`i6_core.corpus.segments.ShuffleAndSplitSegmentsJob`,
    and then merged again (via some more complex pipeline,
    e.g. involving :class:`i6_core.returnn.oggzip.BlissToOggZipJob`),
    we might get sequence tags like `"librispeech-lm-part{split_key}/recording_{split_seq_idx}/line_{split_seq_idx}"`.

    We write those seq tags out to a seq list file.
    This file could be used for :class:`returnn.datasets.meta.MetaDataset` ``seq_list_file``.

    The code is based on :class:`i6_core.corpus.segments.ShuffleAndSplitSegmentsJob`.
    """

    def __init__(
        self,
        *,
        seq_tag_template: str,
        num_seqs: Union[int, tk.Variable],
        split: Dict[str, float],
        shuffle=True,
        shuffle_seed=0x3C5EA3E47D4E0077,
    ):
        """
        :param seq_tag_template: e.g. `"librispeech-lm-part{split_key}/recording_{split_seq_idx}/line_{split_seq_idx}"`
        :param num_seqs: total number of sequences
        :param split: dict of split keys to split ratio
        :param shuffle: whether to shuffle
        :param shuffle_seed: seed for the shuffle
        """
        assert isinstance(split, dict)
        assert all(s > 0 for s in split.values())
        assert abs(sum(split.values()) - 1.0) < 1e-10

        self.seq_tag_template = seq_tag_template
        self.num_seqs = num_seqs
        self.split = split
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        self.out_segments = self.output_path("out.segments")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import random
        import itertools as it

        n = self.num_seqs
        if isinstance(n, tk.Variable):
            n = n.get()
        assert isinstance(n, int)
        segments = list(range(n))

        if self.shuffle:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(segments)

        ordered_keys = sorted(self.split.keys())
        split_indices = [0] + [int(n * c) for c in it.accumulate(self.split[k] for k in ordered_keys)]
        split_indices[-1] = n  # just in case we get numeric errors that drop the last element

        orig_seq_idx_to_new_seq_tag = {}
        for split_idx, split_key in enumerate(ordered_keys):
            for split_seq_idx, orig_seq_idx in enumerate(
                segments[split_indices[split_idx] : split_indices[split_idx + 1]]
            ):
                orig_seq_idx_to_new_seq_tag[orig_seq_idx] = self.seq_tag_template.format(
                    split_key=split_key, split_seq_idx=split_seq_idx
                )

        with uopen(self.out_segments.get_path(), "wt", encoding="utf-8") as f:
            for orig_seq_idx in range(n):
                f.write(f"{orig_seq_idx_to_new_seq_tag[orig_seq_idx]}\n")


class ReorderSeqListJob(Job):
    """
    Reorder a seq list file, based on a given order.
    """

    def __init__(self, *, source_seq_list: tk.Path, source_seq_list_order: tk.Path, target_seq_list: tk.Path):
        """
        :param source_seq_list: input seq list, specifying the dataset seq order.
            This should match the same order (but with potential different seq names) as ``target_seq_list``.
        :param source_seq_list_order: input seq list, specifying the order.
            These are seq names from ``source_seq_list``.
        :param target_seq_list: seq list, using seq names from some potential other dataset,
            using the same order as ``source_seq_list``.
        """
        self.source_seq_list = source_seq_list
        self.source_seq_list_order = source_seq_list_order
        self.target_seq_list = target_seq_list

        self.out_target_seq_list_order = self.output_path("out_seq_list_order.txt")

        self.rqmt = {"cpu": 1, "mem": 4, "time": 1, "gpu": 0}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        print("Reading source seq list", self.source_seq_list.get_path(), "...")
        with uopen(self.source_seq_list.get_path(), "rt") as f:
            source_seq_list = f.read().splitlines()
        print("Number of seqs:", len(source_seq_list))
        print("Reading source seq list order", self.source_seq_list_order.get_path(), "...")
        with uopen(self.source_seq_list_order.get_path(), "rt") as f:
            source_seq_list_order = f.read().splitlines()
        print("Number of seqs in seq order:", len(source_seq_list_order))
        print("Reading target seq list", self.target_seq_list.get_path(), "...")
        with uopen(self.target_seq_list.get_path(), "rt") as f:
            target_seq_list = f.read().splitlines()
        assert len(source_seq_list) == len(target_seq_list)

        source_seq_indices = {seq: i for i, seq in enumerate(source_seq_list)}
        seq_list_order = [source_seq_indices[seq] for seq in source_seq_list_order]
        target_seq_list_reordered = [target_seq_list[i] for i in seq_list_order]

        import tempfile
        import shutil
        import os
        import time

        num_seqs = len(target_seq_list_reordered)
        start_time = time.monotonic()

        def _hms(s: float) -> str:
            """
            :param s: seconds
            :return: e.g. "1:23:45" (hs:ms:secs). see hms_fraction if you want to get fractional seconds
            """
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            return "%d:%02d:%02d" % (h, m, s)

        with tempfile.NamedTemporaryFile(
            suffix="." + os.path.basename(self.out_target_seq_list_order.get_path())
        ) as tmp_file:
            print("Using temp file:", tmp_file.name)
            with uopen(tmp_file.name, "wt") as f:
                for seq_idx, seq_tag in enumerate(target_seq_list_reordered):
                    print(seq_tag, file=f)

                    if seq_idx % 100 == 0:
                        info = [f"seq idx {seq_idx}"]
                        start_elapsed = time.monotonic() - start_time
                        complete = (seq_idx + 1) / num_seqs
                        assert 1 >= complete >= 0, f"{seq_idx} seq idx, {num_seqs} num seqs"
                        total_time_estimated = start_elapsed / complete
                        remaining_estimated = total_time_estimated - start_elapsed
                        info += [
                            f"num seqs {num_seqs}",
                            f"exp. remaining {_hms(remaining_estimated)}",
                            f"complete {complete:.2%}",
                        ]
                        print(", ".join(info))

            print("Copying to final file:", self.out_target_seq_list_order.get_path())
            shutil.copy(tmp_file.name, self.out_target_seq_list_order.get_path())
