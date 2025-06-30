"""
Serialize a given dataset to a text file.
"""

from __future__ import annotations
from typing import Optional, Any, Sequence, Dict, Tuple
from sisyphus import Job, Task, tk


class ReturnnDatasetToTextDictJob(Job):
    """
    Takes any dataset dict, and extracts all data from it, via serialization.
    """

    def __init__(
        self,
        *,
        returnn_dataset: Dict[str, Any],
        returnn_dataset_ext_non_hashed: Optional[Dict[str, Any]] = None,
        returnn_root: Optional[tk.Path] = None,
        multi_proc_dataset_opts: Optional[Dict[str, Any]] = None,
        seq_list: Optional[tk.Path] = None,
        data_key: str,
        vocab: Optional[Dict[str, Any]] = None,
        raw_replacement_list: Sequence[Tuple[str, str]] = (),
        raw_final_strip: bool = False,
    ):
        """
        :param returnn_dataset: dict, the dataset dict, as used in RETURNN.
        :param returnn_dataset_ext_non_hashed: optional addition to the dataset dict but non-hashed
        :param returnn_root: path, optional, the RETURNN root dir.
        :param multi_proc_dataset_opts: dict, optional. if given, wraps the dataset in :class:`MultiProcDataset`.
            This is not hashed.
        :param seq_list: path, optional, a list of seq tags to process. If given, this also defines the order.
        :param data_key: str, the data key to serialize.
        :param vocab: dict, optional, the vocab dict, as used in RETURNN.
            If given, it uses :func:`Vocabulary.get_seq_labels`.
            If not given, it uses :func:`Dataset.serialize` (which might not always do the correct thing).
        :param raw_replacement_list: Can be used to directly transform BPE/SPM to words.
            Example: BPE: ``[("@@ ", "")]``, SPM: ``[(" ", ""), ("▁", " ")]``.
        :param raw_final_strip: If given, will strip the final output.
        """
        self.returnn_dataset = returnn_dataset
        self.returnn_dataset_ext_non_hashed = returnn_dataset_ext_non_hashed
        self.returnn_root = returnn_root
        self.multi_proc_dataset_opts = multi_proc_dataset_opts
        self.seq_list = seq_list
        self.data_key = data_key
        self.vocab = vocab
        self.raw_replacement_list = raw_replacement_list
        self.raw_final_strip = raw_final_strip

        self.out_txt = self.output_path("out.txt.gz")

        self.rqmt = {"cpu": 1, "mem": 4, "time": 1, "gpu": 0}

    @classmethod
    def hash(cls, parsed_args):
        parsed_args = parsed_args.copy()
        parsed_args.pop("returnn_dataset_ext_non_hashed")
        parsed_args.pop("multi_proc_dataset_opts")
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
        from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(1, returnn_root.get_path())

        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset
        from returnn.datasets.util.vocabulary import Vocabulary
        from returnn.log import log
        from returnn.util.basic import hms

        config = Config()
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 4
        log.init_by_config(config)

        if self.seq_list:
            seq_list = util.uopen(self.seq_list.get_path(), "rt").read().splitlines()
        else:
            seq_list = None

        dataset_dict = self.returnn_dataset
        dataset_dict = dict_update_deep(dataset_dict, self.returnn_dataset_ext_non_hashed)
        if self.multi_proc_dataset_opts:
            dataset_dict = {"class": "MultiProcDataset", "dataset": dataset_dict, **self.multi_proc_dataset_opts}
        dataset_dict = util.instanciate_delayed(dataset_dict)
        print("RETURNN dataset dict:", dataset_dict)
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)
        dataset.init_seq_order(epoch=1, seq_list=seq_list)

        if self.vocab:
            vocab = self.vocab
            vocab = util.instanciate_delayed(vocab)
            print("RETURNN vocab:", vocab)
            vocab = Vocabulary.create_vocab(**vocab)
        else:
            assert dataset.labels[self.data_key]
            vocab = Vocabulary.create_vocab_from_labels(dataset.labels[self.data_key])

        # noinspection PyBroadException
        try:
            num_seqs = dataset.num_seqs
        except Exception:  # might not work for all datasets
            num_seqs = None
        start_time = time.monotonic()

        with tempfile.NamedTemporaryFile(suffix="." + os.path.basename(self.out_txt.get_path())) as tmp_file:
            print("Using temp file:", tmp_file.name)
            with util.uopen(tmp_file.name, "wt") as f:
                f.write("{\n")
                seq_idx = 0
                while dataset.is_less_than_num_seqs(seq_idx):
                    if seq_idx % 100 == 0:
                        info = [f"seq idx {seq_idx}"]
                        if num_seqs is not None:
                            start_elapsed = time.monotonic() - start_time
                            complete = seq_idx / num_seqs
                            assert 1 >= complete >= 0, f"{seq_idx} seq idx, {num_seqs} num seqs"
                            total_time_estimated = start_elapsed / (complete or 1e-5)
                            remaining_estimated = total_time_estimated - start_elapsed
                            info += [
                                f"num seqs {num_seqs}",
                                f"exp. remaining {hms(remaining_estimated)}",
                                f"complete {complete:.2%}",
                            ]
                        else:
                            info += ["num seqs unknown"]
                        print(", ".join(info))

                    dataset.load_seqs(seq_idx, seq_idx + 1)
                    seq_tag = dataset.get_tag(seq_idx)
                    if seq_list is not None:
                        assert seq_tag == seq_list[seq_idx], (
                            f"seq_list seq tag mismatch in seq idx {seq_list},"
                            f" dataset tag {dataset.get_tag(seq_idx)!r} != seq list tag {seq_list[seq_idx]!r}"
                        )
                    data = dataset.get_data(seq_idx, self.data_key)
                    s = vocab.get_seq_labels(data)
                    for old, new in self.raw_replacement_list:
                        s = s.replace(old, new)
                    if self.raw_final_strip:
                        s = s.strip()
                    f.write(f"{seq_tag!r}: {s!r},\n")
                    seq_idx += 1
                f.write("}\n")
            print("Copy to final file:", self.out_txt.get_path())
            shutil.copy(tmp_file.name, self.out_txt.get_path())

        if seq_list is not None:
            assert seq_idx == len(seq_list), f"seq_list length mismatch: got {seq_idx} != list {len(seq_list)}"


class ReturnnDatasetToTextLinesJob(Job):
    """
    Takes any dataset dict, and extracts all data from it, via serialization.
    """

    def __init__(
        self,
        *,
        returnn_dataset: Dict[str, Any],
        returnn_dataset_ext_non_hashed: Optional[Dict[str, Any]] = None,
        returnn_root: Optional[tk.Path] = None,
        multi_proc_dataset_opts: Optional[Dict[str, Any]] = None,
        seq_list: Optional[tk.Path] = None,
        data_key: str,
        vocab: Optional[Dict[str, Any]] = None,
        raw_replacement_list: Sequence[Tuple[str, str]] = (),
        raw_final_strip: bool = False,
    ):
        """
        :param returnn_dataset: dict, the dataset dict, as used in RETURNN.
        :param returnn_dataset_ext_non_hashed: optional addition to the dataset dict but non-hashed
        :param returnn_root: path, optional, the RETURNN root dir.
        :param multi_proc_dataset_opts: dict, optional. if given, wraps the dataset in :class:`MultiProcDataset`.
            This is not hashed.
        :param seq_list: path, optional, a list of seq tags to process. If given, this also defines the order.
        :param data_key: str, the data key to serialize.
        :param vocab: dict, optional, the vocab dict, as used in RETURNN.
            If given, it uses :func:`Vocabulary.get_seq_labels`.
            If not given, it uses :func:`Dataset.serialize` (which might not always do the correct thing).
        :param raw_replacement_list: Can be used to directly transform BPE/SPM to words.
            Example: BPE: ``[("@@ ", "")]``, SPM: ``[(" ", ""), ("▁", " ")]``.
        :param raw_final_strip: If given, will strip the final output.
        """
        self.returnn_dataset = returnn_dataset
        self.returnn_dataset_ext_non_hashed = returnn_dataset_ext_non_hashed
        self.returnn_root = returnn_root
        self.multi_proc_dataset_opts = multi_proc_dataset_opts
        self.seq_list = seq_list
        self.data_key = data_key
        self.vocab = vocab
        self.raw_replacement_list = raw_replacement_list
        self.raw_final_strip = raw_final_strip

        self.out_txt = self.output_path("out.txt.gz")

        self.rqmt = {"cpu": 1, "mem": 4, "time": 1, "gpu": 0}

    @classmethod
    def hash(cls, parsed_args):
        parsed_args = parsed_args.copy()
        parsed_args.pop("returnn_dataset_ext_non_hashed")
        parsed_args.pop("multi_proc_dataset_opts")
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
        from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(1, returnn_root.get_path())

        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset
        from returnn.datasets.util.vocabulary import Vocabulary
        from returnn.log import log
        from returnn.util.basic import hms

        config = Config()
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 4
        log.init_by_config(config)

        if self.seq_list:
            seq_list = util.uopen(self.seq_list.get_path(), "rt").read().splitlines()
        else:
            seq_list = None

        dataset_dict = self.returnn_dataset
        dataset_dict = dict_update_deep(dataset_dict, self.returnn_dataset_ext_non_hashed)
        if self.multi_proc_dataset_opts:
            dataset_dict = {"class": "MultiProcDataset", "dataset": dataset_dict, **self.multi_proc_dataset_opts}
        dataset_dict = util.instanciate_delayed(dataset_dict)
        print("RETURNN dataset dict:", dataset_dict)
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)
        dataset.init_seq_order(epoch=1, seq_list=seq_list)

        if self.vocab:
            vocab = self.vocab
            vocab = util.instanciate_delayed(vocab)
            print("RETURNN vocab:", vocab)
            vocab = Vocabulary.create_vocab(**vocab)
        else:
            assert dataset.labels[self.data_key]
            vocab = Vocabulary.create_vocab_from_labels(dataset.labels[self.data_key])

        # noinspection PyBroadException
        try:
            num_seqs = dataset.num_seqs
        except Exception:  # might not work for all datasets
            num_seqs = None
        start_time = time.monotonic()

        with tempfile.NamedTemporaryFile(suffix="." + os.path.basename(self.out_txt.get_path())) as tmp_file:
            print("Using temp file:", tmp_file.name)
            with util.uopen(tmp_file.name, "wt") as f:
                seq_idx = 0
                while dataset.is_less_than_num_seqs(seq_idx):
                    if seq_idx % 100 == 0:
                        info = [f"seq idx {seq_idx}"]
                        if num_seqs is not None:
                            start_elapsed = time.monotonic() - start_time
                            complete = seq_idx / num_seqs
                            assert 1 >= complete >= 0, f"{seq_idx} seq idx, {num_seqs} num seqs"
                            total_time_estimated = start_elapsed / (complete or 1e-5)
                            remaining_estimated = total_time_estimated - start_elapsed
                            info += [
                                f"num seqs {num_seqs}",
                                f"exp. remaining {hms(remaining_estimated)}",
                                f"complete {complete:.2%}",
                            ]
                        else:
                            info += ["num seqs unknown"]
                        print(", ".join(info))

                    dataset.load_seqs(seq_idx, seq_idx + 1)
                    if seq_list is not None:
                        assert dataset.get_tag(seq_idx) == seq_list[seq_idx], (
                            f"seq_list seq tag mismatch in seq idx {seq_list},"
                            f" dataset tag {dataset.get_tag(seq_idx)!r} != seq list tag {seq_list[seq_idx]!r}"
                        )
                    data = dataset.get_data(seq_idx, self.data_key)
                    s = vocab.get_seq_labels(data)
                    for old, new in self.raw_replacement_list:
                        s = s.replace(old, new)
                    if self.raw_final_strip:
                        s = s.strip()
                    f.write(s + "\n")
                    seq_idx += 1
            print("Copy to final file:", self.out_txt.get_path())
            shutil.copy(tmp_file.name, self.out_txt.get_path())

        if seq_list is not None:
            assert seq_idx == len(seq_list), f"seq_list length mismatch: got {seq_idx} != list {len(seq_list)}"
