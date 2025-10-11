"""
Rescoring multiple text dicts / search outputs.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, List, Tuple, Set, Callable
from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedBase

import i6_core.util as util
from i6_core.returnn import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2

from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RescoreDef
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

from i6_experiments.users.zeyer.recog import (
    _returnn_v2_get_model,
    _returnn_v2_get_forward_callback,
    _v2_forward_out_filename,
)

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim, TensorDict
    from i6_experiments.users.zhang.experiments.decoding.prior_rescoring import Prior


def combine_scores(scores: List[Tuple[Union[float, DelayedBase], RecogOutput]], alias: Optional[str] = "") -> RecogOutput:
    """
    Combine scores from multiple sources, linearly weighted by the given weights.

    :param scores: dict: recog output -> weight. We assume they have the same hyp txt.
    :param same_txt: if all the hyps have the same txt across scores.
    :param out_txt: used to define the hyp txt if not same_txt
    :return: combined scores
    """
    assert scores
    from i6_experiments.users.zhang.experiments.decoding.combine_scores import SearchCombineScoresJob as SearchCombineScoresJob_streaming
    #job = SearchCombineScoresJob([(weight, recog_output.output) for weight, recog_output in scores])
    job = SearchCombineScoresJob_streaming([(weight, recog_output.output) for weight, recog_output in scores])
    if alias:
        job.add_alias(alias)
    return RecogOutput(output=job.out_search_results)


class SearchCombineScoresJob(Job):
    """
    Takes a number of files, each with the same N-best list, including scores,
    and combines the scores with some weights.
    """

    def __init__(self, search_py_output: List[Tuple[Union[float, DelayedBase], tk.Path]], *, output_gzip: bool = True, version:int = 1):
        """
        :param search_py_output: list of tuple (search output file from RETURNN in python format (n-best list), weight)
        :param output_gzip: gzip the output
        """
        assert len(search_py_output) > 0
        self.search_py_output = search_py_output
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))
        self.rqmt = {"time": 1, "cpu": 1, "mem": 12} # TODO: dynamically determine mem_rqmt
    def tasks(self):
        """task"""
        yield Task("run", rqmt=self.rqmt)

    # def run(self):
    #     """
    #     Stream-merge N-best lists without requiring same seq_tag order or same N-best order.
    #     - Output preserves the first file's seq_tag order and hypothesis order.
    #     - Other files are scanned lazily; each block is spilled to a temp .jsonl.gz and looked up by seq_tag.
    #     - Hypotheses inside a block are aligned by text; sets can be enforced equal or unioned.
    #     """
    #     import ast
    #     import gzip
    #     import io
    #     import os
    #     import re
    #     import json
    #     import tempfile
    #     from typing import Iterator, Tuple, List, Dict
    #
    #     # ---- knobs ---------------------------------------------------------------
    #     STRICT_EQUAL_TEXT_SETS = True
    #     # If True: all files must have identical hypothesis-text sets per seq_tag.
    #     # If False: we take the union; missing texts in a file contribute 0.0.
    #
    #     DETECT_DUPLICATE_TEXTS = True
    #
    #     # If True: error out if a file has duplicate hypothesis texts within the same seq_tag.
    #
    #     # ---- helpers -------------------------------------------------------------
    #
    #     def _to_path(p):
    #         # Try i6/sisyphus Path.get_path(), fall back to str
    #         return p.get_path() if hasattr(p, "get_path") else str(p)
    #
    #     # Open .gz or plain text transparently, *text* mode with buffering
    #     def _open_text(path: string):
    #         if path.endswith(".gz"):
    #             return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    #         return open(path, "rt", encoding="utf-8")
    #
    #     # Matches a line starting a block like: 'seq_tag': [
    #     _seq_start_re = re.compile(r"""^\s*(['"])(?P<tag>.*?)\1\s*:\s*\[\s*$""")
    #
    #     def _skip_until_open_brace(f):
    #         """Advance the stream to the first '{' (start of the dict)."""
    #         for line in f:
    #             if "{" in line:
    #                 break
    #
    #     def _iter_blocks(stream) -> Iterator[Tuple[str, Iterator[Tuple[float, str]]]]:
    #         """
    #         Yield (seq_tag, tuple_iter) for each block:
    #           'seq_tag': [
    #             (score, text),
    #             ...
    #           ],
    #         Does not materialize the whole list; tuple_iter streams tuples.
    #         """
    #         _skip_until_open_brace(stream)
    #
    #         for line in stream:
    #             line_strip = line.strip()
    #
    #             # End of outer dict
    #             if line_strip.startswith("}"):
    #                 return
    #
    #             # Try to match block start
    #             m = _seq_start_re.match(line)
    #             if not m:
    #                 # Lines like empty, comments, or commas between entries
    #                 continue
    #
    #             seq_tag = m.group("tag")
    #
    #             def tuple_iter():
    #                 # Stream until matching closing "],"
    #                 for inner in stream:
    #                     s = inner.strip()
    #                     if s == "],":
    #                         return
    #                     if not s or s == ",":
    #                         continue
    #                     # Lines are "(score, text)," – strip trailing comma
    #                     trailing_comma = s.endswith(",")
    #                     if trailing_comma:
    #                         s = s[:-1]
    #
    #                     # Fast path: single-line tuple "(..., ...)"
    #                     if s.startswith("(") and s.endswith(")"):
    #                         tup = ast.literal_eval(s)
    #                         yield float(tup[0]), tup[1]
    #                         continue
    #
    #                     # Fallback: accumulate until balanced parentheses
    #                     buf = [s]
    #                     balance = s.count("(") - s.count(")")
    #                     while balance > 0:
    #                         nxt = next(stream)
    #                         t = nxt.strip()
    #                         if t == "],":
    #                             # Malformed; bail gracefully
    #                             return
    #                         if t.endswith(","):
    #                             t = t[:-1]
    #                         buf.append(t)
    #                         balance += t.count("(") - t.count(")")
    #                     tup = ast.literal_eval(" ".join(buf))
    #                     yield float(tup[0]), tup[1]
    #
    #             yield seq_tag, tuple_iter()
    #
    #     # Lazy block fetcher that tolerates different seq_tag *order*
    #     class BlockFetcher:
    #         """
    #         Reads a file forward once. Each encountered block is written to a temp .jsonl.gz.
    #         fetch(seq_tag) returns the temp path (or None if not present).
    #         """
    #
    #         def __init__(self, path: str):
    #             self._stream = _open_text(path)
    #             self._cache: Dict[str, str] = {}  # seq_tag -> temp file path
    #             self._finished = False
    #
    #         def _spill_block(self, seq_tag: str, tuple_iter):
    #             # Write this block as JSONL: [score, text]
    #             fd, temp_path = tempfile.mkstemp(suffix=".jsonl.gz")
    #             os.close(fd)  # we'll reopen with gzip
    #             with gzip.open(temp_path, "wt", encoding="utf-8") as out:
    #                 for score, text in tuple_iter:
    #                     out.write(json.dumps([float(score), text]))
    #                     out.write("\n")
    #             self._cache[seq_tag] = temp_path
    #
    #         def fetch(self, want_tag: str) -> str | None:
    #             """Return temp path for want_tag, reading forward and caching as needed."""
    #             if want_tag in self._cache:
    #                 return self._cache[want_tag]
    #             if self._finished:
    #                 return None
    #
    #             for seq_tag, tuple_iter in _iter_blocks(self._stream):
    #                 self._spill_block(seq_tag, tuple_iter)
    #                 if seq_tag == want_tag:
    #                     return self._cache[seq_tag]
    #
    #             # EOF
    #             self._finished = True
    #             return None
    #
    #         def close(self):
    #             try:
    #                 self._stream.close()
    #             except Exception:
    #                 pass
    #
    #         def cached_paths(self):
    #             return list(self._cache.values())
    #
    #     # ---- load weights and file paths ----------------------------------------
    #
    #     weights: List[float] = []
    #     files: List[str] = []
    #
    #     DelayedCls = globals().get("DelayedBase", None)  # avoid NameError if not present
    #
    #     for weight, fn in self.search_py_output:
    #         if DelayedCls is not None and isinstance(weight, DelayedCls):
    #             weight = weight.get()
    #         if not isinstance(weight, (int, float)):
    #             raise TypeError(f"invalid weight {weight!r} type {type(weight)}")
    #         weights.append(float(weight))
    #         files.append(_to_path(fn))
    #
    #     if not files:
    #         raise AssertionError("No input files")
    #
    #     driver_path = files[0]
    #     other_paths = files[1:]
    #
    #     # Build lazy fetchers for non-driver files
    #     fetchers = [BlockFetcher(p) for p in other_paths]
    #
    #     # ---- write output --------------------------------------------------------
    #
    #     tmp_filename = "tmp.py" + (".gz" if self.out_search_results.get_path().endswith(".gz") else "")
    #     if not tmp_filename.endswith(".gz"):
    #         raise AssertionError("should have .gz extension")
    #
    #     try:
    #         with gzip.open(tmp_filename, "wt", encoding="utf-8") as out:
    #             out.write("{\n")
    #
    #             # Drive seq_tag order from the first file
    #             with _open_text(driver_path) as driver_stream:
    #                 for seq_tag, driver_tuples in _iter_blocks(driver_stream):
    #                     # Materialize driver list to keep order; map for O(1) lookups
    #                     driver_list: List[Tuple[float, str]] = list(driver_tuples)
    #                     driver_map = {t: s for (s, t) in driver_list}
    #                     driver_order = [t for _, t in driver_list]
    #                     driver_text_set = set(driver_order)
    #
    #                     if DETECT_DUPLICATE_TEXTS and len(driver_text_set) != len(driver_order):
    #                         raise AssertionError(f"Duplicate hypothesis texts in first file for {seq_tag!r}")
    #
    #                     # For every other file, fetch this seq_tag block (order-independent)
    #                     others_dicts: List[Dict[str, float]] = []
    #                     for fx in fetchers:
    #                         temp_path = fx.fetch(seq_tag)
    #                         d: Dict[str, float] = {}
    #                         if temp_path is not None:
    #                             with gzip.open(temp_path, "rt", encoding="utf-8") as f:
    #                                 for line in f:
    #                                     score_i, text_i = json.loads(line)
    #                                     if DETECT_DUPLICATE_TEXTS and text_i in d:
    #                                         raise AssertionError(
    #                                             f"Duplicate hypothesis text in an input for {seq_tag!r}: {text_i!r}"
    #                                         )
    #                                     d[text_i] = float(score_i)
    #                         # If missing entirely (seq_tag not present), leave as empty dict (treated as 0.0)
    #                         others_dicts.append(d)
    #
    #                     # Reconcile sets
    #                     if STRICT_EQUAL_TEXT_SETS:
    #                         for idx, d in enumerate(others_dicts, start=1):
    #                             if set(d.keys()) != driver_text_set:
    #                                 missing = driver_text_set - set(d.keys())
    #                                 extra = set(d.keys()) - driver_text_set
    #                                 raise AssertionError(
    #                                     f"Different hypothesis text sets for {seq_tag!r} in file #{idx + 1}. "
    #                                     f"Missing: {len(missing)}, Extra: {len(extra)}"
    #                                 )
    #                         final_order = driver_order
    #                     else:
    #                         final_order = list(driver_order)
    #                         seen = set(final_order)
    #                         # Append extras in first-appearance order across other files
    #                         for d in others_dicts:
    #                             for t in d.keys():
    #                                 if t not in seen:
    #                                     final_order.append(t)
    #                                     seen.add(t)
    #
    #                     # Combine and write block
    #                     out.write(f"{seq_tag!r}: [\n")
    #                     for text in final_order:
    #                         acc = weights[0] * driver_map.get(text, 0.0)
    #                         for w, d in zip(weights[1:], others_dicts):
    #                             acc += w * d.get(text, 0.0)
    #                         out.write(f"({acc!r}, {text!r}),\n")
    #                     out.write("],\n")
    #
    #             out.write("}\n")
    #
    #         # Atomic move into place
    #         os.replace(tmp_filename, self.out_search_results.get_path())
    #     finally:
    #         # Close fetchers and clean up temp files
    #         for fx in fetchers:
    #             try:
    #                 fx.close()
    #             except Exception:
    #                 pass
    #             for p in fx.cached_paths():
    #                 try:
    #                     os.remove(p)
    #                 except Exception:
    #                     pass

    def run(self):
        """run"""
        import tracemalloc

        tracemalloc.start()

        data: List[Tuple[float, Dict[str, List[Tuple[float, str]]]]] = []
        for weight, fn in self.search_py_output:
            if isinstance(weight, DelayedBase):
                weight = weight.get()
            assert isinstance(weight, (int, float)), f"invalid weight {weight!r} type {type(weight)}"
            out = eval(util.uopen(fn, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            data.append((weight, out))

        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current / 1024 ** 2:.1f} MB; Peak: {peak / 1024 ** 2:.1f} MB")

        tracemalloc.stop()
        weights: List[float] = [weight for weight, _ in data]
        seq_tags: List[str] = list(data[0][1].keys())
        seq_tags_set: Set[str] = set(seq_tags)
        assert len(seq_tags) == len(seq_tags_set), "duplicate seq tags"
        for _, d in data:
            assert set(d.keys()) == seq_tags_set, "inconsistent seq tags"

        tmp_filename = "tmp.py" + (".gz" if self.out_search_results.get_path().endswith(".gz") else "")
        import gzip
        assert tmp_filename.endswith(".gz"), "should have .gz extension"
        with gzip.open(tmp_filename, "wt") as out:
            out.write("{\n")
            for seq_tag in seq_tags:
                # references only; do not copy
                entries_per_file = [d[seq_tag] for _, d in data]

                # sanity: same n-best order across files, force same order inside the list
                nb0 = entries_per_file[0]
                hyp_list0 = [hyp.strip().replace("<unk>", "@@") for _, hyp in nb0]
                for i, nb in enumerate(entries_per_file[1:], start=1):
                    if len(nb) != len(nb0):
                        raise AssertionError(f"[Same n-best and order Check]Length Mismatch found at {seq_tag} file_index {i}, lengths ({len(nb)},{len(nb0)})")
                    # TODO: remove this hack, should clean the input rather do this here
                    hyp_list = [hyp.strip().replace("<unk>", "@@") for _, hyp in nb]

                    hyp_set = set(hyp_list)
                    hyp_set0 = set(hyp_list0)
                    if hyp_set != hyp_set0:#any(nb[j][1].strip() != nb0[j][1].strip() for j in range(len(nb0))):
                        #if hyp_set != hyp_set0:
                        diff_idx = zip(hyp_list0, hyp_list)
                        for j, hyp_pair in enumerate(diff_idx):
                            if not hyp_pair[0] and not hyp_pair[1]:
                                # print("DEBUG repr:", repr(hyp_pair[0]), repr(hyp_pair[1]))
                                # print("DEBUG types:", type(hyp_pair[0]), type(hyp_pair[1]))
                                continue
                            if hyp_pair[0] != hyp_pair[1]:
                                break
                        raise AssertionError(f"[Same n-best and order Check]Mismatch found at {seq_tag} file_index {i}, inner index {j} pair {hyp_pair}")
                        # else:
                        #     raise AssertionError(f"[Same n-best and order Check]Mismatch n list order {seq_tag} file_index {i}")

                out.write(f"{seq_tag!r}: [\n")
                for hyp_idx in range(len(nb0)):
                    # only read hyp text from the first file to avoid copies
                    hyp_text = nb0[hyp_idx][1]
                    # if must normalize text, do it in-place once:
                    # hyp_text = hyp_text.strip().replace("<unk>", "@@")
                    acc = 0.0
                    for file_idx, entries in enumerate(entries_per_file):
                        acc += entries[hyp_idx][0] * weights[file_idx]
                    out.write(f"({acc!r}, {hyp_text!r}),\n")
                out.write("],\n")
            out.write("}\n")
        import os
        os.replace(tmp_filename, self.out_search_results.get_path()) #ensures that no process will see the file until it's fully written
        # with gzip.open(tmp_filename, "wt") as out:
        #     out.write("{\n")
        #     for seq_tag in seq_tags:
        #         data_: List[List[Tuple[float, str]]] = [d[seq_tag] for _, d in data]
        #         hyps_: List[List[str]] = [[h.strip().replace("<unk>", "@@") for _, h in entry] for entry in data_]
        #         hyps0: List[str] = hyps_[0]
        #         assert isinstance(hyps0, list) and all(isinstance(h, str) for h in hyps0)
        #
        #         #assert all(hyps0 == hyps for hyps in hyps_)
        #         for i, hyps in enumerate(hyps_):
        #             if hyps != hyps0:
        #                 for j, (h0, h) in enumerate(zip(hyps0, hyps)):
        #                     if h0 != h:
        #                         print(f"Difference at outer index {i}, inner index {j}: '{h0}' != '{h}'")
        #                 # Also catch differing lengths
        #                 if len(hyps0) != len(hyps):
        #                     print(f"Length mismatch at index {i}: len(hyps0)={len(hyps0)}, len(hyps)={len(hyps)}")
        #                 assert False, f"Mismatch found at index {i}"
        #
        #         scores_per_hyp: List[List[float]] = [[score for score, _ in entry] for entry in data_]
        #         # n-best list as [(score, text), ...]
        #         out.write(f"{seq_tag!r}: [\n")
        #         for hyp_idx, hyp in enumerate(hyps0):
        #             score = sum(scores_per_hyp[file_idx][hyp_idx] * weights[file_idx] for file_idx in range(len(data)))
        #             out.write(f"({score!r}, {hyp!r}),\n")
        #         out.write("],\n")
        #     out.write("}\n")



class RescoreCheatJob(Job):
    """
    Add GT into the recog_out N list
    """

    def __init__(self, combined_search_py_output: tk.Path, combined_gt_py_output: tk.Path, *, output_gzip: bool = True, version:int = 3):
        """
        :param combined_search_py_output: rescored search output file from RETURNN in python format (n-best list)
        :param output_gzip: gzip the output
        """
        self.combined_search_py_output = combined_search_py_output
        self.combined_gt_py_output = combined_gt_py_output
        self.out_search_results = self.output_path("search_results_cheat.py" + (".gz" if output_gzip else ""))
        self.rqmt = {"time": 1, "cpu": 1, "mem": 3}

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        n_lists = eval(util.uopen(self.combined_search_py_output, "rt").read(),
                       {"nan": float("nan"), "inf": float("inf")})  # {seq_tag:[(score, hyp)]}
        gts = eval(util.uopen(self.combined_gt_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        seq_tags: List[str] = list(n_lists.keys())
        seq_tags_set: Set[str] = set(seq_tags)
        assert len(seq_tags) == len(seq_tags_set), "duplicate seq tags"
        assert set(gts.keys()) == seq_tags_set, "inconsistent seq tags"

        def ctc_collapse(ctc_sequence, blank_token = "<blank>"):
            """
            Collapses a CTC decoded tensor by removing consecutive duplicates and blank tokens.

            Args:
                ctc_sequence (list or tensor): Decoded CTC output sequence (list of indices).
                blank_token (int): Index representing the blank token in the sequence.

            Returns:
                Tensor: Collapsed sequence without repeated characters and blanks.
            """
            if blank_token not in ctc_sequence:
                #print(f"Warning: Passed likely non search labels to for collapse {ctc_sequence}")
                return ctc_sequence
            collapsed_sequence = []
            prev_token = None

            for token in ctc_sequence:
                if token != prev_token and token != blank_token:  # Remove repetition and blank
                    collapsed_sequence.append(token)
                prev_token = token  # Update previous token

            return collapsed_sequence

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag in seq_tags:
                gt_present = False

                gt_score, gt = gts[seq_tag][0]
                gt = " ".join(gt.split())

                # Write original hyps:
                out.write(f"{seq_tag!r}: [\n")
                targets_search_str = "["
                for score, hyp, *_ in n_lists[seq_tag]:
                    if ctc_collapse(hyp.split()) == ctc_collapse(gt.split()):
                        targets_search_str += f"[\n\t [{hyp}]\n\t"
                        gt_present = True
                    else:
                        targets_search_str += f"\n\t {hyp} \n\t"
                    out.write(f"({score!r}, {hyp!r}),\n")

                max_hyp_score = max([x[0] for x in n_lists[seq_tag]])
                if not gt_present:
                    out.write(f"({gt_score!r}, {gt!r}),\n")
                # else: # Also Write a dummy hyp there
                #     out.write(f"({float(-1e30)!r}, {''!r}),\n")

                with open("cheat_log", "a") as f:
                    log_txt = "Seq Tag: %s\n\tGround-truth score: %f\n\tMax Search score: %f" % (
                        seq_tag,
                        gt_score,
                        max_hyp_score,
                    )
                    log_txt += "\n\tGround-truth seq: %s\n\tSearch seq:       %s" % (
                        gt,
                        targets_search_str,
                    )
                    log_txt += "\n\tGT_present in N-list: %s\n\t-> %s" % (
                        str(gt_present),
                        "Add GT!" if not gt_present else "GT already there",
                    )
                    f.write(log_txt)

                out.write("],\n")
            out.write("}\n")


class RescoreSearchErrorJob(Job):
    """
    Take the combined score of N-best hyps and combined score of GTs, compute search error
    """

    def __init__(self, combined_search_py_output: tk.Path, combined_gt_py_output: tk.Path):
        """
        :param combined_search_py_output: rescored search output file from RETURNN in python format (n-best list)
        :param output_gzip: gzip the output
        """
        self.combined_search_py_output = combined_search_py_output
        self.combined_gt_py_output = combined_gt_py_output
        self.out_search_errors = self.output_path("search_errors")

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        n_lists = eval(util.uopen(self.combined_search_py_output, "rt").read(),
                       {"nan": float("nan"), "inf": float("inf")})  # {seq_tag:[(score, hyp)]}
        gts = eval(util.uopen(self.combined_gt_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        seq_tags: List[str] = list(n_lists.keys())
        seq_tags_set: Set[str] = set(seq_tags)
        assert len(seq_tags) == len(seq_tags_set), "duplicate seq tags"
        assert set(gts.keys()) == seq_tags_set, "inconsistent seq tags"
        search_error = 0
        gt_present_num = 0
        gt_absent_num = 0
        for seq_tag in seq_tags:
            is_search_error = False
            gt_present = False

            gt_score, gt = gts[seq_tag][0]
            gt = " ".join(gt.split())
            hyps = [" ".join(x[1].split()) for x in n_lists[seq_tag]]
            targets_search_str = "["
            for hyp in hyps:
                if hyp == gt:
                    targets_search_str += f"\n\t [{hyp}] \n\t"
                    gt_present = True
                    gt_present_num += 1
                else:
                    targets_search_str += f"\n\t {hyp} \n\t"
                    gt_absent_num += 1
            targets_search_str += " ]"
            if len(n_lists[seq_tag]) > 1:
                max_hyp_score = max([x[0] for x in n_lists[seq_tag] if x[1]])
            else:
                max_hyp_score = n_lists[seq_tag][0][0]
            if not gt_present and gt_score >= max_hyp_score:
                search_error += 1
                is_search_error = True

            with open("search_errors_log", "a") as f:
                log_txt = "\nSeq Tag: %s\n\tGround-truth score: %f\n\tMax Search score: %f" % (
                    seq_tag,
                    gt_score,
                    max_hyp_score,
                )
                log_txt += "\n\tGround-truth seq: %s\n\tSearch seq:       %s" % (
                    gt,
                    targets_search_str,
                )
                log_txt += "\n\tGT_present in N-list: %s\n\t-> %s" % (
                    str(gt_present),
                    "Search error!" if is_search_error else "No search error!",
                )
                f.write(log_txt)
        with open("search_errors_log", "a") as f:
            log_txt = "\n\n\tGT presents: %s\n\tNum_seq: %f\n\t" % (
                gt_present_num,
                len(seq_tags),
            )
            f.write(log_txt)
            num_seqs = gt_absent_num + gt_present_num
            f.write("Search errors: %.2f%%" % ((search_error / len(seq_tags)) * 100) + "\n" +
                    "Search errors/total errors: %.2f%%" % ((search_error / gt_absent_num) * 100) + "\n" +
                    "Sent_ER: %.2f%%" % ((search_error / num_seqs) * 100) + "\n")
        with open(self.out_search_errors.get_path(), "w+") as f:
            f.write("Search errors: %.2f%%" % ((search_error / len(seq_tags)) * 100))  # + "\n" +
            # "Search errors/total errors: %.2f%%" % ((num_search_errors / num_unequal) * 100) + "\n" +
            # "Sent_ER: %.2f%%" % ((num_unequal / num_seqs) * 100) + "\n" +
            # "Sent_OOV: %.2f%%" % ((sent_oov / num_seqs) * 100) + "\n" +
            # "OOV: %.2f%%" % ((num_oov / num_words) * 100) + "\n")


def rescore(
    *,
    recog_output: RecogOutput,
    dataset: Optional[DatasetConfig] = None,
    vocab: tk.Path,
    vocab_opts_file: Optional[tk.Path] = None,
    model: ModelWithCheckpoint,
    rescore_def: RescoreDef,
    config: Optional[Dict[str, Any]] = None,
    forward_post_config: Optional[Dict[str, Any]] = None,
    forward_rqmt: Optional[Dict[str, Any]] = None,
    forward_device: str = "gpu",
    forward_alias_name: Optional[str] = None,
    to_word_func: Optional[Callable] = None,
    prior: Optional[Prior] = None,
) -> RecogOutput:
    """
    Rescore on the specific dataset, given some hypotheses, using the :class:`RescoreDef` interface.

    :param recog_output: output from previous recog, with the hyps to rescore
    :param dataset: dataset to forward, using its get_main_dataset(),
        and also get_default_input() to define the default output,
        and get_extern_data().
    :param vocab:
    :param vocab_opts_file: can contain info about EOS, BOS etc
    :param model:
    :param rescore_def:
    :param config: additional RETURNN config opts for the forward job
    :param forward_post_config: additional RETURNN post config (non-hashed) opts for the forward job
    :param forward_rqmt: additional rqmt opts for the forward job (e.g. "time" (in hours), "mem" (in GB))
    :param forward_device: "cpu" or "gpu". if not given, will be "gpu" if model is given, else "cpu"
    :param forward_alias_name: optional alias name for the forward job
    :return: new scores
    """
    env_updates = None
    if (config and config.get("__env_updates")) or (forward_post_config and forward_post_config.get("__env_updates")):
        env_updates = (config and config.pop("__env_updates", None)) or (
            forward_post_config and forward_post_config.pop("__env_updates", None)
        )
    model_ckpt = (model.checkpoint.path if model.checkpoint is not None else None)
    if config:
        model_ckpt = model_ckpt if not config.get("allow_random_model_init", False) else None

    forward_job = ReturnnForwardJobV2(
        model_checkpoint=model_ckpt,
        returnn_config=_returnn_rescore_config(
            recog_output=recog_output,
            vocab=vocab,
            vocab_opts_file=vocab_opts_file,
            dataset=dataset,
            model_def=model.definition,
            rescore_def=rescore_def,
            config=config,
            post_config=forward_post_config,
            to_word_func=to_word_func,
            prior=prior,
        ),
        output_files=[_v2_forward_out_filename],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        device=forward_device,
    )
    forward_job.rqmt["mem"] = 16  # often needs more mem
    if forward_rqmt:
        forward_job.rqmt.update(forward_rqmt)
    if env_updates:
        for k, v in env_updates.items():
            forward_job.set_env(k, v)
    if forward_alias_name:
        forward_job.add_alias(forward_alias_name)
    return RecogOutput(output=forward_job.out_files[_v2_forward_out_filename])


# Those are applied for both training, recog and potential others.
# The values are only used if they are neither set in config nor post_config already.
# They should also not infer with other things from the epilog.
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}


def _returnn_rescore_config(
    *,
    recog_output: RecogOutput,
    vocab: tk.Path,
    vocab_opts_file: Optional[tk.Path] = None,
    dataset: Optional[DatasetConfig] = None,
    model_def: Union[ModelDef, ModelDefWithCfg],
    rescore_def: RescoreDef,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
    to_word_func: Optional[Callable] = None,
    prior: Optional[Prior] = None,
) -> ReturnnConfig:
    """
    Create config for rescoring.
    """
    from returnn.tensor import Tensor, Dim, batch_dim
    from i6_experiments.users.zhang.serialization_v2 import ReturnnConfigWithNewSerialization
    from i6_experiments.users.zeyer.returnn.config import config_dict_update_

    config_ = config
    config = {}

    # Note: we should not put SPM/BPE directly here,
    # because the recog output still has individual labels,
    # so no SPM/BPE encoding on the text.
    vocab_opts = {"class": "Vocabulary", "vocab_file": vocab}
    if vocab_opts_file:
        vocab_opts["special_symbols_via_file"] = vocab_opts_file
    else:
        vocab_opts["unknown_label"] = None

    # Beam dim size unknown. Usually static size, but it's ok to leave this unknown here (right?).
    beam_dim = Dim(Tensor("beam_size", dims=[], dtype="int32"), name="beam")

    data_flat_spatial_dim = Dim(None, name="data_flat_spatial")

    default_input = None  # no input
    forward_data = {"class": "TextDictDataset",
                    "filename": recog_output.output,
                    "vocab": vocab_opts#, "seq_ordering": "default"
                    }
    extern_data = {
        # data_flat dyn dim is the flattened dim, no need to define dim tags now
        "data_flat": {"dims": [batch_dim, data_flat_spatial_dim], "dtype": "int32", "vocab": vocab_opts},
        "data_seq_lens": {"dims": [batch_dim, beam_dim], "dtype": "int32"},
    }
    if dataset:
        ds_extern_data = dataset.get_extern_data()
        default_input = dataset.get_default_input()
        assert default_input in ds_extern_data
        ds_target = dataset.get_default_target()
        for key, value in ds_extern_data.items():
            if key == ds_target:
                continue  # skip (mostly also to keep hashes consistent)
            assert key not in extern_data
            extern_data[key] = value

        # New way for serialization seems does not handle Delayed format correctly?
        orig_data = dataset.get_main_dataset()
        data_path = orig_data.get("path")
        if not isinstance(data_path,str):
            if isinstance(data_path, list):
                if isinstance(data_path[0],str):
                    pass
            else:
                from i6_core.returnn.config import CodeWrapper
                assert isinstance(data_path,CodeWrapper)
                delayed = data_path.code
                from sisyphus.delayed_ops import DelayedFormat
                assert isinstance(delayed,DelayedFormat)
                orig_data["path"] = delayed.kwargs["file"]

        forward_data = {
            "class": "MetaDataset",
            "datasets": {"orig_data": orig_data, "hyps": forward_data},
            "data_map": {
                **{key: ("orig_data", key) for key in ds_extern_data if key != ds_target},
                "data_flat": ("hyps", "data_flat"),
                "data_seq_lens": ("hyps", "data_seq_lens"),
            },
            "seq_order_control_dataset": "hyps",
        }

    config.update(
        {
            "forward_data": forward_data,
            "default_input": default_input,
            "target": "data_flat",  # needed for get_model to know the target dim
            "_beam_dim": beam_dim,
            "_data_flat_spatial_dim": data_flat_spatial_dim,
            "extern_data": extern_data,
        }
    )

    if "backend" not in config:
        config["backend"] = model_def.backend
    config["behavior_version"] = max(model_def.behavior_version, config.get("behavior_version", 0))

    if isinstance(model_def, ModelDefWithCfg):
        config["_model_def"] = model_def.model_def
        config.update(model_def.config)
    else:
        config["_model_def"] = model_def
    config["get_model"] = _returnn_v2_get_model
    config["_rescore_def"] = rescore_def
    config["forward_step"] = _returnn_score_step
    config["forward_callback"] = _returnn_v2_get_forward_callback

    if config_:
        if config_.get("preload_from_files", False):
            from i6_core.returnn.training import ReturnnTrainingJob, Checkpoint as _TfCheckpoint, \
                PtCheckpoint as _PtCheckpoint
            for key, value in config_["preload_from_files"].items():
                if isinstance(value["filename"], _PtCheckpoint):
                    config_["preload_from_files"][key]["filename"] = value["filename"].path#str(value["filename"])
        config_dict_update_(config, config_)

    # post_config is not hashed
    post_config_ = dict(
        log_batch_size=True,
        # debug_add_check_numerics_ops = True
        # debug_add_check_numerics_on_output = True
        torch_log_memory_usage=True,
        watch_memory=True,
        use_lovely_tensors=True,
    )
    if post_config:
        post_config_.update(post_config)
    post_config = post_config_

    batch_size_dependent = False
    if "__batch_size_dependent" in config:
        batch_size_dependent = config.pop("__batch_size_dependent")
    if "__batch_size_dependent" in post_config:
        batch_size_dependent = post_config.pop("__batch_size_dependent")
    for k, v in dict(
        batching="sorted",
        batch_size=(20_000 * model_def.batch_size_factor) if model_def else (20_000 * 160),
        max_seqs=200,
    ).items():
        if k in config:
            v = config.pop(k)
        if k in post_config:
            v = post_config.pop(k)
        (config if batch_size_dependent else post_config)[k] = v

    for k, v in SharedPostConfig.items():
        if k in config or k in post_config:
            continue
        post_config[k] = v
    # When us a word Ngram LM
    if to_word_func:
        config["to_word_func"] = to_word_func
    if prior:
        config["prior"] = prior
    #Hot fix for f16kHz model
    path = "/nas/models/asr/hzhang/setups/2025-07-20--combined/work/i6_core/tools/git/CloneGitRepositoryJob.maXwYTjr7NZe/output/i6_models"
    return ReturnnConfigWithNewSerialization(config, post_config, extra_sys_paths=[path])


def _returnn_score_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    # Similar to i6_experiments.users.zeyer.recog._returnn_v2_forward_step,
    # but using score_def instead of recog_def.
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim
    from returnn.config import get_global_config

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx+1}/{batch_size} seq_tag: {seq_tag!r}")

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    if default_input_key:
        data = extern_data[default_input_key]
        data_spatial_dim = data.get_time_dim_tag()
    else:
        data, data_spatial_dim = None, None

    targets_beam_dim = config.typed_value("_beam_dim")
    targets_flat = extern_data["data_flat"]
    targets_flat_time_dim = config.typed_value("_data_flat_spatial_dim")
    targets_seq_lens = extern_data["data_seq_lens"]  # [B, beam]
    # TODO stupid that targets_seq_lens first is copied CPU->GPU and now back to CPU...
    targets_spatial_dim = Dim(rf.copy_to_device(targets_seq_lens, "cpu"), name="targets_spatial")
    targets = rf.pad_packed(targets_flat, in_dim=targets_flat_time_dim, dims=[targets_beam_dim, targets_spatial_dim])

    rescore_def: RescoreDef = config.typed_value("_rescore_def")
    # # TODO: Skip this static check for now, later should define your own protocol
    # rescore_def = config.typed_value("_rescore_def")
    try:
        to_word_func = config.typed_value("to_word_func")
        prior = config.typed_value("prior")
    except KeyError:
        to_word_func = None
        prior = None
    scores = rescore_def(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        targets_beam_dim=targets_beam_dim,
        to_word_func=to_word_func,
        prior=prior,
    )
    assert isinstance(scores, Tensor)
    rf.get_run_ctx().mark_as_output(targets, "hyps", dims=[batch_dim, targets_beam_dim, targets_spatial_dim])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, targets_beam_dim])
