from __future__ import annotations

from sisyphus.delayed_ops import DelayedBase
#from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob, GnuPlotJob
from typing import Tuple, Dict, Set, List, Optional, Union
from sisyphus import Job, Task, tk, gs
#from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
#import i6_core.util as util
#from i6_experiments.users.zhang.experiments.exp_wer_ppl import EVAL_DATASET_KEYS
#from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import _get_lm_model
#from collections import namedtuple


class SearchCombineScoresJob(Job):
    """
    Takes a number of files, each with the same N-best list, including scores,
    and combines the scores with some weights.
    """

    def __init__(self, search_py_output: List[Tuple[Union[float, DelayedBase], tk.Path]], *, output_gzip: bool = True):
        """
        :param search_py_output: list of tuple (search output file from RETURNN in python format (n-best list), weight)
        :param output_gzip: gzip the output
        """
        assert len(search_py_output) > 0
        self.search_py_output = search_py_output
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))
        self.rqmt = {"time": 1, "cpu": 1, "mem": 12}  # TODO: dynamically determine mem_rqmt

    def tasks(self):
        """task"""
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """
        Stream-merge N-best lists without requiring same seq_tag order or same N-best order.
        - Output preserves the first file's seq_tag order and hypothesis order.
        - Other files are scanned lazily; each block is spilled to a temp .jsonl.gz and looked up by seq_tag.
        - Hypotheses inside a block are aligned by text; sets can be enforced equal or unioned.
        """
        import ast
        import gzip
        import io
        import os
        import re
        import json
        import tempfile
        from typing import Iterator, Tuple, List, Dict

        # ---- knobs ---------------------------------------------------------------
        STRICT_EQUAL_TEXT_SETS = True
        # If True: all files must have identical hypothesis-text sets per seq_tag.
        # If False: we take the union; missing texts in a file contribute 0.0.

        DETECT_DUPLICATE_TEXTS = True

        # If True: error out if a file has duplicate hypothesis texts within the same seq_tag.

        # ---- helpers -------------------------------------------------------------

        def _to_path(p):
            # Try i6/sisyphus Path.get_path(), fall back to str
            return p.get_path() if hasattr(p, "get_path") else str(p)

        # Open .gz or plain text transparently, *text* mode with buffering
        def _open_text(path: str):
            if path.endswith(".gz"):
                return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
            return open(path, "rt", encoding="utf-8")

        # Matches a line starting a block like: 'seq_tag': [
        _seq_start_re = re.compile(r"""^\s*(['"])(?P<tag>.*?)\1\s*:\s*\[\s*$""")

        def _skip_until_open_brace(f):
            """Advance the stream to the first '{' (start of the dict)."""
            for line in f:
                if "{" in line:
                    break

        def _iter_blocks(stream) -> Iterator[Tuple[str, Iterator[Tuple[float, str]]]]:
            """
            Yield (seq_tag, tuple_iter) for each block:
              'seq_tag': [
                (score, text),
                ...
              ],
            Does not materialize the whole list; tuple_iter streams tuples.
            """
            _skip_until_open_brace(stream)

            for line in stream:
                line_strip = line.strip()

                # End of outer dict
                if line_strip.startswith("}"):
                    return

                # Try to match block start
                m = _seq_start_re.match(line)
                if not m:
                    # Lines like empty, comments, or commas between entries
                    continue

                seq_tag = m.group("tag")

                def tuple_iter():
                    # Stream until matching closing "],"
                    for inner in stream:
                        s = inner.strip()
                        if s == "],":
                            return
                        if not s or s == ",":
                            continue
                        # Lines are "(score, text)," – strip trailing comma
                        trailing_comma = s.endswith(",")
                        if trailing_comma:
                            s = s[:-1]

                        # Fast path: single-line tuple "(..., ...)"
                        if s.startswith("(") and s.endswith(")"):
                            tup = ast.literal_eval(s)
                            yield float(tup[0]), tup[1]
                            continue

                        # Fallback: accumulate until balanced parentheses
                        buf = [s]
                        balance = s.count("(") - s.count(")")
                        while balance > 0:
                            nxt = next(stream)
                            t = nxt.strip()
                            if t == "],":
                                # Malformed; bail gracefully
                                return
                            if t.endswith(","):
                                t = t[:-1]
                            buf.append(t)
                            balance += t.count("(") - t.count(")")
                        tup = ast.literal_eval(" ".join(buf))
                        yield float(tup[0]), tup[1]

                yield seq_tag, tuple_iter()

        # Lazy block fetcher that tolerates different seq_tag *order*
        class BlockFetcher:
            """
            Reads a file forward once. Each encountered block is written to a temp .jsonl.gz.
            fetch(seq_tag) returns the temp path (or None if not present).
            """

            def __init__(self, path: str):
                self._stream = _open_text(path)
                self._cache: Dict[str, str] = {}  # seq_tag -> temp file path
                self._finished = False

            def _spill_block(self, seq_tag: str, tuple_iter):
                # Write this block as JSONL: [score, text]
                fd, temp_path = tempfile.mkstemp(suffix=".jsonl.gz")
                os.close(fd)  # we'll reopen with gzip
                with gzip.open(temp_path, "wt", encoding="utf-8") as out:
                    for score, text in tuple_iter:
                        out.write(json.dumps([float(score), text]))
                        out.write("\n")
                self._cache[seq_tag] = temp_path

            def fetch(self, want_tag: str) -> str | None:
                """Return temp path for want_tag, reading forward and caching as needed."""
                if want_tag in self._cache:
                    return self._cache[want_tag]
                if self._finished:
                    return None

                for seq_tag, tuple_iter in _iter_blocks(self._stream):
                    self._spill_block(seq_tag, tuple_iter)
                    if seq_tag == want_tag:
                        return self._cache[seq_tag]

                # EOF
                self._finished = True
                return None

            def close(self):
                try:
                    self._stream.close()
                except Exception:
                    pass

            def cached_paths(self):
                return list(self._cache.values())

        # ---- load weights and file paths ----------------------------------------

        weights: List[float] = []
        files: List[str] = []

        DelayedCls = globals().get("DelayedBase", None)  # avoid NameError if not present

        for weight, fn in self.search_py_output:
            if DelayedCls is not None and isinstance(weight, DelayedCls):
                weight = weight.get()
            if not isinstance(weight, (int, float)):
                raise TypeError(f"invalid weight {weight!r} type {type(weight)}")
            weights.append(float(weight))
            files.append(_to_path(fn))

        if not files:
            raise AssertionError("No input files")

        driver_path = files[0]
        other_paths = files[1:]

        # Build lazy fetchers for non-driver files
        fetchers = [BlockFetcher(p) for p in other_paths]

        # ---- write output --------------------------------------------------------

        tmp_filename = "tmp.py" + (".gz" if self.out_search_results.get_path().endswith(".gz") else "")
        if not tmp_filename.endswith(".gz"):
            raise AssertionError("should have .gz extension")

        try:
            with gzip.open(tmp_filename, "wt", encoding="utf-8") as out:
                out.write("{\n")

                # Drive seq_tag order from the first file
                with _open_text(driver_path) as driver_stream:
                    for seq_tag, driver_tuples in _iter_blocks(driver_stream):
                        # Materialize driver list to keep order; map for O(1) lookups
                        driver_list: List[Tuple[float, str]] = list(driver_tuples)
                        driver_map = {t: s for (s, t) in driver_list}
                        driver_order = [t for _, t in driver_list]
                        driver_text_set = set(driver_order)

                        if DETECT_DUPLICATE_TEXTS and len(driver_text_set) != len(driver_order):
                            raise AssertionError(f"Duplicate hypothesis texts in first file for {seq_tag!r}")

                        # For every other file, fetch this seq_tag block (order-independent)
                        others_dicts: List[Dict[str, float]] = []
                        for fx in fetchers:
                            temp_path = fx.fetch(seq_tag)
                            d: Dict[str, float] = {}
                            if temp_path is not None:
                                with gzip.open(temp_path, "rt", encoding="utf-8") as f:
                                    for line in f:
                                        score_i, text_i = json.loads(line)
                                        if DETECT_DUPLICATE_TEXTS and text_i in d:
                                            raise AssertionError(
                                                f"Duplicate hypothesis text in an input for {seq_tag!r}: {text_i!r}"
                                            )
                                        d[text_i] = float(score_i)
                            # If missing entirely (seq_tag not present), leave as empty dict (treated as 0.0)
                            others_dicts.append(d)

                        # Reconcile sets
                        if STRICT_EQUAL_TEXT_SETS:
                            for idx, d in enumerate(others_dicts, start=1):
                                if set(d.keys()) != driver_text_set:
                                    missing = driver_text_set - set(d.keys())
                                    extra = set(d.keys()) - driver_text_set
                                    raise AssertionError(
                                        f"Different hypothesis text sets for {seq_tag!r} in file #{idx + 1}. "
                                        f"Missing: {len(missing)}, Extra: {len(extra)}"
                                    )
                            final_order = driver_order
                        else:
                            final_order = list(driver_order)
                            seen = set(final_order)
                            # Append extras in first-appearance order across other files
                            for d in others_dicts:
                                for t in d.keys():
                                    if t not in seen:
                                        final_order.append(t)
                                        seen.add(t)

                        # Combine and write block
                        out.write(f"{seq_tag!r}: [\n")
                        for text in final_order:
                            acc = weights[0] * driver_map.get(text, 0.0)
                            for w, d in zip(weights[1:], others_dicts):
                                acc += w * d.get(text, 0.0)
                            out.write(f"({acc!r}, {text!r}),\n")
                        out.write("],\n")

                out.write("}\n")

            # Atomic move into place
            os.replace(tmp_filename, self.out_search_results.get_path())
        finally:
            # Close fetchers and clean up temp files
            for fx in fetchers:
                try:
                    fx.close()
                except Exception:
                    pass
                for p in fx.cached_paths():
                    try:
                        os.remove(p)
                    except Exception:
                        pass

from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from i6_experiments.users.zhang.recog import clean_RecogOut
from i6_core.returnn.search import (
    ReturnnSearchJobV2,
    SearchRemoveLabelJob,
    SearchCollapseRepeatedLabelsJob,
    SearchTakeBestJob,
)
def _spm_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_core.returnn.search import SearchOutputRawReplaceJob

    words = SearchOutputRawReplaceJob(bpe.output, [(" ", ""), ("▁", " ")], output_gzip=True).out_search_results
    return RecogOutput(output=words)

def py():
    # from i6_experiments.users.zhang.experiments.decoding.rescoring import SearchCombineScoresJob as SearchCombineScoresJob1
    # file1 = "/nas/models/asr/hzhang/setups/2025-07-20--combined/work/i6_core/returnn/forward/ReturnnForwardJobV2.9jP8jxwUa00b/output/output.py.gz"
    # file2 = "/nas/models/asr/hzhang/setups/2025-07-20--combined/work/i6_core/returnn/forward/ReturnnForwardJobV2.mpz441VR58h4/output/output.py.gz"
    # file3 = "/nas/models/asr/hzhang/setups/2025-07-20--combined/work/i6_core/returnn/search/SearchRemoveLabelJob.4NMJDjOSJzr2/output/search_results.py.gz"
    # input = [(1.0, tk.Path(file1)), (0.5, tk.Path(file2)), (-0.3, tk.Path(file3))]
    # #job = SearchCombineScoresJob(search_py_output=input)
    # tk.register_output("test/combinescores", SearchCombineScoresJob(search_py_output=input).out_search_results)
    from i6_experiments.users.zhang.experiments.lm.llm import get_llm, LLM_Batch_size_PPL, HuggingFaceLmPerplexityJobV2, LLM_rqmt
    from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import \
        get_corpus_text_dict as ES_get_corpus_text_dict
    input_text_dict = "/nas/models/asr/hzhang/setups/2025-07-20--combined/work/i6_experiments/users/zhang/experiments/decoding/lm_rescoring/LmRescoringJob.1bdD1JpX7Xd0/output/output.py.gz"
    model_id = "microsoft/phi-4"
    llm_config, _ = get_llm(model_ids=[model_id],batch_sizes=[LLM_Batch_size_PPL[model_id]],word_ppl=True,task_name="ES")
    llm_config = llm_config["phi-4"]
    lm_rescore_res = _spm_to_words(clean_RecogOut(RecogOutput(output=tk.Path(input_text_dict)))).output
    ds_name = "test_set.ES.f8kHz.mtp_dev_heldout-v2.ref.ff_wer"
    lm_rescore_res = SearchTakeBestJob(lm_rescore_res).out_best_search_results

    ppl_job = HuggingFaceLmPerplexityJobV2(
        model_dir=llm_config["model_dir"],
        text_file=[ES_get_corpus_text_dict(key=ds_name)],  # get_test_corpus_text(keys=[ds_name])
        batch_size=llm_config["batch_size"],
        lower_case=True,
        word_ppl=True,
        prompt=lm_rescore_res,
        eos_symbol="\n",
        use_prev_context=True, # For now only check for this setting
        context_len_limit=llm_config["ctx_len_limit"],
        llm_name=model_id,
    )
    ppl_job.rqmt.update(LLM_rqmt[model_id])
    tk.register_output(f"test/phi_4_rescor_ppl", ppl_job.out_ppl)
