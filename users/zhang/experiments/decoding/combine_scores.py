from __future__ import annotations

from sisyphus.delayed_ops import DelayedBase
#from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob, GnuPlotJob
from typing import Tuple, Dict, Set, List, Optional, Union, Iterator, Any
from sisyphus import Job, Task, tk, gs
#from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
# import i6_core.util as util
#from i6_experiments.users.zhang.experiments.exp_wer_ppl import EVAL_DATASET_KEYS
#from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import _get_lm_model
#from collections import namedtuple
import os, io, gzip, json, ast
from contextlib import ExitStack

GZIP_MAGIC = b"\x1f\x8b"
# ---------- helpers: tiny streaming parser for your current input format ----------
def _smart_text_open(fn: str, mode: str = "rt", encoding: str = "utf-8"):
    """
    Open as gzip if the magic header matches; otherwise open as plain text.
    Works even if the extension is misleading.
    """
    assert "t" in mode, "use text mode (e.g., 'rt')"
    # Open the file once in binary to sniff the magic
    fb = open(fn, "rb")
    head = fb.read(2)
    fb.seek(0)
    if head == GZIP_MAGIC:
        # Let gzip read from the existing file object
        return io.TextIOWrapper(gzip.GzipFile(fileobj=fb, mode="rb"), encoding=encoding)
    else:
        # Wrap the same fb in a text wrapper; no second OS open
        return io.TextIOWrapper(fb, encoding=encoding)

def _next_char(stream: io.TextIOBase) -> str:
    return stream.read(1)


def _consume_ws(stream: io.TextIOBase) -> str:
    """Read until a non-whitespace char or EOF; return that char (or '')."""
    while True:
        c = stream.read(1)
        if not c or not c.isspace():
            return c

def _read_quoted(stream: io.TextIOBase, quote: str) -> str:
    """Read a Python-quoted string literal starting after its first char `quote`."""
    # We were passed the first quote already; include it in the result.
    buf = [quote]
    esc = False
    while True:
        ch = stream.read(1)
        if not ch:
            raise ValueError("EOF while reading quoted string")
        buf.append(ch)
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == quote:
            # Done (we included closing quote)
            return ''.join(buf)

def _read_balanced_list_literal(stream: io.TextIOBase) -> str:
    """
    Read a list literal starting at (or just before) '['.
    Returns the exact text of the list, stopping right after the matching ']'.
    Does not consume any trailing comma/next key.
    """
    # Skip whitespace to first char of the value
    c = _consume_ws(stream)
    if c != '[':
        raise ValueError(f"Expected '[' to start list literal, got {c!r}")

    buf = ['[']
    depth = 1
    in_str = False
    str_quote = ''
    esc = False

    while True:
        ch = stream.read(1)
        if not ch:
            raise ValueError("EOF while reading list literal")
        buf.append(ch)

        if in_str:
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == str_quote:
                in_str = False
            continue

        # not in string
        if ch in ("'", '"'):
            in_str = True
            str_quote = ch
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return ''.join(buf)
        # else: other chars don't matter

def iter_py_dict_items(stream: io.TextIOBase) -> Iterator[Tuple[str, List[Tuple[float, str]]]]:
    """
    Stream (seq_tag, nbest_list) from a top-level Python dict literal like:
      { 'tag': [(score, "hyp"), ...], ... }
    """
    # Expect opening "{"
    c = _consume_ws(stream)
    if c != "{":
        raise ValueError("expected '{' at start of dict")

    def _safe_eval_floaty_literals(text: str):
        # No builtins; only allow inf/nan names.
        env = {"__builtins__": {}, "inf": float("inf"), "nan": float("nan")}
        return eval(text, env, {})
    while True:
        # Next significant: '}' (end) or a quoted key
        c = _consume_ws(stream)
        if not c:
            raise ValueError("unexpected EOF (missing closing '}')")
        if c == "}":
            return
        if c not in ("'", '"'):
            raise ValueError(f"expected quoted key, got {c!r}")

        key_lit = _read_quoted(stream, c)  # includes both quotes
        key = ast.literal_eval(key_lit)

        # Expect colon
        c = _consume_ws(stream)
        if c != ":":
            raise ValueError("expected ':' after key")

        # Read value list literal *exactly*
        val_lit = _read_balanced_list_literal(stream)
        try:
            val = ast.literal_eval(val_lit)
        except Exception:
            # Fallback for lists that may contain inf/-inf/nan
            val = _safe_eval_floaty_literals(val_lit)

        # After value, optionally consume a single trailing comma
        pos = stream.tell()
        c = _consume_ws(stream)
        if c != ",":
            # Not a comma; rewind so the next loop sees '}' or the next key's quote
            try:
                stream.seek(pos)
            except (io.UnsupportedOperation, OSError):
                pass

        yield key, val

# ---------- conversion: Python-dict-literal -> NDJSON(.gz) per input file ----------

def _ndjson_path_for(fn: str) -> str:
    # keep alongside original, but make it deterministic
    base = os.path.basename(fn)
    # drop trailing .gz if present
    if base.endswith(".gz"):
        base = base[:-3]
    # replace common extensions
    for ext in (".py", ".txt", ".json", ".data"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return os.path.join(os.path.dirname(fn), base + ".ndjson.gz")

def _is_probably_ndjson(fn: str) -> bool:
    # very light heuristic: .ndjson(.gz) or .jsonl(.gz)
    bn = os.path.basename(fn)
    return bn.endswith(".ndjson") or bn.endswith(".ndjson.gz") or bn.endswith(".jsonl") or bn.endswith(".jsonl.gz")


# ---------- NDJSON streamer ----------

def iter_ndjson(fn: str) -> Iterator[Tuple[str, List[Tuple[float, str]]]]:
    with _smart_text_open(fn, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # Convert lists back to tuples for downstream code style
            nbest = [(float(s), h) for s, h in obj["nbest"]]
            yield obj["seq_tag"], nbest


class ConvertPyLiteralToNDJSONJob(Job):
    """
    One input file (Python dict literal or .py.gz) -> one output .ndjson.gz
    Output path is stable so downstream jobs can depend on it.

    Guarantees a deterministic global order: lines are sorted by seq_tag.
    """
    def __init__(self, in_file: tk.Path):
        self.in_file = in_file  # tk.Path or string path
        self.out = self.output_path("converted.ndjson.gz")

    def tasks(self):
        yield Task("run")

    def run(self):
        import os, io, gzip, json
        src = self.in_file.get_path()
        dst = self.out.get_path()
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        tmp = dst + ".tmp"

        GZIP_MAGIC = b"\x1f\x8b"

        def _open_text_auto_in(path: str, mode: str = "rt", encoding: str = "utf-8"):
            """
            Robust text opener for inputs: detect gzip by magic, ignore misleading suffixes.
            """
            assert "t" in mode, "Use text mode for readers"
            fb = open(path, "rb")
            head = fb.read(2)
            fb.seek(0)
            if head == GZIP_MAGIC:
                return io.TextIOWrapper(gzip.GzipFile(fileobj=fb, mode="rb"), encoding=encoding)
            # not gzip: wrap the same fb in a text wrapper so we don’t reopen
            return io.TextIOWrapper(fb, encoding=encoding)

        def _open_text_auto_out(path: str, mode: str = "wt", encoding: str = "utf-8"):
            """
            Text opener for outputs: respect .gz suffix; fix mtime for reproducibility.
            """
            if path.endswith(".gz"):
                # Python’s gzip supports mtime=0 for reproducible archives
                gf = gzip.open(path, mode, mtime=0)
                if "t" in mode:
                    return io.TextIOWrapper(gf, encoding=encoding)
                return gf
            return open(path, mode, encoding=encoding)

        # ---- Read all items, then sort by seq_tag to guarantee identical order
        items = []
        with _open_text_auto_in(src, "rt") as inp:
            for seq_tag, nbest in iter_py_dict_items(inp):
                items.append((seq_tag, nbest))

        # Deterministic global order
        items.sort(key=lambda x: x[0])

        # ---- Write as NDJSON (one object per line), keys always in the same order
        with _open_text_auto_out(tmp, "wt") as out:
            for seq_tag, nbest in items:
                # keep nbest order as-is; only tuples->lists + float cast
                nbest_json = [[float(score), hyp] for (score, hyp) in nbest]
                # construct in desired key order
                obj = {"seq_tag": seq_tag, "nbest": nbest_json}
                json.dump(obj, out, ensure_ascii=False, separators=(",", ":"))
                out.write("\n")

        os.replace(tmp, dst)

#less work/i6_experiments/users/zhang/experiments/decoding/rescoring/SearchCombineScoresJob.tRgOj7WckLOE/log.run.1
#work/i6_experiments/users/zhang/experiments/decoding/rescoring/SearchCombineScoresJob.eDm9woEaR1Ju
class SearchCombineScoresJob(Job):
    """
    Expects:
      self.search_py_output: List[Tuple[Union[float, DelayedBase], tk.Path]] # (weight, path_to_input)
    """

    def __init__(self, search_py_output: List[Tuple[Union[float, DelayedBase], tk.Path]], *, output_gzip: bool = True):
        self.search_py_output = search_py_output
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))
        self.rqmt = {"time": 1, "cpu": 1, "mem": 2}
        # --- conversion from py dict to json ---
        converted = []
        for weight, fn in self.search_py_output:
            src = fn
            if _is_probably_ndjson(src):
                # already NDJSON
                converted.append((weight, src))
                continue
            converted.append((weight, ConvertPyLiteralToNDJSONJob(in_file=src).out))
        # replace inputs with ndjson paths
        self.search_ndjson_inputs = converted

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """Streaming merge over NDJSON; writes gzipped Python-dict-literal (your old output)."""
        import gzip

        # Resolve weights and input list
        weights: List[float] = []
        ndjson_files: List[str] = []
        for weight, fn in self.search_ndjson_inputs:
            if isinstance(weight, DelayedBase):
                weight = float(weight.get())
            assert isinstance(weight, (int, float)), f"invalid weight {weight!r} type {type(weight)}"
            weights.append(float(weight))
            ndjson_files.append(fn)

        # Prepare output
        tmp_filename = "tmp.py" + (".gz" if self.out_search_results.get_path().endswith(".gz") else "")
        assert tmp_filename.endswith(".gz"), "should have .gz extension"

        from contextlib import ExitStack
        with ExitStack() as stack, gzip.open(tmp_filename, "wt") as out:
            gens: List[Iterator[Tuple[str, List[Tuple[float, str]]]]] = [
                iter_ndjson(fn) for fn in ndjson_files
            ]

            out.write("{\n")
            first = True
            item_idx = 0

            while True:
                rows: List[Any] = []
                for g in gens:
                    try:
                        rows.append(next(g))
                    except StopIteration:
                        rows.append(None)

                if any(r is None for r in rows):
                    if not all(r is None for r in rows):
                        raise AssertionError("inconsistent number of seq_tags across files")
                    break  # all exhausted simultaneously

                tag0, nb0 = rows[0]
                # enforce same tag order across files for O(1) memory
                for i, (tag_i, nb_i) in enumerate(rows[1:], start=1):
                    if tag_i != tag0:
                        raise AssertionError(f"[Seq tag order] mismatch at item {item_idx}: {tag_i!r} != {tag0!r}")

                # sanity: same n-best length and same hypotheses (order)
                hyp_list0 = [hyp.strip().replace("<unk>", "@@") for _, hyp in nb0]
                for i, (_, nb_i) in enumerate(rows[1:], start=1):
                    if len(nb_i) != len(nb0):
                        raise AssertionError(
                            f"[Same n-best & order] length mismatch at {tag0} file_index {i}, lengths ({len(nb_i)},{len(nb0)})"
                        )
                    hyp_list_i = [hyp.strip().replace("<unk>", "@@") for _, hyp in nb_i]
                    if hyp_list_i != hyp_list0:
                        j = next((j for j, (a, b) in enumerate(zip(hyp_list0, hyp_list_i)) if a != b), -1)
                        mism_pair = (hyp_list0[j] if j >= 0 else None, hyp_list_i[j] if j >= 0 else None)
                        raise AssertionError(
                            f"[Same n-best & order] mismatch at {tag0} file_index {i}, inner index {j} pair {mism_pair}"
                        )

                # write this block
                if not first:
                    out.write(",\n")
                first = False
                out.write(f"{tag0!r}: [\n")

                for hyp_idx in range(len(nb0)):
                    hyp_text = nb0[hyp_idx][1]  # only from first file to avoid dup strings
                    acc = 0.0
                    for file_idx, (_, nb_i) in enumerate(rows):
                        acc += float(nb_i[hyp_idx][0]) * weights[file_idx]
                    out.write(f"({acc!r}, {hyp_text!r}),\n")
                out.write("]")

                item_idx += 1

            out.write("\n}\n")

        os.replace(tmp_filename, self.out_search_results.get_path())