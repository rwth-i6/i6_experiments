from __future__ import annotations

from typing import Tuple, Dict, Set, List, Optional, Union, Iterator, Any
from sisyphus import Job, Task, tk, gs
import os, io, gzip, json, ast
from i6_experiments.users.zhang.experiments.decoding.combine_scores import ConvertPyLiteralToNDJSONJob, _is_probably_ndjson

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
        env = {"__builtins__": {}, "inf": float(1e30), "nan": float("nan")}
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



class MergeNBestJob(Job):
    """
    Merge multiple N-best lists per seq_tag into a single deduplicated N-best per tag.

    Inputs:
      - in_files: List[tk.Path]  # paths to N-best outputs (each either NDJSON or Python-dict-literal)
    Options:
      - score_mode: {"max","sum","avg"}  # how to combine scores when a hyp appears in multiple inputs
      - output_gzip: bool                # write gzipped python-dict-literal (compatible with your old consumers)
      - topk: Optional[int]              # keep only top-k after merge (None => keep all)

    Output:
      - self.out_merged: path to gzipped Python dict literal:
          { "seq_tag": [(score, "hyp"), (score, "hyp"), ...], ... }
        with each N-best sorted by descending score and duplicates removed.
    """

    def __init__(self,
                 in_files: List[tk.Path],
                 *,
                 score_mode: str = "max",
                 output_gzip: bool = True,
                 topk: Optional[int] = None):
        assert len(in_files) >= 1, "need at least one input"
        assert score_mode in {"max", "sum", "avg"}, f"invalid score_mode: {score_mode}"

        self.in_files = in_files
        self.score_mode = score_mode
        self.topk = topk

        self.out_merged = self.output_path("merged_nbest.py" + (".gz" if output_gzip else ""))
        self.rqmt = {"time": 2, "cpu": 1, "mem": 4}

        # Normalize inputs to NDJSON, mirroring your combine job’s pattern
        converted = []
        for fn in self.in_files:
            src = fn
            if _is_probably_ndjson(src):
                converted.append(src)
            else:
                converted.append(ConvertPyLiteralToNDJSONJob(in_file=src).out)
        self.ndjson_inputs = converted

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import gzip, json

        # Small helper: open NDJSON (maybe gz) and yield (tag, [(score, hyp), ...])
        def iter_ndjson_lines(path: str) -> Iterator[Tuple[str, List[Tuple[float, str]]]]:
            def _open_maybe_gz(p, mode):
                return gzip.open(p, mode) if p.endswith(".gz") else open(p, mode)
            with _open_maybe_gz(path, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tag, rows = json.loads(line)
                    # rows is [[score, hyp], ...]
                    yield str(tag), [(float(s), str(h)) for s, h in rows]

        # A normalization for dedup keys; keep original text for output but dedup on normalized form
        def _norm_hyp(s: str) -> str:
            # strip, and avoid treating "<unk>" changes as different
            return s.strip().replace("<unk>", "@@")

        # open all generators in lockstep (enforce same tag order)
        gens: List[Iterator[Tuple[str, List[Tuple[float, str]]]]] = [
            iter_ndjson(fn.get_path()) for fn in self.ndjson_inputs
        ]

        # temp output file (atomic replace at the end)
        tmp_filename = "tmp.py" + (".gz" if self.out_merged.get_path().endswith(".gz") else "")
        assert tmp_filename.endswith(".gz"), "we expect to write gz for consistency"

        with gzip.open(tmp_filename, "wt") as out:
            out.write("{\n")
            first_block = True
            item_idx = 0

            while True:
                rows: List[Optional[Tuple[str, List[Tuple[float, str]]]]] = []
                for g in gens:
                    try:
                        rows.append(next(g))
                    except StopIteration:
                        rows.append(None)

                # All done? Or inconsistent lengths?
                if any(r is None for r in rows):
                    if not all(r is None for r in rows):
                        raise AssertionError("inconsistent number of seq_tags across files; tags must align in order")
                    break

                # Enforce same tag order
                tag0, nb0 = rows[0]
                for i, pair in enumerate(rows[1:], start=1):
                    tag_i, _ = pair
                    if tag_i != tag0:
                        raise AssertionError(f"[Seq tag order] mismatch at item {item_idx}: {tag_i!r} != {tag0!r}")

                # Merge + dedup
                # We aggregate scores by normalized hyp text. For the display text, keep the first appearance.
                agg_scores: Dict[str, float] = {}
                keep_text: Dict[str, str] = {}
                counts: Dict[str, int] = {}

                # Accumulate from all inputs
                for tag_i, nbest_i in rows:
                    for s, hyp in nbest_i:
                        key = _norm_hyp(hyp)
                        if key not in keep_text:  # remember original text from first sighting
                            keep_text[key] = hyp
                        if self.score_mode == "max":
                            agg_scores[key] = max(s, agg_scores.get(key, float("-inf")))
                        elif self.score_mode == "sum":
                            agg_scores[key] = agg_scores.get(key, 0.0) + s
                        elif self.score_mode == "avg":
                            agg_scores[key] = agg_scores.get(key, 0.0) + s
                            counts[key] = counts.get(key, 0) + 1

                if self.score_mode == "avg":
                    for key, total in list(agg_scores.items()):
                        c = counts.get(key, 1)
                        agg_scores[key] = total / float(c)

                # Sort by score desc; tie-breaker = stable by first appearance order using keep_text insertion order
                # Python 3.7+ dict preserves insertion order; we can construct an index for reproducibility.
                first_idx: Dict[str, int] = {k: i for i, k in enumerate(keep_text.keys())}
                items = sorted(agg_scores.items(), key=lambda kv: (-kv[1], first_idx[kv[0]]))

                # Optional truncation
                if self.topk is not None:
                    items = items[: self.topk]

                # Write block
                if not first_block:
                    out.write(",\n")
                first_block = False
                out.write(f"{tag0!r}: [\n")
                for key, score in items:
                    hyp_text = keep_text[key]
                    out.write(f"({score!r}, {hyp_text!r}),\n")
                out.write("]")

                item_idx += 1

            out.write("\n}\n")

        os.replace(tmp_filename, self.out_merged.get_path())
