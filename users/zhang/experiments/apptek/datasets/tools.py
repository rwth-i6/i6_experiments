from sisyphus import Job, Task, tk
import os, gzip, math, shlex, subprocess, xml.etree.ElementTree as ET
from typing import Optional, Dict
import json
import pickle
from typing import Dict, List, Tuple, IO

class BlissStripOrthPunctJob(Job):
    """
    Remove Unicode punctuation from all <orth>...</orth> contents in a Bliss corpus file.
    - Works with .xml or .xml.gz
    - Does not change structure or attributes outside <orth>
    - Inside each <orth>, transforms .text and .tail of all descendants (so nested tags are safe)
    """

    __sis_hash_exclude__ = set()

    def __init__(
        self,
        bliss_in: tk.Path,
        *,
        normalize_spaces: bool = True,
        encoding: str = "utf-8",
        output_name: str = "corpus.xml.gz",
        version: int = 1,
    ):
        self.bliss_in = bliss_in
        self.normalize_spaces = normalize_spaces
        self.encoding = encoding
        self.out_corpus = self.output_path(output_name)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import os, gzip, io, unicodedata, re
        import xml.etree.ElementTree as ET

        in_path = self.bliss_in.get_path()
        out_path = self.out_corpus.get_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        def read_bytes(p):
            if p.endswith(".gz"):
                with gzip.open(p, "rb") as f:
                    return f.read()
            with open(p, "rb") as f:
                return f.read()

        def write_bytes(p, data: bytes):
            if p.endswith(".gz"):
                with gzip.open(p, "wb") as f:
                    f.write(data)
            else:
                with open(p, "wb") as f:
                    f.write(data)

        # --- punctuation predicate (Unicode-aware) ---
        def is_punct(ch: str) -> bool:
            # All categories beginning with 'P' are punctuation: Pc, Pd, Pe, Pf, Pi, Po, Ps
            return unicodedata.category(ch).startswith("P")

        def strip_punct(s: str) -> str:
            if s is None:
                return None
            cleaned = "".join(ch for ch in s if not is_punct(ch))
            if self.normalize_spaces:
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        # --- parse XML ---
        raw = read_bytes(in_path)
        # Keep the XML declaration; ET.fromstring() + .write() will add one if requested
        tree = ET.ElementTree(ET.fromstring(raw.decode(self.encoding)))
        root = tree.getroot()

        # --- transform only <orth> subtrees ---
        n_changed = 0
        for orth in root.iter("orth"):
            # Walk the subtree; update .text and .tail everywhere under <orth>
            for node in orth.iter():
                if node.text:
                    new_text = strip_punct(node.text).lower()
                    if new_text != node.text:
                        node.text = new_text
                        n_changed += 1
                if node.tail:
                    new_tail = strip_punct(node.tail).lower()
                    if new_tail != node.tail:
                        node.tail = new_tail
                        n_changed += 1

        # --- serialize back (UTF-8, with XML decl) ---
        buf = io.BytesIO()
        tree.write(buf, encoding=self.encoding, xml_declaration=True)
        out_bytes = buf.getvalue()
        write_bytes(out_path, out_bytes)

        print(f"[OK] Updated {n_changed} text/tail fields under <orth>. Output -> {out_path}")

class OggZipFixTxtTextualJob(Job):
    """
    Text-only rewrite of the metadata TXT(.gz) inside an OGG/OPUS zip.
    We DO NOT parse or re-serialize Python literals. We just regex-rewrite the
    value of the 'file' field in-place to satisfy Returnn's literal_py_to_pickle.
    """

    __sis_hash_exclude__ = set()

    def __init__(
        self,
        zip_file: tk.Path,
        member_candidates=None,
        encoding: str = "utf-8",
        output_name: str = "out.ogg.zip",
    ):
        self.zip_file = zip_file
        self.member_candidates = member_candidates or [
            "out.ogg.txt",
            "out.ogg.txt.gz",
            "out.opus.txt",
            "out.opus.txt.gz",
            "out.txt",
            "out.txt.gz",
        ]
        self.encoding = encoding
        self.out_ogg_zip = self.output_path(output_name)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import zipfile, gzip, re, os

        in_path = self.zip_file.get_path()
        out_path = self.out_ogg_zip.get_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with zipfile.ZipFile(in_path, "r") as zin:
            members = {zi.filename: zi for zi in zin.infolist()}
            target = next((c for c in self.member_candidates if c in members), None)
            if target is None:
                raise RuntimeError(
                    "No metadata TXT(.gz) found. "
                    f"Candidates={self.member_candidates}. "
                    f"Archive members sample={list(members.keys())[:30]}"
                )

            # --- read raw text (bytes -> maybe gunzip -> text) ---
            raw = zin.read(target)
            is_gz = target.endswith(".gz")
            if is_gz:
                raw = gzip.decompress(raw)
            text = raw.decode(self.encoding, errors="strict")

            # --- regex: capture original quoting and spacing, change only the value ---
            # Matches:  'file'   :   'VALUE'   or   "file": "VALUE"
            # group1 = key+colon+opening quote, group2 = value, group3 = closing quote
            pattern = re.compile(r"""(['"]file['"]\s*:\s*['"])([^'"]+)(['"])""")

            def _fix_value(m):
                prefix, val, suffix = m.group(1), m.group(2), m.group(3)
                new = val.lstrip("/\\")
                new = re.sub(r"/{2,}", "/", new)
                if new != val:
                    return f"{prefix}{new}{suffix}"
                return m.group(0)

            fixed_text, n_subs = pattern.subn(_fix_value, text)

            if n_subs == 0:
                # No change needed; still rewrite zip to be explicit, or you can early-return.
                pass

            # Convert back to bytes (and re-gzip if needed)
            payload = fixed_text.encode(self.encoding)
            if is_gz:
                payload = gzip.compress(payload)

            # --- write new zip, preserving entry attributes ---
            with zipfile.ZipFile(out_path, "w") as zout:
                for name, zi in members.items():
                    data = payload if name == target else zin.read(name)
                    zi2 = zipfile.ZipInfo(filename=zi.filename, date_time=zi.date_time)
                    zi2.compress_type = zi.compress_type
                    zi2.external_attr = zi.external_attr
                    try:
                        zout.writestr(
                            zi2, data,
                            compress_type=zi.compress_type,
                            compresslevel=getattr(zi, "compresslevel", None) or None
                        )
                    except TypeError:
                        zout.writestr(zi2, data, compress_type=zi.compress_type)

            print(f"[OK] Rewrote {n_subs} 'file' occurrences in {target}. Output -> {out_path}")

class FixInfEndInBlissJob(Job):
    """
    Fixes open-ended BLISS segments (end="inf"/"Infinity") by replacing the end time
    with the actual audio duration (via ffprobe). Also clamps nonsensical ends.
    Input and output keep BLISS XML structure; I/O transparently supports .gz files.

    Requires: ffprobe in PATH (from ffmpeg).
    """

    __sis_hash_exclude__ = {}  # nothing dynamic

    def __init__(self, in_corpus: tk.Path):
        """
        :param in_corpus: BLISS XML(.gz) path
        :param out_basename: name of the produced gzipped BLISS
        """
        self.in_corpus = in_corpus
        self.out_corpus = self.output_path("corpus_fixed.xml.gz")

    def tasks(self):
        yield Task("run")

    # ---------- helpers ----------
    @staticmethod
    def _uopen(path: str, mode: str):
        """
        Text/binary auto open; respects .gz suffix.
        """
        if path.endswith(".gz"):
            # mode like "rt"/"wt" or "rb"/"wb"
            return gzip.open(path, mode)
        return open(path, mode)

    @staticmethod
    def _ffprobe_duration(audio_path: str) -> Optional[float]:
        """
        Ask ffprobe for the container duration (seconds, float).
        Returns None if ffprobe fails.
        """
        # format-level duration is robust across codecs/containers
        cmd = f'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 {shlex.quote(audio_path)}'
        try:
            out = subprocess.check_output(cmd, shell=True, text=True).strip()
            dur = float(out)
            if math.isfinite(dur) and dur > 0.0:
                return dur
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_time(x: Optional[str]) -> Optional[float]:
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in {"inf", "infinity"}:
            return math.inf
        try:
            v = float(s)
            return v
        except Exception:
            # weird token → treat as unknown
            return None

    def _iter_recordings(self, root):
        # BLISS topology: <corpus><recording audio="..."><segment .../></recording>...
        for rec in root.findall(".//recording"):
            yield rec

    # ---------- main ----------
    def run(self):
        inp = self.in_corpus.get_path()
        out = self.out_corpus.get_path()

        # Parse XML (works fine for BLISS)
        with self._uopen(inp, "rb") as f:
            tree = ET.parse(f)
        root = tree.getroot()

        # Cache durations per audio file (some recs have many segments)
        dur_cache: Dict[str, Optional[float]] = {}

        changed_segments = 0
        total_segments = 0

        for rec in self._iter_recordings(root):
            audio = rec.get("audio")
            if not audio:
                # nothing we can do—skip
                continue
            if audio not in dur_cache:
                dur_cache[audio] = self._ffprobe_duration(audio)
            dur = dur_cache[audio]

            # If we cannot probe, we leave segments unchanged (but log once)
            if dur is None:
                print(f"[FixInfEndInBlissJob] warn: ffprobe failed for {audio}; leaving segments as-is")
                continue

            dur = float(dur)
            # Walk segments of this recording
            for seg in rec.findall("./segment"):
                total_segments += 1
                start_str = seg.get("start", "0.0")
                end_str = seg.get("end")
                start = self._parse_time(start_str)
                end = self._parse_time(end_str)

                # defaults
                if start is None or not math.isfinite(start) or start < 0:
                    start = 0.0

                need_fix = False
                new_end = None

                # Case A: explicit infinity / invalid end → set to file duration
                if end is None or (isinstance(end, float) and math.isinf(end)):
                    need_fix = True
                    new_end = dur
                else:
                    # finite end provided; validate and clamp
                    # - if end <= start, use file duration
                    # - if end > file duration (allow tiny epsilon), clamp to file duration
                    if (end <= start) or (end > dur + 1e-3) or (not math.isfinite(end)):
                        need_fix = True
                        new_end = min(max(start, 0.0), dur) if end <= start else dur

                if need_fix and new_end is not None:
                    seg.set("end", f"{new_end:.6f}")
                    # ensure start is sane formatting too
                    seg.set("start", f"{start:.6f}")
                    changed_segments += 1

        # Write out gzipped XML
        with self._uopen(out, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)

        print(
            f"[FixInfEndInBlissJob] done. segments total={total_segments}, changed={changed_segments}, "
            f"unique_audios={len(dur_cache)} → {out}"
        )

import re

_SPECIAL_RE = re.compile(r"<[^>]+>")  # matches <s>, </s>, <blank>, etc.
# Example header:
# "# eval_voice_call-v2/zm_website_11eb690b/2-10.70000-19.70000 (10.7-19.7)"
# We want seg_tag: "zm_website_11eb690b/2"
_SEG_HEADER_RE = re.compile(r"^#\s*[^/]+/([^/]+)/(\d+)-")

class ConvertNbestTextToDictJob(Job):
    """
    Parse a text N-best dump from asrmon? into a dict: { seg_tag: [(am_score, hyp), ... 80 items] }.
    - seg_tag extraction per user's rule: "<dataset>/<group>/<idx-...>" -> "<group>/<idx>"
    - keep only AM score (the 2nd float on each N-best line)
    - strip all special tokens like <s>, </s>, <blank>
    - pad to 80 with (-1e-30, "")
    - truncate if > 80
    """

    def __init__(self, in_text: tk.Path, nbest_size: int = 80):
        self.in_text = in_text
        self.nbest_size = int(nbest_size)

        self.out_nbest_dict = self.output_path("out_nbest_dict.py")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 1})

    def _open(self) -> IO[str]:
        path = self.in_text.get_path()
        if path.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return open(path, "rt", encoding="utf-8", errors="replace")

    @staticmethod
    def _normalize_hyp(tokens: List[str]) -> str:
        # Remove all <...> tokens, collapse whitespace.
        words = [t for t in tokens if not _SPECIAL_RE.fullmatch(t)]
        hyp = " ".join(words)
        return " ".join(hyp.split())  # collapse any accidental multi-spaces

    @staticmethod
    def _parse_am_and_tokens(line: str) -> Tuple[float, List[str]]:
        """
        Robustly parse one N-best line. Expected shape:
          total_score  am_score  lm_score  N  <tokens...>
        There can be variable spacing/tabs.
        """
        parts = line.strip().split()
        # We need at least 4 numeric columns before tokens
        # 0: total, 1: am, 2: lm, 3: token_count, 4+: tokens
        if len(parts) < 5:
            raise ValueError(f"N-best line too short: {line!r}")

        # Parse AM score as the second float
        try:
            am_score = float(parts[1])
        except Exception as e:
            raise ValueError(f"Cannot parse AM score from line: {line!r}") from e

        tokens = parts[4:]
        return am_score, tokens

    @staticmethod
    def _seg_tag_from_header(header_line: str) -> str:
        """
        From header like:
          "# eval_voice_call-v2/zm_website_11eb690b/2-10.70000-19.70000 (10.7-19.7)"
        return: "zm_website_11eb690b/2"
        """
        m = _SEG_HEADER_RE.match(header_line.strip())
        if not m:
            raise ValueError(f"Cannot extract seg tag from header: {header_line!r}")
        group, idx = m.group(1), m.group(2)
        return f"{group}/{idx}"

    def run(self):
        current_seg: str | None = None
        current_list: List[Tuple[float, str]] = []

        def flush_current(o):
            if current_seg is None:
                return
            # Enforce exactly self.nbest_size
            if len(current_list) > self.nbest_size:
                trimmed = current_list[: self.nbest_size]
            else:
                trimmed = current_list + [(-1e-30, "")] * (self.nbest_size - len(current_list))
            for score, hyp in trimmed:
                o.write("(%g, %r),\n" % (score, hyp))

        first_entry = True
        with self._open() as f, open(self.out_nbest_dict.get_path(), "wt", encoding="utf-8") as o:
            o.write("{\n")
            for raw in f:
                line = raw.rstrip("\n")

                if not line:
                    continue

                if line.startswith("# "):
                    # Start of a new segment header?
                    if _SEG_HEADER_RE.match(line):
                        # new segment -> flush previous
                        flush_current(o)
                        if not first_entry:
                            o.write("],\n")
                        first_entry = False
                        current_seg = "corpus/" + self._seg_tag_from_header(line)
                        o.write("%r: [\n" % (current_seg,))
                        current_list = []
                    # Other comment lines (# n=..., # am/..., etc.): ignore.
                    continue

                # Hypothesis line: parse scores + tokens
                if current_seg is None:
                    # Defensive: ignore hypothesis lines before any segment header
                    continue

                try:
                    am_score, tokens = self._parse_am_and_tokens(line)
                except ValueError:
                    # Skip malformed lines rather than exploding
                    continue

                hyp = self._normalize_hyp(tokens)
                current_list.append((am_score, hyp))

            # flush last segment
            flush_current(o)
            o.write("],\n")
            o.write("}\n")


def py():
    in_path = tk.Path("/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/"
                      "work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.yobSzRP0Yx57/output/recognition.n-best.1")
    job = ConvertNbestTextToDictJob(in_path, nbest_size=80)
    tk.register_output("test/convert/nbest", job.out_nbest_dict)