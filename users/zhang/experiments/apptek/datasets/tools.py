from sisyphus import Job, Task, tk

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
