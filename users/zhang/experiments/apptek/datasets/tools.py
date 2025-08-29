# jobs/oggzip_fix_txt_textual.py
from sisyphus import Job, Task, tk

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
