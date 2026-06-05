from __future__ import annotations

from sisyphus import Job, Task, tk


def _get_path(path):
    return path.get_path() if hasattr(path, "get_path") else str(path)


class OggZipTextToPhonemeHDFJob(Job):
    """
    Convert the text manifest inside an OggZip dataset to deterministic phoneme sequences.

    The lexicon is interpreted deterministically: for every word, the first non-empty
    pronunciation variant is used. The output is a sparse RETURNN-style HDF with
    integer phoneme ids in `inputs`, sequence lengths in `seqLengths`, sequence tags
    in `seqTags`, and the phoneme inventory in `labels`.
    """

    def __init__(
        self,
        *,
        oggzip: tk.Path,
        lexicon: tk.Path,
        output_filename: str = "phoneme_text.hdf",
        dump_audio_hdf: bool = False,
        audio_output_filename: str = "audio.hdf",
        audio_num_splits: int = 10,
        unknown_word_mode: str = "error",
        lowercase: bool = False,
        strip_eow: bool = False,
        progress_interval: int = 10_000,
        mem_rqmt: int = 8,
        time_rqmt: int = 4,
    ):
        if unknown_word_mode not in {"error", "skip"}:
            raise ValueError(f"unknown_word_mode must be 'error' or 'skip', got {unknown_word_mode!r}")
        self.oggzip = oggzip
        self.lexicon = lexicon
        self.unknown_word_mode = unknown_word_mode
        self.lowercase = lowercase
        self.strip_eow = strip_eow
        self.dump_audio_hdf = dump_audio_hdf
        self.audio_num_splits = audio_num_splits
        self.progress_interval = progress_interval
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_hdf = self.output_path(output_filename)
        if dump_audio_hdf and audio_num_splits <= 0:
            raise ValueError(f"audio_num_splits must be positive when dump_audio_hdf=True, got {audio_num_splits}")
        self.audio_output_filename = audio_output_filename
        self.out_audio_hdfs = [
            self.output_path(self._audio_part_filename(audio_output_filename, part_idx, audio_num_splits))
            for part_idx in range(audio_num_splits)
        ] if dump_audio_hdf else []
        self.out_audio_stats = [
            self.output_path(f"audio_part_{part_idx}_stats.txt")
            for part_idx in range(audio_num_splits)
        ] if dump_audio_hdf else []
        self.out_vocab = self.output_path("phoneme_vocab.txt")
        self.out_stats = self.output_path("phoneme_text_stats.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})
        if self.dump_audio_hdf:
            yield Task(
                "run_audio_part",
                args=range(self.audio_num_splits),
                rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt},
            )

    @staticmethod
    def _audio_part_filename(filename: str, part_idx: int, num_parts: int) -> str:
        if "." in filename:
            stem, suffix = filename.rsplit(".", 1)
            return f"{stem}.{part_idx:0{len(str(num_parts - 1))}d}.{suffix}"
        return f"{filename}.{part_idx:0{len(str(num_parts - 1))}d}"

    @staticmethod
    def _open_maybe_gzip(path, mode="rt"):
        import gzip

        path = _get_path(path)
        if path.endswith(".gz"):
            return gzip.open(path, mode, encoding="utf8" if "t" in mode else None)
        return open(path, mode)

    def _normalize_word(self, word: str) -> str:
        return word.lower() if self.lowercase else word

    def _normalize_phoneme(self, phoneme: str) -> str:
        return phoneme.rstrip("#") if self.strip_eow else phoneme

    def _read_lexicon(self):
        import xml.etree.ElementTree as ET

        word_to_pron = {}
        phonemes = []
        phoneme_seen = set()

        with self._open_maybe_gzip(self.lexicon, "rt") as f:
            tree = ET.parse(f)
        root = tree.getroot()

        phonemes = []
        phoneme_seen = set()
        for phoneme_elem in root.findall(".//phoneme-inventory/phoneme"):
            symbol_elem = phoneme_elem.find("symbol")
            if symbol_elem is None:
                continue
            phoneme = self._normalize_phoneme((symbol_elem.text or "").strip())
            if phoneme and phoneme not in phoneme_seen:
                phoneme_seen.add(phoneme)
                phonemes.append(phoneme)

        if not phonemes:
            raise ValueError("Lexicon phoneme-inventory is empty or missing.")

        for lemma in root.findall(".//lemma"):
            orths = [
                self._normalize_word((orth.text or "").strip())
                for orth in lemma.findall("orth")
                if (orth.text or "").strip()
            ]
            if not orths:
                continue

            first_pron = None
            for phon in lemma.findall("phon"):
                pron = [
                    self._normalize_phoneme(token)
                    for token in (phon.text or "").strip().split()
                    if self._normalize_phoneme(token)
                ]
                if pron:
                    first_pron = pron
                    break
            if first_pron is None:
                continue

            for orth in orths:
                if orth not in word_to_pron:
                    word_to_pron[orth] = first_pron

        phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(phonemes)}
        missing_pron_phonemes = sorted(
            {
                phoneme
                for pron in word_to_pron.values()
                for phoneme in pron
                if phoneme not in phoneme_to_idx
            }
        )
        if missing_pron_phonemes:
            raise ValueError(
                "Pronunciations contain phonemes not present in lexicon phoneme-inventory: "
                + ", ".join(missing_pron_phonemes[:50])
            )
        return word_to_pron, phonemes, phoneme_to_idx

    @staticmethod
    def _read_oggzip_manifest(oggzip_path):
        import ast
        import zipfile

        with zipfile.ZipFile(_get_path(oggzip_path), "r") as zf:
            txt_names = [name for name in zf.namelist() if name.endswith(".txt")]
            if len(txt_names) != 1:
                raise ValueError(f"Expected exactly one manifest .txt in OggZip, found {txt_names[:20]}")
            with zf.open(txt_names[0], "r") as f:
                return ast.literal_eval(f.read().decode("utf8"))

    @staticmethod
    def _dump_audio_hdf(*, oggzip_path, manifest, out_path, progress_prefix: str = "audio"):
        import io
        import zipfile

        import h5py
        import numpy as np
        import soundfile as sf

        audio_seqs = []
        seq_tags = []
        sample_rates = set()
        with zipfile.ZipFile(_get_path(oggzip_path), "r") as zf:
            for entry_idx, entry in enumerate(manifest, start=1):
                seq_tag = entry["seq_name"]
                audio_file = "out.ogg/" + entry["file"]
                with zf.open(audio_file, "r") as f:
                    audio, sample_rate = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)
                if audio.ndim == 2:
                    if audio.shape[1] != 1:
                        audio = audio.mean(axis=1)
                    else:
                        audio = audio[:, 0]
                audio_seqs.append(np.asarray(audio, dtype="float32"))
                seq_tags.append(seq_tag)
                sample_rates.add(int(sample_rate))
                if entry_idx % 1000 == 0 or entry_idx == len(manifest):
                    print(f"{progress_prefix}: decoded {entry_idx}/{len(manifest)} utterances", flush=True)

        if len(sample_rates) != 1:
            raise ValueError(f"Expected a single sample rate, got {sorted(sample_rates)}")

        seq_lengths = np.asarray([len(seq) for seq in audio_seqs], dtype="int32")
        total_len = int(seq_lengths.sum())
        inputs = np.zeros((total_len, 1), dtype="float32")
        offset = 0
        for seq in audio_seqs:
            next_offset = offset + int(seq.shape[0])
            inputs[offset:next_offset, 0] = seq
            offset = next_offset

        with h5py.File(out_path, "w") as hdf:
            hdf.create_dataset("inputs", data=inputs)
            hdf.create_dataset("seqLengths", data=seq_lengths[:, None])
            dt = h5py.special_dtype(vlen=bytes)
            hdf.create_dataset(
                "seqTags",
                data=np.asarray([tag.encode("utf8") for tag in seq_tags], dtype=object),
                dtype=dt,
            )
            hdf.create_dataset("labels", data=np.asarray([], dtype="S5"))
            hdf.attrs["inputPattSize"] = 1
            hdf.attrs["numSeqs"] = len(audio_seqs)
            hdf.attrs["numTimesteps"] = total_len
            hdf.attrs["sampleRate"] = sorted(sample_rates)[0]
        return total_len, sorted(sample_rates)[0]

    def _manifest_part(self, manifest, part_idx: int):
        import math

        part_size = int(math.ceil(len(manifest) / float(self.audio_num_splits)))
        start = part_idx * part_size
        end = min(len(manifest), start + part_size)
        return manifest[start:end]

    def run(self):
        import h5py
        import numpy as np

        word_to_pron, phonemes, phoneme_to_idx = self._read_lexicon()
        manifest = self._read_oggzip_manifest(self.oggzip)

        seq_tags = []
        seqs = []
        unknown_counts = {}
        num_words = 0

        for entry_idx, entry in enumerate(manifest, start=1):
            seq_tag = entry["seq_name"]
            words = str(entry.get("text", "")).split()
            seq = []
            for word in words:
                num_words += 1
                norm_word = self._normalize_word(word)
                pron = word_to_pron.get(norm_word)
                if pron is None:
                    unknown_counts[norm_word] = unknown_counts.get(norm_word, 0) + 1
                    if self.unknown_word_mode == "error":
                        continue
                    if self.unknown_word_mode == "skip":
                        continue
                else:
                    seq.extend(phoneme_to_idx[phoneme] for phoneme in pron)
            seq_tags.append(seq_tag)
            seqs.append(np.asarray(seq, dtype="int32"))
            if self.progress_interval > 0 and (
                entry_idx % self.progress_interval == 0 or entry_idx == len(manifest)
            ):
                print(f"converted {entry_idx}/{len(manifest)} utterances", flush=True)

        if unknown_counts and self.unknown_word_mode == "error":
            examples = ", ".join(f"{word}:{count}" for word, count in sorted(unknown_counts.items())[:50])
            raise ValueError(f"Found {len(unknown_counts)} unknown words while converting OggZip text: {examples}")

        seq_lengths = np.asarray([len(seq) for seq in seqs], dtype="int32")
        total_len = int(seq_lengths.sum())
        inputs = np.zeros((total_len,), dtype="int32")
        offset = 0
        for seq in seqs:
            next_offset = offset + int(seq.shape[0])
            inputs[offset:next_offset] = seq
            offset = next_offset

        with h5py.File(self.out_hdf.get_path(), "w") as hdf:
            hdf.create_dataset("inputs", data=inputs)
            hdf.create_dataset("seqLengths", data=seq_lengths[:, None])
            dt = h5py.special_dtype(vlen=bytes)
            hdf.create_dataset(
                "seqTags",
                data=np.asarray([tag.encode("utf8") for tag in seq_tags], dtype=object),
                dtype=dt,
            )
            hdf.create_dataset(
                "labels",
                data=np.asarray([phoneme.encode("utf8") for phoneme in phonemes], dtype=object),
                dtype=dt,
            )
            hdf.attrs["inputPattSize"] = len(phonemes)
            hdf.attrs["numLabels"] = len(phonemes)
            hdf.attrs["numSeqs"] = len(seqs)
            hdf.attrs["numTimesteps"] = total_len

        with open(self.out_vocab.get_path(), "w") as f:
            for idx, phoneme in enumerate(phonemes):
                f.write(f"{idx}\t{phoneme}\n")

        with open(self.out_stats.get_path(), "w") as f:
            f.write(f"oggzip: {_get_path(self.oggzip)}\n")
            f.write(f"lexicon: {_get_path(self.lexicon)}\n")
            f.write(f"num_sequences: {len(seqs)}\n")
            f.write(f"num_words: {num_words}\n")
            f.write(f"num_phoneme_tokens: {total_len}\n")
            f.write(f"num_phoneme_labels: {len(phonemes)}\n")
            f.write("phoneme_index_order: lexicon phoneme-inventory\n")
            f.write(f"progress_interval: {self.progress_interval}\n")
            f.write(f"dump_audio_hdf: {self.dump_audio_hdf}\n")
            f.write(f"audio_num_splits: {self.audio_num_splits if self.dump_audio_hdf else 0}\n")
            for idx, out_audio_hdf in enumerate(self.out_audio_hdfs):
                f.write(f"audio_hdf_part_{idx}: {out_audio_hdf.get_path()}\n")
            f.write(f"unknown_word_mode: {self.unknown_word_mode}\n")
            f.write(f"num_unknown_word_types: {len(unknown_counts)}\n")
            f.write(f"num_unknown_word_tokens: {sum(unknown_counts.values())}\n")
            for word, count in sorted(unknown_counts.items(), key=lambda item: (-item[1], item[0]))[:100]:
                f.write(f"unknown\t{word}\t{count}\n")

    def run_audio_part(self, part_idx: int):
        manifest = self._read_oggzip_manifest(self.oggzip)
        part_manifest = self._manifest_part(manifest, part_idx)
        total_len, sample_rate = self._dump_audio_hdf(
            oggzip_path=self.oggzip,
            manifest=part_manifest,
            out_path=self.out_audio_hdfs[part_idx].get_path(),
            progress_prefix=f"audio part {part_idx}/{self.audio_num_splits}",
        )
        with open(self.out_audio_stats[part_idx].get_path(), "w") as f:
            f.write(f"part_idx: {part_idx}\n")
            f.write(f"num_parts: {self.audio_num_splits}\n")
            f.write(f"num_sequences: {len(part_manifest)}\n")
            f.write(f"audio_num_samples: {total_len}\n")
            f.write(f"audio_sample_rate: {sample_rate}\n")
            f.write(f"audio_hdf: {self.out_audio_hdfs[part_idx].get_path()}\n")
