from __future__ import annotations

import subprocess
import textwrap

from sisyphus import Job, Task, tk


def _get_path(path):
    return path.get_path() if hasattr(path, "get_path") else str(path)


class GeneratePhonemeCipherHDFJob(Job):
    """
    Generate synthetic ciphertext from a phoneme-index HDF.

    For each phoneme p, `num_cipher_labels_per_phoneme` cipher labels are reserved:
    p * M, ..., p * M + M - 1. A random conditional distribution p(cipher|phoneme)
    is sampled independently for every phoneme, then every phoneme token in the
    input HDF is sampled into one of its reserved cipher labels.
    """

    def __init__(
        self,
        *,
        phoneme_hdf: tk.Path,
        phoneme_vocab: tk.Path,
        num_cipher_labels_per_phoneme: int = 3,
        random_seed: int = 1,
        output_filename: str = "cipher_text.hdf",
        distribution_filename: str = "cipher_distribution.npz",
        empirical_distribution_filename: str = "cipher_empirical_distribution.npz",
        hdf_format_version: int = 2,
        mem_rqmt: int = 24,
        time_rqmt: int = 4,
    ):
        if num_cipher_labels_per_phoneme <= 0:
            raise ValueError(
                f"num_cipher_labels_per_phoneme must be positive, got {num_cipher_labels_per_phoneme}"
            )
        self.phoneme_hdf = phoneme_hdf
        self.phoneme_vocab = phoneme_vocab
        self.num_cipher_labels_per_phoneme = num_cipher_labels_per_phoneme
        self.random_seed = random_seed
        self.hdf_format_version = hdf_format_version
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_hdf = self.output_path(output_filename)
        self.out_distribution = self.output_path(distribution_filename)
        self.out_empirical_distribution = self.output_path(empirical_distribution_filename)
        self.out_stats = self.output_path("cipher_generation_stats.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

    @staticmethod
    def _read_vocab(path):
        vocab = []
        with open(_get_path(path), "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx_s, phoneme = line.split(None, 1)
                vocab.append((int(idx_s), phoneme))
        vocab.sort()
        expected = list(range(len(vocab)))
        actual = [idx for idx, _ in vocab]
        if actual != expected:
            raise ValueError(f"Expected contiguous vocab indices {expected[:5]}..., got {actual[:20]}")
        return [phoneme for _, phoneme in vocab]

    def run(self):
        import h5py
        import numpy as np

        phonemes = self._read_vocab(self.phoneme_vocab)
        num_phonemes = len(phonemes)
        num_cipher_per_phone = int(self.num_cipher_labels_per_phoneme)
        num_cipher_labels = num_phonemes * num_cipher_per_phone

        rng = np.random.default_rng(self.random_seed)
        raw_probs = rng.random((num_phonemes, num_cipher_per_phone), dtype=np.float64)
        cond_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
        local_cdf = np.cumsum(cond_probs, axis=1)
        local_cdf[:, -1] = 1.0

        with h5py.File(_get_path(self.phoneme_hdf), "r") as in_hdf:
            inputs = in_hdf["inputs"][:]
            if inputs.ndim > 1:
                inputs = inputs[:, 0]
            inputs = inputs.astype("int64", copy=False)
            if inputs.size and (inputs.min() < 0 or inputs.max() >= num_phonemes):
                raise ValueError(
                    f"Input phoneme ids out of vocab range: min={inputs.min()}, max={inputs.max()}, "
                    f"num_phonemes={num_phonemes}"
                )

            random_values = rng.random(inputs.shape[0], dtype=np.float64)
            sampled_local = (random_values[:, None] > local_cdf[inputs]).sum(axis=1).astype("int64")
            cipher_inputs = (inputs * num_cipher_per_phone + sampled_local).astype("int32")

            counts = np.zeros((num_phonemes, num_cipher_per_phone), dtype="int64")
            np.add.at(counts, (inputs, sampled_local), 1)
            count_sums = counts.sum(axis=1, keepdims=True)
            empirical = np.divide(
                counts,
                count_sums,
                out=np.zeros_like(counts, dtype="float64"),
                where=count_sums > 0,
            )

            with h5py.File(self.out_hdf.get_path(), "w") as out_hdf:
                out_hdf.create_dataset("inputs", data=cipher_inputs)
                seq_lengths = in_hdf["seqLengths"][:]
                if seq_lengths.ndim != 2 or seq_lengths.shape[1] != 1:
                    raise ValueError(f"Expected input seqLengths shape [num_seqs, 1], got {seq_lengths.shape}")
                # Legacy RETURNN HDFDataset creates a dummy "classes" target key
                # when a labels dataset exists, so seqLengths needs one dummy
                # target-length column although there is no targets group.
                out_hdf.create_dataset("seqLengths", data=np.concatenate([seq_lengths, seq_lengths], axis=1))
                dt = h5py.special_dtype(vlen=bytes)
                out_hdf.create_dataset("seqTags", data=in_hdf["seqTags"][:], dtype=dt)
                cipher_labels = np.asarray([str(i).encode("utf8") for i in range(num_cipher_labels)], dtype=object)
                out_hdf.create_dataset("labels", data=cipher_labels, dtype=dt)
                for key, value in in_hdf.attrs.items():
                    out_hdf.attrs[key] = value
                out_hdf.attrs["inputPattSize"] = num_cipher_labels
                out_hdf.attrs["numLabels"] = num_cipher_labels
                out_hdf.attrs["numCipherLabelsPerPhoneme"] = num_cipher_per_phone
                out_hdf.attrs["randomSeed"] = self.random_seed

        cipher_label_ids = np.arange(num_cipher_labels, dtype="int64").reshape(num_phonemes, num_cipher_per_phone)
        np.savez(
            self.out_distribution.get_path(),
            probs=cond_probs,
            cipher_label_ids=cipher_label_ids,
            phoneme_tokens=np.asarray(phonemes, dtype=object),
            num_cipher_labels_per_phoneme=num_cipher_per_phone,
            random_seed=self.random_seed,
        )
        np.savez(
            self.out_empirical_distribution.get_path(),
            probs=empirical,
            counts=counts,
            cipher_label_ids=cipher_label_ids,
            phoneme_tokens=np.asarray(phonemes, dtype=object),
            num_cipher_labels_per_phoneme=num_cipher_per_phone,
        )

        with open(self.out_stats.get_path(), "wt") as f:
            f.write(f"phoneme_hdf: {_get_path(self.phoneme_hdf)}\n")
            f.write(f"phoneme_vocab: {_get_path(self.phoneme_vocab)}\n")
            f.write(f"num_phonemes: {num_phonemes}\n")
            f.write(f"num_cipher_labels_per_phoneme: {num_cipher_per_phone}\n")
            f.write(f"num_cipher_labels: {num_cipher_labels}\n")
            f.write(f"num_tokens: {int(inputs.shape[0])}\n")
            f.write(f"random_seed: {self.random_seed}\n")
            f.write(f"hdf_format_version: {self.hdf_format_version}\n")
            f.write(f"distribution_npz: {self.out_distribution.get_path()}\n")
            f.write(f"empirical_distribution_npz: {self.out_empirical_distribution.get_path()}\n")


class EvaluateCipherTableTransferJob(Job):
    """
    Evaluate a trained table-transfer model on a segment-filtered HDF subset.

    Decoding is one-to-one: each cipher label is mapped to the phoneme with the
    maximum model score p(cipher|phoneme). The reported error rate is the
    corpus-level substitution rate against the reference phoneme HDF.
    """

    def __init__(
        self,
        *,
        model_checkpoint: tk.Path,
        cipher_hdf: tk.Path,
        phoneme_hdf: tk.Path,
        segment_file: tk.Path,
        python_exe: tk.Path,
        output_filename: str = "cipher_table_eval_report.txt",
        mem_rqmt: int = 8,
        time_rqmt: int = 2,
    ):
        self.model_checkpoint = model_checkpoint
        self.cipher_hdf = cipher_hdf
        self.phoneme_hdf = phoneme_hdf
        self.segment_file = segment_file
        self.python_exe = python_exe
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_report = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

    def run(self):
        script = self.output_path("eval_cipher_table_transfer.py")
        script_text = textwrap.dedent(
            f"""
            import h5py
            import torch

            checkpoint_path = {repr(tk.uncached_path(self.model_checkpoint))}
            cipher_hdf_path = {repr(tk.uncached_path(self.cipher_hdf))}
            phoneme_hdf_path = {repr(tk.uncached_path(self.phoneme_hdf))}
            segment_file = {repr(tk.uncached_path(self.segment_file))}
            report_path = {repr(self.out_report.get_path())}

            def _decode_tag(tag):
                return tag.decode("utf8") if isinstance(tag, bytes) else str(tag)

            def _read_hdf_index(path):
                hdf = h5py.File(path, "r")
                seq_lengths = hdf["seqLengths"][:]
                if seq_lengths.ndim != 2:
                    raise ValueError(f"{{path}}: expected 2-D seqLengths, got {{seq_lengths.shape}}")
                data_lens = seq_lengths[:, 0].astype("int64")
                starts = data_lens.cumsum()
                starts = starts - data_lens
                tags = [_decode_tag(tag) for tag in hdf["seqTags"][:]]
                tag_to_idx = {{tag: i for i, tag in enumerate(tags)}}
                return hdf, data_lens, starts, tag_to_idx

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            if any(key.startswith("module.") for key in state_dict):
                state_dict = {{key.removeprefix("module."): value for key, value in state_dict.items()}}
            if "emission_table" not in state_dict:
                raise KeyError(f"checkpoint does not contain emission_table; keys={{list(state_dict)[:20]}}")

            emission_table = state_dict["emission_table"].float()
            log_p_input_given_output = torch.log_softmax(emission_table, dim=0)
            cipher_to_phone = log_p_input_given_output.argmax(dim=1).cpu().numpy()

            with open(segment_file, "rt", encoding="utf8") as f:
                selected_tags = [line.strip() for line in f if line.strip()]

            cipher_hdf, cipher_lens, cipher_starts, cipher_tag_to_idx = _read_hdf_index(cipher_hdf_path)
            phoneme_hdf, phoneme_lens, phoneme_starts, phoneme_tag_to_idx = _read_hdf_index(phoneme_hdf_path)

            total = 0
            errors = 0
            missing = []
            length_mismatches = []
            num_seqs = 0
            try:
                cipher_inputs = cipher_hdf["inputs"]
                phoneme_inputs = phoneme_hdf["inputs"]
                for n, tag in enumerate(selected_tags, start=1):
                    if tag not in cipher_tag_to_idx or tag not in phoneme_tag_to_idx:
                        missing.append(tag)
                        continue
                    ci = cipher_tag_to_idx[tag]
                    pi = phoneme_tag_to_idx[tag]
                    clen = int(cipher_lens[ci])
                    plen = int(phoneme_lens[pi])
                    if clen != plen:
                        length_mismatches.append((tag, clen, plen))
                        continue
                    cstart = int(cipher_starts[ci])
                    pstart = int(phoneme_starts[pi])
                    cipher_seq = cipher_inputs[cstart : cstart + clen]
                    if cipher_seq.ndim > 1:
                        cipher_seq = cipher_seq[:, 0]
                    ref_seq = phoneme_inputs[pstart : pstart + plen]
                    if ref_seq.ndim > 1:
                        ref_seq = ref_seq[:, 0]
                    pred_seq = cipher_to_phone[cipher_seq]
                    errors += int((pred_seq != ref_seq).sum())
                    total += clen
                    num_seqs += 1
                    if n == 1 or n % 500 == 0 or n == len(selected_tags):
                        print(f"evaluated {{n}}/{{len(selected_tags)}} selected seqs", flush=True)
            finally:
                cipher_hdf.close()
                phoneme_hdf.close()

            if missing:
                raise ValueError(f"missing {{len(missing)}} tags, first={{missing[:5]}}")
            if length_mismatches:
                raise ValueError(f"length mismatches {{len(length_mismatches)}}, first={{length_mismatches[:5]}}")

            error_rate = errors / total if total else 0.0
            with open(report_path, "wt", encoding="utf8") as f:
                f.write(f"checkpoint: {{checkpoint_path}}\\n")
                f.write(f"cipher_hdf: {{cipher_hdf_path}}\\n")
                f.write(f"phoneme_hdf: {{phoneme_hdf_path}}\\n")
                f.write(f"segment_file: {{segment_file}}\\n")
                f.write(f"num_selected_sequences: {{len(selected_tags)}}\\n")
                f.write(f"num_evaluated_sequences: {{num_seqs}}\\n")
                f.write(f"total_labels: {{total}}\\n")
                f.write(f"substitution_errors: {{errors}}\\n")
                f.write(f"substitution_error_rate: {{error_rate:.8f}}\\n")
            """
        ).strip() + "\n"

        with open(script.get_path(), "w", encoding="utf-8") as f:
            f.write(script_text)

        subprocess.check_call([tk.uncached_path(self.python_exe), script.get_path()])
