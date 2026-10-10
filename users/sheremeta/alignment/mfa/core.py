import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path as SysPath
from typing import Any, Dict, List, Set

from sisyphus import Job, Task, tk

ws_reg = re.compile(r"\s+")

_HELPER_SCRIPT = SysPath(__file__).parent / "mfa_pipeline_helper.py"

_DEFAULT_MFA_OVERRIDES: Dict[str, Any] = {
    "NUM_JOBS": 16,
    "AUTO_SERVER": True,
    "BLAS_NUM_THREADS": 1,  # keep blas single-threaded since mfa parallelizes across NUM_JOBS
    "CLEANUP_TEXTGRIDS": False,
    "DATABASE_LIMITED_MODE": True,
    "DEBUG": False,
    "OVERWRITE": False,
    "QUIET": False,
    "SEED": 0,
    "SINGLE_SPEAKER": True,  # one utterance per file, so skip cross-utterance speaker grouping
    "USE_MP": True,
    "USE_POSTGRES": True,
    "USE_THREADING": True,
    "VERBOSE": False,
}


def _resolve_container_python() -> str:
    # mfa is pinned to python3.10 in the apptainer image, distinct from the host venv running this job
    return subprocess.check_output(["which", "python3.10"], text=True).strip()


def _normalize_orth(text: str) -> str:
    return ws_reg.sub(" ", text.strip())


def _merge_jsonl_gz(inputs: List[str], output: str) -> int:
    """Concatenate JSONL.gz alignment shards, raising on a duplicate seq_tag."""
    seen: Set[str] = set()
    n = 0
    with gzip.open(output, "wt", encoding="utf-8") as out_f:
        for path in inputs:
            with gzip.open(path, "rt", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    tag = json.loads(line)["seq_tag"]
                    if tag in seen:
                        raise RuntimeError(f"duplicate seq_tag across shards: {tag!r}")
                    seen.add(tag)
                    out_f.write(line + "\n")
                    n += 1
    return n


class EnsureMfaModelsJob(Job):
    """Download and cache the MFA acoustic and dictionary models so concurrent align jobs don't race to fetch them."""

    def __init__(
        self,
        acoustic_model: str = "english_us_arpa",
        dictionary: str = "english_us_arpa",
    ):
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.out_done = self.output_var("done")

    def tasks(self):
        # must run as a mini_task on the local (login-node) engine, which has network for the mfa
        # model download fallback when a model is not baked into the container
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 1, "gpu": 0}, mini_task=True)

    def run(self):
        canonical_root = SysPath(os.environ["MFA_ROOT_DIR"])
        canonical_root.mkdir(parents=True, exist_ok=True)
        sif_root = SysPath("/opt/mfa")

        for model_type, model_name in (
            ("acoustic", self.acoustic_model),
            ("dictionary", self.dictionary),
        ):
            dst_dir = canonical_root / "pretrained_models" / model_type
            if dst_dir.exists() and any(dst_dir.glob(f"{model_name}.*")):
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)

            sif_dir = sif_root / "pretrained_models" / model_type
            sif_matches = (
                list(sif_dir.glob(f"{model_name}.*")) if sif_dir.exists() else []
            )
            if sif_matches:
                # prefer the image-baked model to avoid a network download on every fresh root
                shutil.copy2(sif_matches[0], dst_dir / sif_matches[0].name)
            else:
                env = os.environ.copy()
                env["MFA_ROOT_DIR"] = str(canonical_root)
                subprocess.check_call(
                    ["mfa", "model", "download", model_type, model_name],
                    env=env,
                )

            if not any(dst_dir.glob(f"{model_name}.*")):
                raise RuntimeError(
                    f"Failed to bootstrap MFA {model_type} model {model_name!r} into {dst_dir}; "
                    f"tried SIF-baked copy and mfa model download."
                )

        self.out_done.set(True)


class MergeWordAlignmentStoresJob(Job):
    """Merges the word alignment outputs from multiple shards into a final JSONL store."""

    def __init__(self, shards: List[tk.Path]):
        self.shards = shards
        self.out_store = self.output_path("word_alignments.jsonl.gz")
        self.out_num_seqs = self.output_var("num_seqs")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 2, "gpu": 0})

    def run(self):
        n = _merge_jsonl_gz(
            [p.get_path() for p in self.shards], self.out_store.get_path()
        )
        self.out_num_seqs.set(n)


def _store_jsonl_to_sqlite(jsonl_path: str, sqlite_path: str) -> int:
    """Stream a gzipped JSONL word-alignment store into a sqlite table, returns the row count."""
    import sqlite3

    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE alignments (seq_tag TEXT PRIMARY KEY, words TEXT NOT NULL)")
    n = 0
    batch: List[tuple] = []
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            batch.append((rec["seq_tag"], json.dumps(rec["words"])))
            n += 1
            if len(batch) >= 100_000:
                conn.executemany("INSERT INTO alignments VALUES (?, ?)", batch)
                batch = []
    if batch:
        conn.executemany("INSERT INTO alignments VALUES (?, ?)", batch)
    conn.commit()
    conn.close()
    return n


class ConvertWordAlignmentStoreToSqliteJob(Job):
    """Gzipped JSONL word-alignment store -> sqlite keyed by seq_tag, for per-seq lazy lookup.

    Loading the whole store into a python dict OOMs at Loquacious scale, so each worker queries the
    sqlite file read-only per seq_tag instead and memory stays flat.
    """

    def __init__(self, store: tk.Path):
        self.store = store
        self.rqmt = {"cpu": 1, "mem": 8, "time": 4}
        self.out_sqlite = self.output_path("word_alignments.sqlite")
        self.out_num_seqs = self.output_var("num_seqs")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        n = _store_jsonl_to_sqlite(self.store.get_path(), self.out_sqlite.get_path())
        self.out_num_seqs.set(n)


def word_times_hdf_from_jsonl(jsonl_path: str, hdf_path: str, *, sample_rate: int) -> int:
    """
    Writes a word-alignment store as an HDF holding per word its start and end in samples and its label index.

    :param jsonl_path: the gzipped JSONL store, one ``{"seq_tag": ..., "words": [[word, start, end], ...]}`` per line
    :param hdf_path: the HDF to write, its labels are the sorted words of the store
    :param sample_rate: samples per second of the stored times, kept in the ``sample_rate`` attribute of the HDF
    :return: the number of sequences written
    """
    import h5py
    import numpy as np
    from returnn.datasets.hdf import SimpleHDFWriter

    def records():
        """
        :return: the records of the store, one per sequence
        """
        with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    labels = sorted({word for rec in records() for word, _start, _end in rec["words"]})
    index = {word: i for i, word in enumerate(labels)}
    writer = SimpleHDFWriter(filename=hdf_path, dim=len(labels), ndim=1, labels=labels)
    n = 0
    for rec in records():
        values = []
        for word, start, end in rec["words"]:
            times = [int(round(t * sample_rate)) for t in (start, end)]
            assert all(float(np.float32(v / sample_rate)) == t for v, t in zip(times, (start, end))), (
                rec["seq_tag"],
                word,
            )
            values += times + [index[word]]
        data = np.asarray(values, dtype=np.int32)
        writer.insert_batch(data[None, :], seq_len={0: [int(data.shape[0])]}, seq_tag=[rec["seq_tag"]])
        n += 1
    writer.close()
    with h5py.File(hdf_path, "r+") as f:
        f.attrs["sample_rate"] = sample_rate
    return n


class MfaWordStoreToWordTimeHdfJob(Job):
    """
    Converts a gzipped JSONL MFA word-alignment store into the word-time HDF of :func:`word_times_hdf_from_jsonl`.
    """

    def __init__(self, store: tk.Path, *, sample_rate: int):
        """
        :param store: the gzipped JSONL store
        :param sample_rate: samples per second of the stored times
        """
        self.store = store
        self.sample_rate = sample_rate
        self.rqmt = {"cpu": 1, "mem": 8, "time": 4}
        self.out_hdf = self.output_path("word_times.hdf")
        self.out_num_seqs = self.output_var("num_seqs")

    def tasks(self):
        """
        :return: the conversion task
        """
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        """
        Writes the HDF.
        """
        self.out_num_seqs.set(
            word_times_hdf_from_jsonl(self.store.get_path(), self.out_hdf.get_path(), sample_rate=self.sample_rate)
        )


def word_time_hdf_seq_tags(hdf_path: str) -> List[str]:
    """
    :param hdf_path: the word-time HDF of :func:`word_times_hdf_from_jsonl`
    :return: the tags of the sequences it covers, in its order
    """
    import h5py

    with h5py.File(hdf_path, "r") as f:
        return [tag.decode("utf-8") if isinstance(tag, bytes) else str(tag) for tag in f["seqTags"][...].tolist()]


class WordTimeHdfSeqTagsJob(Job):
    """
    Lists the sequences a word-time HDF covers, one tag per line, the segment file of a dataset that reads only them.
    """

    def __init__(self, hdf: tk.Path):
        """
        :param hdf: the word-time HDF
        """
        self.hdf = hdf
        self.out_seq_tags = self.output_path("seq_tags.txt")
        self.out_num_seqs = self.output_var("num_seqs")

    def tasks(self):
        """
        :return: the listing task
        """
        yield Task("run", mini_task=True)

    def run(self):
        """
        Writes the tags.
        """
        tags = word_time_hdf_seq_tags(self.hdf.get_path())
        with open(self.out_seq_tags.get_path(), "w", encoding="utf-8") as f:
            f.writelines(f"{tag}\n" for tag in tags)
        self.out_num_seqs.set(len(tags))


class MfaStoreToTextDictJob(Job):
    """MFA word-alignment store -> a TextDict (out.txt.gz, ``{seq_tag: 'words'}``) for sclite scoring.

    Joins each seq's aligned words (skipping ``<...>`` markers), lowercased, as the WER reference for
    the native-tokenizer arms.
    """

    def __init__(self, store: tk.Path, *, lowercase: bool = True):
        self.store = store
        self.lowercase = lowercase
        self.out_txt = self.output_path("out.txt.gz")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 2, "gpu": 0})

    def run(self):
        with gzip.open(self.store.get_path(), "rt", encoding="utf-8") as f_in, gzip.open(
            self.out_txt.get_path(), "wt", encoding="utf-8"
        ) as f_out:
            f_out.write("{\n")
            for line in f_in:
                rec = json.loads(line)
                words = [
                    w for (w, _b, _e) in rec["words"] if not (w.startswith("<") and w.endswith(">"))
                ]
                text = " ".join(words)
                if self.lowercase:
                    text = text.lower()
                f_out.write(f"{rec['seq_tag']!r}: {text!r},\n")
            f_out.write("}\n")


class _MfaAlignBase(Job):
    """Shared MFA forced-alignment job, parameterized over how the corpus is materialized.

    Subclasses implement ``_materialize`` to stage audio/text into scratch and supply the output
    paths, counts and config attributes referenced here.
    """

    def tasks(self):
        # size the slurm request to mfa's own worker count so each mfa job gets a cpu and ~2g
        n = self.mfa_config_overrides["NUM_JOBS"]
        yield Task("run", rqmt={"cpu": n, "mem": n * 2, "time": 3, "gpu": 0})

    def _scratch_name(self) -> str:
        return "mfa"

    def _scratch_root(self) -> SysPath:
        canonical_root = SysPath(os.environ["MFA_ROOT_DIR"])
        slurm_id = os.environ.get("SLURM_JOB_ID", "local")
        # slurm id + pid keep this path unique so concurrent align jobs never share a root or postgres db
        return (
            canonical_root.parent
            / "s"
            / f"{self._scratch_name()}_{slurm_id}_{os.getpid()}"
        )

    def _mfa_root_shim(self) -> SysPath:
        # postgres rejects unix socket paths over ~107 bytes and mfa puts its socket under
        # MFA_ROOT_DIR (deep in hpcwork scratch), so mfa is handed a short node-local symlink instead
        return SysPath(tempfile.gettempdir()) / f"mfa_{os.getpid()}"

    def _setup_per_job_mfa_root(self, per_job_root: SysPath) -> dict:
        """Build an isolated writable MFA root for this job that symlinks the shared read-only models."""
        per_job_root.mkdir(parents=True, exist_ok=True)
        canonical_root = SysPath(os.environ["MFA_ROOT_DIR"])
        assert (canonical_root / "pretrained_models").exists(), (
            f"Canonical MFA root {canonical_root} has no pretrained_models; "
            f"EnsureMfaModelsJob should have populated it (check MFA_ROOT_DIR in worker_wrapper)."
        )
        for shared in (
            "pretrained_models",
            "extracted_models",
            "global_config.yaml",
            "command_history.yaml",
        ):
            src = canonical_root / shared
            if not src.exists():
                continue
            dst = per_job_root / shared
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        shim = self._mfa_root_shim()
        if shim.is_symlink() or shim.exists():
            shim.unlink()
        shim.symlink_to(per_job_root)
        env = os.environ.copy()
        env["MFA_ROOT_DIR"] = str(shim)
        return env

    def _materialize(self, corpus_dir: SysPath) -> Dict[str, str]:
        """Stage the dataset into ``corpus_dir`` and return the {audio_stem: seq_tag} mapping."""
        raise NotImplementedError

    def _run_helper(
        self,
        corpus_dir: SysPath,
        seq_tag_map_path: SysPath,
        counts_path: SysPath,
        env: dict,
    ) -> None:
        # mfa runs in a separate python3.10 subprocess since it can't share this job's host interpreter
        helper_cmd = [
            _resolve_container_python(),
            str(_HELPER_SCRIPT),
            "--corpus-dir",
            str(corpus_dir),
            "--seq-tag-map",
            str(seq_tag_map_path),
            "--dictionary",
            self.dictionary,
            "--acoustic-model",
            self.acoustic_model,
            "--mfa-config",
            json.dumps(self.mfa_config_overrides, sort_keys=True),
            "--out-alignments",
            self.out_alignments.get_path(),
            "--out-missing",
            self.out_missing.get_path(),
            "--out-counts",
            str(counts_path),
        ]
        if not self.skip_missing:
            helper_cmd.append("--no-skip-missing")
        with open(self.out_align_log.get_path(), "w", encoding="utf-8") as f:
            subprocess.check_call(
                helper_cmd, env=env, stdout=f, stderr=subprocess.STDOUT
            )

    def run(self):
        scratch = self._scratch_root()
        # mfa's clean=True rmtree's MFA_ROOT_DIR/<corpus basename> on startup, so the staged corpus
        # must live next to the per-job mfa root, never inside it, else mfa wipes the corpus first
        per_job_root = scratch / "mfa_root"
        corpus_dir = scratch / "corpus"
        mfa_env = self._setup_per_job_mfa_root(per_job_root)
        seq_tag_map_path = scratch / "seq_tag_map.json"
        counts_path = scratch / "counts.json"

        # breadcrumb symlink so the scratch workspace is reachable from the job dir while running
        scratch_link = SysPath("mfa_scratch")
        if scratch_link.is_symlink():
            scratch_link.unlink()
        if not scratch_link.exists():
            scratch_link.symlink_to(scratch)

        try:
            print(f"materializing corpus into {corpus_dir} ...", flush=True)
            seq_tag_map = self._materialize(corpus_dir)
            print(f"materialized {len(seq_tag_map)} seqs, starting mfa (log: output/align.log)", flush=True)
            seq_tag_map_path.write_text(json.dumps(seq_tag_map))
            self._run_helper(corpus_dir, seq_tag_map_path, counts_path, mfa_env)
            counts = json.loads(counts_path.read_text())
            self.out_num_seqs.set(counts["num_seqs"])
            self.out_num_words.set(counts["num_words"])
        finally:
            # always stop the auto-started postgres server, otherwise it leaks and holds the per-job root
            subprocess.run(["mfa", "server", "stop"], env=mfa_env, check=False)
            shim = self._mfa_root_shim()
            if shim.is_symlink():
                shim.unlink()
            if not self.debug:
                shutil.rmtree(scratch, ignore_errors=True)
                if scratch_link.is_symlink():
                    scratch_link.unlink()
