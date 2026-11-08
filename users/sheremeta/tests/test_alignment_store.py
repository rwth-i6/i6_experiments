"""The staged alignment store lives in a per-node namespace, so shared temp directories cannot race."""

import os
import socket

import numpy as np

from i6_experiments.users.sheremeta.datasets import alignment_store


def _write_align_hdf(path):
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=np.array([5, 1, 5, 2, 5], dtype=np.int32))
        f.create_dataset("seqLengths", data=np.array([[5]], dtype=np.int32))
        f.create_dataset("seqTags", data=np.array([b"seq0"]))


def test_stage_base_is_namespaced_by_host(tmp_path, monkeypatch):
    hdf = tmp_path / "align.hdf"
    _write_align_hdf(str(hdf))
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    monkeypatch.setattr(socket, "gethostname", lambda: "node-a")
    base_a = alignment_store._stage_base([str(hdf)])
    monkeypatch.setattr(socket, "gethostname", lambda: "node-b")
    base_b = alignment_store._stage_base([str(hdf)])

    assert base_a != base_b, "two nodes sharing TMPDIR must not share a stage path"
    assert os.path.dirname(base_a) != str(tmp_path), "the stage path must sit in a per-node directory"
    assert base_a.startswith(str(tmp_path)), "the stage path must stay under TMPDIR"
    assert "node-a" in base_a and "node-b" in base_b


def test_staging_creates_the_store_inside_the_node_directory(tmp_path, monkeypatch):
    hdf = tmp_path / "align.hdf"
    _write_align_hdf(str(hdf))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(socket, "gethostname", lambda: "node-a")

    base = alignment_store._stage_base([str(hdf)])
    alignment_store._ensure_staged([str(hdf)], base)
    store = alignment_store.MmapAlignmentStore(base)

    assert store.get("seq0").tolist() == [5, 1, 5, 2, 5]
    assert os.path.isfile(f"{base}.done")
    assert not os.path.isdir(f"{base}.lock")


def _write_many_seqs_hdf(path, num_seqs):
    """
    Writes an alignment HDF of several one-frame sequences.

    :param path: where to write it
    :param num_seqs: number of sequences
    """
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=np.arange(num_seqs, dtype=np.int32))
        f.create_dataset("seqLengths", data=np.ones((num_seqs, 1), dtype=np.int32))
        f.create_dataset("seqTags", data=np.array([f"seq{i}".encode() for i in range(num_seqs)]))


def test_a_holder_whose_lock_was_removed_still_finishes(tmp_path, monkeypatch):
    import shutil

    hdf = tmp_path / "align.hdf"
    _write_align_hdf(str(hdf))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    base = alignment_store._stage_base([str(hdf)])
    stage = alignment_store._stage_store

    def _lose_the_lock_then_stage(paths, base, heartbeat=None, dtype=np.uint16):
        shutil.rmtree(f"{base}.lock")
        stage(paths, base, heartbeat=heartbeat, dtype=dtype)

    monkeypatch.setattr(alignment_store, "_stage_store", _lose_the_lock_then_stage)
    alignment_store._ensure_staged([str(hdf)], base)

    assert alignment_store.MmapAlignmentStore(base).get("seq0").tolist() == [5, 1, 5, 2, 5]
    assert os.path.isfile(f"{base}.done")


def test_the_lock_is_refreshed_while_lengths_and_tags_are_read(tmp_path, monkeypatch):
    import glob

    hdf = tmp_path / "align.hdf"
    _write_many_seqs_hdf(str(hdf), num_seqs=5)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(alignment_store, "_STAGE_ROWS", 2)
    base = alignment_store._stage_base([str(hdf)])
    before_the_copy = []

    def _heartbeat():
        before_the_copy.append(not glob.glob(f"{base}.build*.bin.npy"))

    alignment_store._stage_store([str(hdf)], base, heartbeat=_heartbeat)

    assert sum(before_the_copy) == 3, before_the_copy
    assert alignment_store.MmapAlignmentStore(base).get("seq4").tolist() == [4]


def test_a_failed_pass_leaves_no_build_files(tmp_path, monkeypatch):
    import glob

    import pytest

    hdf = tmp_path / "align.hdf"
    _write_align_hdf(str(hdf))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    base = alignment_store._stage_base([str(hdf)])

    def _fail_during_the_copy():
        if glob.glob(f"{base}.build*.bin.npy"):
            raise RuntimeError("staging interrupted")

    with pytest.raises(RuntimeError):
        alignment_store._stage_store([str(hdf)], base, heartbeat=_fail_during_the_copy)

    assert not glob.glob(f"{base}.build*"), glob.glob(f"{base}.build*")
    assert not os.path.exists(f"{base}.done")
