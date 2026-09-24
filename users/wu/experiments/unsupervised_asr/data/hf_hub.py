"""New in the port (no source counterpart): a pinned, checksummed Hugging Face Hub snapshot as a Job.

The source read three Hugging Face repos straight out of a shared, pre-populated ``HF_HOME`` cache
(``snapshots/*`` globs and hardcoded directories): the LibriSpeech parquet dump
(``openslr/librispeech_asr``), the MFA alignments (``gilkeyio/librispeech-alignments``) and the
wav2vec2 checkpoint (``facebook/wav2vec2-large-lv60``, via another user's download job).  i6_core
has no job that fetches a Hub repo at a fixed revision, so this one does:

* ``huggingface_hub.snapshot_download(repo_id, repo_type=..., revision=<full commit sha>,
  allow_patterns=<the listed files>, cache_dir=<HF_HOME>/hub)`` -- a download into the normal HF cache
  when the Hub is reachable, a pure cache lookup when ``HF_HUB_OFFLINE`` is set;
* every listed file is checked against its sha256 (for LFS files the Hub's own blob id), so a
  different revision or a corrupt cache fails here and not as drifted features downstream;
* ``output/snapshot/<relpath>`` is a symlink to the cache file (no copy of the multi-GB parquets).

The job name starts with ``DownloadHuggingFace`` on purpose: the reference ``settings.py`` routes
jobs of that name to the (online) login node and turns ``HF_HUB_OFFLINE`` off for them.
"""

from __future__ import annotations

from typing import Dict

from sisyphus import Job, Task, tk

__all__ = ["DownloadHuggingFaceSnapshotJob", "sha256_file"]


def sha256_file(path: str, chunk_size: int = 1 << 24) -> str:
    """Hex sha256 of a file, streamed."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class DownloadHuggingFaceSnapshotJob(Job):
    """Fetch the listed files of a Hub repo at a pinned revision and verify their sha256.

    :param repo_id: e.g. ``"openslr/librispeech_asr"``.
    :param repo_type: ``"dataset"`` or ``"model"``.
    :param revision: the full commit sha; branch names are refused (they move).
    :param files: ``{relative path in the repo: sha256 hex}`` -- exactly the files fetched.
    :param hf_home: the Hugging Face home; the cache is ``<hf_home>/hub``.  Give it a
        ``hash_overwrite`` (``default_tools.HF_HOME`` has one) so moving the cache moves no hash.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        files: Dict[str, str],
        hf_home: tk.Path,
    ):
        super().__init__()
        if repo_type not in ("dataset", "model"):
            raise ValueError(f"repo_type must be 'dataset' or 'model', got {repo_type!r}")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"revision must be a full 40-hex commit sha, got {revision!r}")
        if not files:
            raise ValueError("files must list at least one file")
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.files = {k: files[k] for k in sorted(files)}
        self.hf_home = hf_home
        self.out_dir = self.output_path("snapshot", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import os

        from huggingface_hub import snapshot_download

        offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
        snap = snapshot_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            revision=self.revision,
            allow_patterns=list(self.files),
            cache_dir=os.path.join(self.hf_home.get_path(), "hub"),
            local_files_only=offline,
        )
        out = self.out_dir.get_path()
        for rel, want in self.files.items():
            src = os.path.join(snap, rel)
            assert os.path.isfile(src), f"{self.repo_id}@{self.revision}: {rel} missing in {snap}"
            got = sha256_file(src)
            assert got == want, f"{self.repo_id}@{self.revision}: {rel} sha256 {got} != pinned {want}"
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.realpath(src), dst)
            print(f"{rel}: sha256 ok", flush=True)

    def get_file(self, rel: str) -> tk.Path:
        """The output path of one listed file."""
        assert rel in self.files, f"{rel} is not one of the pinned files of {self.repo_id}"
        return self.out_dir.join_right(rel)
