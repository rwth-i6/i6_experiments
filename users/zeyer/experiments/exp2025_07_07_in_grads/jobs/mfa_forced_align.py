"""Montreal Forced Aligner (MFA) baseline.

Two jobs:
- :class:`MfaDownloadModelJob` downloads MFA pretrained models (acoustic / dictionary / g2p) into a
  tracked model store (an ``MFA_ROOT_DIR``), so the align job needs no network at run time.
- :class:`MfaForcedAlignJob` runs ``mfa align`` on a dataset's reference transcripts and writes
  per-word boundaries in the same HDF layout as the other forced-align baselines.

Both take a configurable ``mfa_exe`` (RETURNN-style): a native ``mfa`` binary, or an
Apptainer/Singularity wrapper (see :class:`...jobs.apptainer.ApptainerExeWrapperJob`).
"""

from __future__ import annotations

import os
import sys
import glob
import json
import subprocess
from typing import List, Optional, Tuple, Union

from sisyphus import Job, Task, tk


def _exe(mfa_exe: Union[tk.Path, str]) -> str:
    return mfa_exe.get_path() if isinstance(mfa_exe, tk.Path) else mfa_exe


class MfaDownloadModelJob(Job):
    """Download MFA pretrained models into a tracked model store (mini_task: login-node internet)."""

    def __init__(self, *, mfa_exe: Union[tk.Path, str], models: List[Tuple[str, str]]):
        """
        :param mfa_exe: native ``mfa`` or a containerized wrapper.
        :param models: list of ``(kind, name)``, e.g. ``[("acoustic", "english_us_arpa"),
            ("dictionary", "english_us_arpa"), ("g2p", "english_us_arpa")]``.
        """
        super().__init__()
        self.mfa_exe = mfa_exe
        self.models = list(models)
        self.out_model_root = self.output_path("mfa_root", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # APPTAINERENV_* injects the var INTO the container, overriding the image's own ENV (the MFA
        # image hard-sets MFA_ROOT_DIR=/mfa, a read-only path). Set both so the job works whether
        # mfa_exe is a native binary or an apptainer wrapper.
        # realpath: the setup `work` dir is a multi-hop symlink (home -> hpcwork-pXXXX -> /rwthfs/.../
        # hpcwork) that the container cannot follow; the REAL path is directly bind-mounted, so hand MFA
        # the resolved path (no in-container symlink traversal).
        root = os.path.realpath(self.out_model_root.get_path())
        env = dict(os.environ, MFA_ROOT_DIR=root, APPTAINERENV_MFA_ROOT_DIR=root)
        for kind, name in self.models:
            subprocess.check_call([_exe(self.mfa_exe), "model", "download", kind, name], env=env)


class MfaForcedAlignJob(Job):
    """``mfa align`` a dataset's reference transcripts -> per-word boundaries HDF (seconds).

    Output matches :class:`...forced_align_baseline.ForcedAlignBaselineJob`: ``out_hdf`` holds one
    ``[n_words, 2]`` (start, end in seconds) array per sequence tag. Only sequences whose MFA word
    count matches the reference are written; ``out_coverage`` reports the covered fraction.
    """

    # v2: job-local MFA_ROOT_DIR (concurrency-safe model unpack) + default textgrid cleanup (1:1 word
    # counts). Forces a re-run of the v1 attempts (timit errored, buckeye 0-coverage).
    __sis_version__ = 2

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        mfa_exe: Union[tk.Path, str],
        model_root: tk.Path,
        acoustic_model: str = "english_us_arpa",
        dictionary: str = "english_us_arpa",
        g2p_model: Optional[str] = "english_us_arpa",
        num_jobs: int = 4,
        dataset_offset_factors: int = 1,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.mfa_exe = mfa_exe
        self.model_root = model_root
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.g2p_model = g2p_model
        self.num_jobs = num_jobs
        self.dataset_offset_factors = dataset_offset_factors
        self.returnn_root = returnn_root
        # WBE computed in-job (robust to partial MFA coverage); same align_metrics as the grad-align jobs.
        self.out_wbe = self.output_var("wbe.txt")
        self.out_metrics = self.output_var("metrics.txt")
        self.out_acc50 = self.output_var("acc50.txt")
        self.out_coverage = self.output_var("coverage.txt")
        # Aligned per-word (start, end) in SECONDS (float), [n_words, 2] per covered seq,
        # tags "seq-{idx}". (Jobs finished before this output existed lack the file.)
        self.out_word_boundaries_hdf = self.output_path("word_boundaries.hdf")
        # mem 48: hyp-transcript corpora hit heavy OOV/G2P load (junk hyp words) and
        # OOM-killed a worker at 16G -> MFA multiprocessing deadlocked until walltime.
        # (Forced-mode segA ran in ~13 min within 16G; rqmt changes don't affect the hash.)
        self.rqmt = {"cpu": num_jobs + 1, "mem": 48, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np
        import soundfile as sf
        from i6_experiments.users.zeyer.external_models.huggingface import (
            set_hf_offline_mode,
            get_content_dir_from_hub_cache_dir,
        )

        set_hf_offline_mode()
        import i6_experiments

        # i6_core lives next to i6_experiments in the recipe dir; put it on the path (like the other jobs).
        sys.path.insert(0, os.path.dirname(os.path.dirname(i6_experiments.__file__)))
        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())
        from returnn.datasets.hdf import SimpleHDFWriter
        from datasets import load_dataset

        # realpath: see MfaDownloadModelJob -- the container can't follow the work symlink chain, so
        # operate on the resolved real path (which IS bind-mounted) for all MFA-accessed dirs.
        cwd = os.path.realpath(os.getcwd())
        corpus, out_dir, tmp = (os.path.join(cwd, d) for d in ("corpus", "out", "mfa_tmp"))
        for d in (corpus, out_dir, tmp):
            os.makedirs(d, exist_ok=True)

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        print(f"num seqs: {len(ds)}", flush=True)

        # Build the MFA corpus: one u{i}.wav + u{i}.lab per sequence with a UNIQUE index key (dataset
        # ids can collide -- e.g. TIMIT). Keep the ref word boundaries (seconds) for the in-job WBE.
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
            per_utt_boundary_errors,
            aggregate_corpus,
        )

        ref = {}  # uid -> list of (start, end) seconds
        for i in range(len(ds)):
            uid = f"u{i:06d}"
            sr = int(ds[i]["audio"]["sampling_rate"])
            words = [w.lower() for w in ds[i]["word_detail"]["utterance"]]
            sf.write(os.path.join(corpus, f"{uid}.wav"), np.asarray(ds[i]["audio"]["array"], dtype=np.float32), sr)
            with open(os.path.join(corpus, f"{uid}.lab"), "w") as f:
                f.write(" ".join(words))
            ref[uid] = [
                (s * self.dataset_offset_factors / sr, e * self.dataset_offset_factors / sr)
                for s, e in zip(ds[i]["word_detail"]["start"], ds[i]["word_detail"]["stop"])
            ]

        # Each concurrent align job needs its OWN MFA_ROOT_DIR: MFA unpacks the acoustic archive into
        # MFA_ROOT_DIR/extracted_models, so jobs sharing the download store race on it ("File exists").
        # Symlink the read-only pretrained models into a job-local root; extracted/ stays job-local.
        # APPTAINERENV_* injects MFA_ROOT_DIR into the container; realpath so the container sees the
        # bind-mounted real path, not the unfollowable work symlink.
        mfa_root = os.path.join(cwd, "mfa_root")
        os.makedirs(mfa_root, exist_ok=True)
        _src_pre = os.path.join(os.path.realpath(self.model_root.get_path()), "pretrained_models")
        _dst_pre = os.path.join(mfa_root, "pretrained_models")
        if not os.path.exists(_dst_pre):
            os.symlink(_src_pre, _dst_pre)
        env = dict(os.environ, MFA_ROOT_DIR=mfa_root, APPTAINERENV_MFA_ROOT_DIR=mfa_root)
        # NOTE: default textgrid cleanup (recombine clitics) keeps MFA's word count 1:1 with the
        # reference; --no_textgrid_cleanup splits clitics and breaks the count match.
        cmd = [
            _exe(self.mfa_exe),
            "align",
            "--single_speaker",
            "--clean",
            "--output_format",
            "json",
            "-t",
            tmp,
            "-j",
            str(self.num_jobs),
            "--quiet",
        ]
        if self.g2p_model:
            # auto-generate pronunciations for OOV words (spontaneous Buckeye has many).
            cmd += ["--g2p_model_path", self.g2p_model]
        cmd += [corpus, self.dictionary, self.acoustic_model, out_dir]
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, env=env)

        boundaries_writer = SimpleHDFWriter(self.out_word_boundaries_hdf.get_path(), dim=2, ndim=2)
        n_total, n_covered, n_missing, n_mismatch = len(ref), 0, 0, 0
        utt_errs = []
        for uid, ref_se in ref.items():
            jf = os.path.join(out_dir, f"{uid}.json")
            if not os.path.exists(jf):
                n_missing += 1
                continue
            entries = json.load(open(jf))["tiers"]["words"]["entries"]
            if len(entries) != len(ref_se):
                n_mismatch += 1
                continue
            hyp_se = [(float(e[0]), float(e[1])) for e in entries]
            boundaries_writer.insert_batch(np.array([hyp_se], dtype="float32"), [len(hyp_se)], [f"seq-{int(uid[1:])}"])
            utt_errs.append(per_utt_boundary_errors(hyp_se, ref_se))
            n_covered += 1
        boundaries_writer.close()
        metrics = aggregate_corpus(utt_errs)
        cov = n_covered / max(n_total, 1)
        print(f"coverage {n_covered}/{n_total} = {cov:.3f} (missing {n_missing}, mismatch {n_mismatch})")
        print("CORPUS METRICS:", metrics)
        self.out_wbe.set(metrics["wbe"])
        self.out_metrics.set(metrics)
        self.out_acc50.set(metrics["acc_50ms"])
        self.out_coverage.set(
            {"covered": n_covered, "total": n_total, "fraction": cov, "missing": n_missing, "mismatch": n_mismatch}
        )
