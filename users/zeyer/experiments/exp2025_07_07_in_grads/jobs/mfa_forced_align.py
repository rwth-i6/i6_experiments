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


def _recombine_to_ref_words(mfa_entries, ref_words):
    """Recombine MFA's word-tier intervals back to the reference words.

    MFA re-tokenizes the gold transcript (it splits hyphenated words, e.g. ``um-hum`` -> ``um``,
    ``hum``, and strips leading/trailing apostrophes, e.g. ``'em`` -> ``em``), so its word tier can
    differ token-by-token from the reference even when the alignment itself is correct. We walk the
    reference words in order and, for each, consume consecutive MFA intervals until their
    de-hyphen/de-apostrophe'd concatenation equals the (de-hyphen/de-apostrophe'd) reference word;
    that word's boundary is ``(first.start, last.end)`` -- MFA's own boundaries, just re-joined.

    :return: per-reference-word ``(start, end)`` list, or ``None`` if MFA's tokens cannot be re-joined
        to the reference 1:1 (a genuine mismatch, surfaced by the caller).
    """

    def _norm(w):
        return str(w).lower().replace("-", "").replace("'", "").strip()

    toks = [(float(s), float(e), _norm(lab)) for s, e, lab in mfa_entries if _norm(lab)]
    out = []
    j = 0
    for rw in ref_words:
        target = _norm(rw)
        if not target:
            return None
        acc, start, end = "", None, None
        while j < len(toks) and acc != target:
            s, e, lab = toks[j]
            if start is None:
                start = s
            end = e
            acc += lab
            j += 1
            if not target.startswith(acc):
                return None  # diverged -> cannot recombine
        if acc != target:
            return None  # ran out of MFA tokens before matching this reference word
        out.append((start, end))
    if j != len(toks):
        return None  # leftover MFA tokens not consumed by any reference word
    return out


class MfaForcedAlignJob(Job):
    """``mfa align`` a dataset's reference transcripts -> per-word boundaries HDF (seconds).

    Output matches :class:`...forced_align_baseline.ForcedAlignBaselineJob`: ``out_hdf`` holds one
    ``[n_words, 2]`` (start, end in seconds) array per sequence tag. Only sequences whose MFA word
    count matches the reference are written; ``out_coverage`` reports the covered fraction.
    """

    # v2: job-local MFA_ROOT_DIR (concurrency-safe model unpack) + default textgrid cleanup (1:1 word
    # counts). Forces a re-run of the v1 attempts (timit errored, buckeye 0-coverage).
    __sis_version__ = 4  # strict per-word count+identity sanity check (no silent skip); see what fails

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
        beam: int = 10,
        retry_beam: int = 400,
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
        # Kaldi alignment beams. Hashed (not excluded) so sweeping them is a deliberate new run.
        self.beam = beam
        self.retry_beam = retry_beam
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

        # Corpus + MFA temp/out are intermediate scratch (thousands of small wav/lab files): keep them
        # on local /tmp (fast NVMe, immediately consistent), NOT on hpcwork/Lustre. Writing many small
        # files to Lustre is slow, burns the shared quota, and races the apptainer container stat'ing
        # freshly-written metadata ("Corpus directory does not exist"). Only the final
        # word_boundaries.hdf (a declared output) lives on hpcwork. Apptainer mounts host /tmp.
        import tempfile
        import shutil

        scratch = tempfile.mkdtemp(prefix="mfa_", dir="/tmp")
        corpus, out_dir, tmp = (os.path.join(scratch, d) for d in ("corpus", "out", "mfa_tmp"))
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
        ref_words = {}  # uid -> reference words (for the strict per-word identity sanity check)
        for i in range(len(ds)):
            uid = f"u{i:06d}"
            sr = int(ds[i]["audio"]["sampling_rate"])
            words = [w.lower() for w in ds[i]["word_detail"]["utterance"]]
            ref_words[uid] = words
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
        mfa_root = os.path.join(scratch, "mfa_root")
        os.makedirs(mfa_root, exist_ok=True)
        _src_pre = os.path.join(os.path.realpath(self.model_root.get_path()), "pretrained_models")
        _dst_pre = os.path.join(mfa_root, "pretrained_models")
        if not os.path.exists(_dst_pre):
            os.symlink(_src_pre, _dst_pre)
        env = dict(os.environ, MFA_ROOT_DIR=mfa_root, APPTAINERENV_MFA_ROOT_DIR=mfa_root, APPTAINER_BIND="/tmp")
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
            # Two-stage beam: --beam is the initial (narrow, fast) pass for the easy majority;
            # --retry_beam is a wider beam applied ONLY to utterances that failed the first pass.
            # The MFA defaults (beam 10 / retry_beam 40) prune the correct path on hard/dense
            # utterances, leaving them "unaligned" (no TextGrid). The Viterbi path always exists -- a
            # wide-enough retry_beam finds it, so every seq aligns (no missing_json), at the cost of the
            # wide search only on the few hard ones.
            "--beam",
            str(self.beam),
            "--retry_beam",
            str(self.retry_beam),
        ]
        if self.g2p_model:
            # auto-generate pronunciations for OOV words (spontaneous Buckeye has many).
            cmd += ["--g2p_model_path", self.g2p_model]
        cmd += [corpus, self.dictionary, self.acoustic_model, out_dir]
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, env=env)

        boundaries_writer = SimpleHDFWriter(self.out_word_boundaries_hdf.get_path(), dim=2, ndim=2)
        n_total = len(ref)
        utt_errs = []
        # Strict: a forced aligner fed the GOLD transcript must reproduce it 1:1. MFA re-tokenizes
        # (hyphen splits, apostrophe stripping), so we recombine its word tier back to the reference
        # words (boundaries are MFA's own, just re-joined). Anything that still does not reproduce the
        # transcript 1:1 (e.g. an utterance MFA failed to align at all) is collected with detail and
        # raised at the end -- never silently dropped.
        problems = []  # (uid, kind, detail)
        for uid, ref_se in ref.items():
            ref_w = ref_words[uid]
            jf = os.path.join(out_dir, f"{uid}.json")
            if not os.path.exists(jf):
                problems.append((uid, "missing_json", f"MFA exported no alignment ({len(ref_w)} ref words)"))
                continue
            entries = json.load(open(jf))["tiers"]["words"]["entries"]
            rec = _recombine_to_ref_words(entries, ref_w)
            if rec is None:
                mfa_words = [str(e[2]) for e in entries]
                problems.append(
                    (uid, "no_recombine", f"mfa={len(mfa_words)} ref={len(ref_w)} MFA={mfa_words} REF={ref_w}")
                )
                continue
            boundaries_writer.insert_batch(np.array([rec], dtype="float32"), [len(rec)], [f"seq-{int(uid[1:])}"])
            utt_errs.append(per_utt_boundary_errors(rec, ref_se))
        boundaries_writer.close()
        shutil.rmtree(scratch, ignore_errors=True)

        if problems:
            print(f"=== MFA SANITY FAILURES: {len(problems)}/{n_total} seqs ===", flush=True)
            for uid, kind, detail in problems[:300]:
                print(f"  [{kind}] {uid}: {detail}", flush=True)
            raise AssertionError(
                f"MFA word boundaries failed the strict sanity check on {len(problems)}/{n_total} seqs."
                " A forced aligner fed the gold transcript must reproduce it 1:1; see the per-seq detail above."
            )
        metrics = aggregate_corpus(utt_errs)
        print("CORPUS METRICS:", metrics)
        self.out_wbe.set(metrics["wbe"])
        self.out_metrics.set(metrics)
        self.out_acc50.set(metrics["acc_50ms"])
        # Strict mode reaches here only if ALL seqs passed -> full coverage.
        self.out_coverage.set({"covered": len(utt_errs), "total": n_total, "fraction": 1.0})
