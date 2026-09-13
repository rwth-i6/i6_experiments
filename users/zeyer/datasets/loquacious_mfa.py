"""
MFA (Montreal Forced Aligner) on the LoquaciousSet HF ogg data.

:class:`MfaAlignLoquaciousSubsetJob` aligns a per-source sample of a split,
optionally with a controlled corruption of the transcripts or the audio,
and keeps MFA's per-utterance diagnostics (``alignment_analysis.csv``, ``unaligned.txt``)
plus the JSON alignments.
Purpose: find out whether MFA's scores separate bad transcripts / bad audio from good ones,
before aligning the full split for the pseudo-speech phone tables.

The MFA invocation follows :class:`...exp2025_07_07_in_grads.jobs.mfa_forced_align.MfaForcedAlignJob`
(corpus on local /tmp, job-local MFA_ROOT_DIR, beams via a config file).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional, Union

from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.datasets.loquacious import loquacious_source_from_id

__all__ = ["MfaAlignLoquaciousSubsetJob", "MfaAlignmentsToHfDatasetJob", "Corruptions"]

# Transcript / audio corruptions of the probe. "none" = the data as is.
Corruptions = [
    "none",
    "wrong_transcript",  # the transcript of another utterance of the same source
    "drop_second_half",  # speech without transcript: only the first half of the words
    "append_other",  # transcript without speech: another utterance's words appended
    "substitute_20pct",  # every 5th word replaced by a random word of the same source
    "noise_0db",  # white noise at 0 dB SNR added to the audio
]


class MfaAlignLoquaciousSubsetJob(Job):
    """``mfa align`` a per-source sample of a LoquaciousSet split, with MFA's per-utterance diagnostics."""

    __sis_hash_exclude__ = {"keep_audio": False}

    def __init__(
        self,
        *,
        hf_data_dir: tk.Path,
        per_source: int,
        corruption: str = "none",
        seed: int = 1,
        mfa_exe: Union[tk.Path, str],
        model_root: tk.Path,
        acoustic_model: str = "english_us_arpa",
        dictionary: str = "english_us_arpa",
        g2p_model: Optional[str] = "english_us_arpa",
        num_jobs: int = 8,
        beam: int = 10,
        retry_beam: int = 400,
        keep_audio: bool = False,
    ):
        """
        :param hf_data_dir: the split dir, with data-*-of-*.arrow shards (audio column = ogg bytes)
        :param per_source: utterances per source
        :param corruption: one of :data:`Corruptions`
        :param seed: sampling and corruption seed
        :param mfa_exe: native ``mfa`` or a containerized wrapper
        :param model_root: MFA model store (:class:`...mfa_forced_align.MfaDownloadModelJob`)
        :param acoustic_model:
        :param dictionary:
        :param g2p_model: pronunciations for OOV words
        :param num_jobs: MFA worker processes
        :param beam: Kaldi beam of the first pass
        :param retry_beam: wider beam for the utterances that failed the first pass
        :param keep_audio: also output the aligned wav files (as fed to MFA, i.e. after the corruption),
            for :class:`MfaAlignmentsToHfDatasetJob`
        """
        super().__init__()
        assert corruption in Corruptions, corruption
        self.hf_data_dir = hf_data_dir
        self.per_source = per_source
        self.corruption = corruption
        self.seed = seed
        self.mfa_exe = mfa_exe
        self.model_root = model_root
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.g2p_model = g2p_model
        self.num_jobs = num_jobs
        self.beam = beam
        self.retry_beam = retry_beam
        self.keep_audio = keep_audio

        self.out_analysis = self.output_path("alignment_analysis.csv")  # MFA's per-utterance diagnostics
        self.out_unaligned = self.output_path("unaligned.txt")  # utterances MFA gave up on
        self.out_utterances = self.output_path("utterances.json")  # uid -> id, source, transcript as aligned
        self.out_alignments = self.output_path("alignments.tar.gz")  # MFA JSON alignments
        self.out_summary = self.output_path("summary.json")  # per source: quantiles of the diagnostics
        self.out_audio = self.output_path("audio.tar") if keep_audio else None  # uid.wav, 16 kHz
        self.rqmt = {"cpu": num_jobs + 1, "mem": 32, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _sample(self, rnd) -> List[Dict[str, str]]:
        """:return: per-source samples, each a dict with id, source, text, audio (ogg bytes)"""
        import pyarrow as pa
        from returnn.datasets.huggingface import get_arrow_shard_files_from_hf_dataset_dir

        # shards are source-contiguous (up to a few boundary shards): the first id names the shard's source
        by_source: Dict[str, List[str]] = {}
        for fn in get_arrow_shard_files_from_hf_dataset_dir(self.hf_data_dir.get_path()):
            with pa.memory_map(fn) as src:
                first_id = pa.ipc.open_stream(src).read_all().column("id")[0].as_py()
            by_source.setdefault(loquacious_source_from_id(first_id), []).append(fn)
        samples = []
        for source, files in sorted(by_source.items()):
            # a few random shards per source, the utterances random within them
            shards = [files[i] for i in rnd.choice(len(files), size=min(4, len(files)), replace=False)]
            rows = []
            for fn in shards:
                with pa.memory_map(fn) as src:
                    t = pa.ipc.open_stream(src).read_all()
                idx = rnd.choice(t.num_rows, size=min(t.num_rows, -(-self.per_source // len(shards))), replace=False)
                for i in sorted(idx.tolist()):
                    row = t.slice(i, 1).to_pylist()[0]
                    rows.append(
                        {"id": row["id"], "source": source, "text": row["text"], "audio": row["audio"]["bytes"]}
                    )
            samples += rows[: self.per_source]
        return samples

    def _corrupt(self, samples: List[Dict[str, str]], rnd):
        """Apply self.corruption in place (text) or return per-sample audio noise flags."""
        by_source: Dict[str, List[int]] = {}
        for i, s in enumerate(samples):
            by_source.setdefault(s["source"], []).append(i)
        if self.corruption == "none" or self.corruption == "noise_0db":
            return
        for source, idxs in by_source.items():
            words_pool = [w for i in idxs for w in samples[i]["text"].split()]
            perm = rnd.permutation(len(idxs))
            for k, i in enumerate(idxs):
                words = samples[i]["text"].split()
                if self.corruption == "wrong_transcript":
                    other = idxs[perm[k]] if idxs[perm[k]] != i else idxs[perm[(k + 1) % len(idxs)]]
                    samples[i]["text"] = samples[other]["text"]
                elif self.corruption == "drop_second_half":
                    samples[i]["text"] = " ".join(words[: max(1, len(words) // 2)])
                elif self.corruption == "append_other":
                    other = idxs[perm[k]] if idxs[perm[k]] != i else idxs[perm[(k + 1) % len(idxs)]]
                    samples[i]["text"] = samples[i]["text"] + " " + samples[other]["text"]
                elif self.corruption == "substitute_20pct":
                    for j in range(0, len(words), 5):
                        words[j] = words_pool[rnd.integers(len(words_pool))]
                    samples[i]["text"] = " ".join(words)
                else:
                    raise ValueError(self.corruption)

    def run(self):
        import io
        import json
        import tarfile
        import tempfile
        import numpy as np
        import soundfile as sf

        rnd = np.random.default_rng(self.seed)
        samples = self._sample(rnd)
        self._corrupt(samples, rnd)

        scratch = tempfile.mkdtemp(prefix="mfa_", dir="/tmp")
        corpus, out_dir, tmp = (os.path.join(scratch, d) for d in ("corpus", "out", "mfa_tmp"))
        for d in (corpus, out_dir, tmp):
            os.makedirs(d, exist_ok=True)
        utterances = {}
        for i, s in enumerate(samples):
            uid = f"u{i:05d}"
            audio, sr = sf.read(io.BytesIO(s["audio"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if self.corruption == "noise_0db":
                noise = rnd.standard_normal(len(audio)).astype("float32")
                audio = audio + noise * (np.sqrt(np.mean(audio**2)) / np.sqrt(np.mean(noise**2)) + 1e-8)
            sf.write(os.path.join(corpus, f"{uid}.wav"), audio, sr)
            with open(os.path.join(corpus, f"{uid}.lab"), "w", encoding="utf-8") as f:
                f.write(s["text"].lower())
            utterances[uid] = {"id": s["id"], "source": s["source"], "text": s["text"], "duration": len(audio) / sr}
        with open(self.out_utterances.get_path(), "w") as f:
            json.dump(utterances, f, indent=1)

        mfa_root = os.path.join(scratch, "mfa_root")
        os.makedirs(mfa_root, exist_ok=True)
        os.symlink(
            os.path.join(os.path.realpath(self.model_root.get_path()), "pretrained_models"),
            os.path.join(mfa_root, "pretrained_models"),
        )
        env = dict(os.environ, MFA_ROOT_DIR=mfa_root, APPTAINERENV_MFA_ROOT_DIR=mfa_root)
        align_config = os.path.join(scratch, "align_config.yaml")
        # output_analysis: MFA 3.4 writes alignment_analysis.csv only with this flag
        with open(align_config, "w") as f:
            f.write(f"beam: {self.beam}\nretry_beam: {self.retry_beam}\noutput_analysis: true\n")
        mfa_exe = self.mfa_exe.get_path() if isinstance(self.mfa_exe, tk.Path) else self.mfa_exe
        cmd = [mfa_exe, "align", "--single_speaker", "--clean", "--output_format", "json", "-t", tmp]
        cmd += ["-j", str(self.num_jobs), "--quiet", "-c", align_config]
        if self.g2p_model:
            cmd += ["--g2p_model_path", self.g2p_model]
        cmd += [corpus, self.dictionary, self.acoustic_model, out_dir]
        print("RUN:", " ".join(cmd), flush=True)
        # cwd on local disk: the job dir is not always visible from a containerized mfa (proot getcwd)
        subprocess.check_call(cmd, env=env, cwd=scratch)

        analysis = os.path.join(out_dir, "alignment_analysis.csv")
        assert os.path.exists(analysis), "MFA wrote no alignment_analysis.csv"
        shutil.copy(analysis, self.out_analysis.get_path())
        unaligned = os.path.join(out_dir, "unaligned.txt")
        with open(self.out_unaligned.get_path(), "w") as f:
            f.write(open(unaligned).read() if os.path.exists(unaligned) else "")
        with tarfile.open(self.out_alignments.get_path(), "w:gz") as tar:
            for fn in sorted(os.listdir(out_dir)):
                if fn.endswith(".json"):
                    tar.add(os.path.join(out_dir, fn), arcname=fn)
        if self.keep_audio:
            with tarfile.open(self.out_audio.get_path(), "w") as tar:
                for fn in sorted(os.listdir(corpus)):
                    if fn.endswith(".wav"):
                        tar.add(os.path.join(corpus, fn), arcname=fn)
        self._write_summary(utterances)
        shutil.rmtree(scratch, ignore_errors=True)

    def _write_summary(self, utterances):
        import csv
        import json
        import numpy as np

        cols = ["overall_log_likelihood", "speech_log_likelihood", "phone_duration_deviation", "snr"]
        per_source: Dict[str, Dict[str, list]] = {}
        with open(self.out_analysis.get_path()) as f:
            for row in csv.DictReader(f):
                uid = row["file"]
                if uid not in utterances:
                    continue
                d = per_source.setdefault(utterances[uid]["source"], {c: [] for c in cols})
                for c in cols:
                    try:
                        d[c].append(float(row[c]))
                    except (TypeError, ValueError):
                        pass
        unaligned = [ln.split()[0] for ln in open(self.out_unaligned.get_path()) if ln.strip()]
        summary = {"corruption": self.corruption, "num_unaligned": len(unaligned), "per_source": {}}
        for source, d in sorted(per_source.items()):
            summary["per_source"][source] = {
                "num": len(d[cols[0]]),
                **{
                    c: [round(float(q), 3) for q in np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])]
                    for c, v in d.items()
                    if v
                },
            }
        with open(self.out_summary.get_path(), "w") as f:
            json.dump(summary, f, indent=1)


class MfaAlignmentsToHfDatasetJob(Job):
    """
    Filter the MFA alignments of :class:`MfaAlignLoquaciousSubsetJob` by MFA's diagnostics
    and write them as an HF dataset in the format of the LibriSpeech MFA alignments
    (``gilkeyio/librispeech-alignments``, see :mod:`...datasets.hf_librispeech_mfa_alignments`),
    so :class:`...ComputeMfaPhoneMeanLogMelJob` and :class:`...ComputeMfaPhoneDurationStatsJob`
    read it unchanged (split ``train``; the audio as a plain float32 array, no decoding backend needed).

    The filter, from the corruption probe on loq medium (see the project notes):
    utterances without MFA scores (alignment failed) are dropped,
    a low ``phone_duration_deviation`` marks transcript errors,
    a low ``snr`` marks noisy audio and speech without transcript.
    The log-likelihoods are not usable as global thresholds.
    """

    def __init__(
        self,
        *,
        align_job: MfaAlignLoquaciousSubsetJob,
        min_phone_duration_deviation: Union[None, float, str] = "q0.05",
        min_snr: Union[None, float, str, Dict[str, Union[float, str]]] = "q0.05",
        drop_unscored: bool = True,
        sample_rate: int = 16_000,
    ):
        """
        :param align_job: with ``keep_audio=True``
        :param min_phone_duration_deviation: reject below.
            A float, or ``"q<frac>"`` for that quantile of the aligned corpus per source
            (MFA's statistic is relative to the corpus's own phone durations, so its scale is corpus dependent).
        :param min_snr: reject below, global or per source, float or quantile as above.
        :param drop_unscored: reject utterances without MFA diagnostics (alignment failed)
        :param sample_rate: of the wav files
        """
        super().__init__()
        assert align_job.keep_audio, "the audio is needed"
        self.align_job = align_job
        self.min_phone_duration_deviation = min_phone_duration_deviation
        self.min_snr = min_snr
        self.drop_unscored = drop_unscored
        self.sample_rate = sample_rate
        self.out_dataset = self.output_path("dataset", directory=True)
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import csv
        import io
        import json
        import tarfile
        import numpy as np
        import soundfile as sf
        from datasets import Dataset, DatasetDict, Features, Value, Sequence

        utterances = json.load(open(self.align_job.out_utterances.get_path()))
        scores = {}
        with open(self.align_job.out_analysis.get_path()) as f:
            for row in csv.DictReader(f):
                try:
                    scores[row["file"]] = {
                        "phone_duration_deviation": float(row["phone_duration_deviation"]),
                        "snr": float(row["snr"]),
                    }
                except (TypeError, ValueError):
                    pass
        rejected = {"unscored": 0, "phone_duration_deviation": 0, "snr": 0}
        by_source = {}
        for uid, u in utterances.items():
            if uid in scores:
                by_source.setdefault(u["source"], []).append(uid)

        def _threshold(spec, metric, source):
            if isinstance(spec, dict):
                spec = spec.get(source)
            if isinstance(spec, str):
                assert spec.startswith("q"), spec
                values = [scores[uid][metric] for uid in by_source.get(source, [])]
                return float(np.quantile(values, float(spec[1:]))) if values else None
            return spec

        thresholds = {
            src: {
                "phone_duration_deviation": _threshold(
                    self.min_phone_duration_deviation, "phone_duration_deviation", src
                ),
                "snr": _threshold(self.min_snr, "snr", src),
            }
            for src in by_source
        }
        kept = []
        for uid, u in utterances.items():
            s = scores.get(uid)
            if s is None:
                rejected["unscored"] += 1
                if self.drop_unscored:
                    continue
            else:
                thr = thresholds[u["source"]]
                if (
                    thr["phone_duration_deviation"] is not None
                    and s["phone_duration_deviation"] < thr["phone_duration_deviation"]
                ):
                    rejected["phone_duration_deviation"] += 1
                    continue
                if thr["snr"] is not None and s["snr"] < thr["snr"]:
                    rejected["snr"] += 1
                    continue
            kept.append(uid)
        kept_set = set(kept)

        alignments = {}
        with tarfile.open(self.align_job.out_alignments.get_path(), "r:gz") as tar:
            for m in tar.getmembers():
                uid = m.name[: -len(".json")]
                if uid in kept_set:
                    alignments[uid] = json.load(tar.extractfile(m))
        audio = {}
        with tarfile.open(self.align_job.out_audio.get_path(), "r") as tar:
            for m in tar.getmembers():
                uid = m.name[: -len(".wav")]
                if uid in kept_set:
                    audio[uid] = tar.extractfile(m).read()

        def _tier(ali, name):
            # MFA json: tiers -> {name: {"type": "interval", "entries": [[start, end, label], ...]}};
            # silence labels dropped as in the LS HF formatting (gaps = silence)
            return [
                {"phoneme" if name == "phones" else "word": lab, "start": float(s), "end": float(e)}
                for s, e, lab in ali["tiers"][name]["entries"]
                if lab not in ("", "sil", "sp", "<eps>")
            ]

        def _gen():
            for uid in kept:
                if uid not in alignments or uid not in audio:
                    continue
                wav, sr = sf.read(io.BytesIO(audio[uid]), dtype="float32")
                assert sr == self.sample_rate, (uid, sr)
                ali = alignments[uid]
                yield {
                    "id": utterances[uid]["id"],
                    "source": utterances[uid]["source"],
                    "text": utterances[uid]["text"],
                    "audio": {"array": wav.tolist(), "sampling_rate": sr},
                    "phonemes": _tier(ali, "phones"),
                    "words": _tier(ali, "words"),
                }

        features = Features(
            {
                "id": Value("string"),
                "source": Value("string"),
                "text": Value("string"),
                "audio": {"array": Sequence(Value("float32")), "sampling_rate": Value("int32")},
                "phonemes": [{"phoneme": Value("string"), "start": Value("float64"), "end": Value("float64")}],
                "words": [{"word": Value("string"), "start": Value("float64"), "end": Value("float64")}],
            }
        )
        ds = Dataset.from_generator(_gen, features=features)
        DatasetDict({"train": ds}).save_to_disk(self.out_dataset.get_path())
        per_source = {}
        for uid in kept:
            src = utterances[uid]["source"]
            per_source[src] = per_source.get(src, 0) + 1
        hours = sum(utterances[uid]["duration"] for uid in kept) / 3600
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(
                {
                    "num_input": len(utterances),
                    "num_kept": len(kept),
                    "num_written": len(ds),
                    "hours_kept": round(hours, 2),
                    "rejected": rejected,
                    "kept_per_source": per_source,
                    "filter": {
                        "min_phone_duration_deviation": self.min_phone_duration_deviation,
                        "min_snr": self.min_snr,
                        "drop_unscored": self.drop_unscored,
                        "thresholds": thresholds,
                    },
                },
                f,
                indent=1,
            )
        assert np.isfinite(hours)
