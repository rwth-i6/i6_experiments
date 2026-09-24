"""New in the port (no source counterpart): the run-time check of the ffmpeg pin.

The package re-encodes the openslr LibriSpeech FLAC to 16 kHz Ogg Vorbis with i6_core's
``BlissChangeEncodingJob`` (:func:`.librispeech.get_bliss_corpus`).  The encoded audio depends on
the ffmpeg build: the reference cluster's conda ffmpeg 7.1.1 reproduces the banked audio sample for
sample, while another build (Lavc61.3.100) moved most dev-other waveforms to 20-100 dB SNR (median
62 dB).  The binary enters the job hash only through a fixed label
(``default_tools.FFMPEG_HASH_OVERWRITE``), so a hash cannot tell two builds apart.  This job does it
at run time instead.

:class:`FfmpegPinCheckJob` encodes EVERY dev-other recording (:data:`NUM_REFERENCE_RECORDINGS`) with
i6_core's own ``BlissChangeEncodingJob`` command for the package's call (:data:`ENCODE_KWARGS`),
decodes each result with the same binary to s16le PCM, and compares each PCM sha256 with the shipped
per-recording list :data:`REFERENCE_LIST` (``ffmpeg_pin_dev_other_pcm_sha256.txt.gz``: one
``<recording name> <sha256>`` line per recording, sorted by name).  A build can change some recordings
and leave others bit-identical (fix-round review: an alternative libvorbis build changed 73% of a
dev-other sample but left 116-288045-0002..0004 bit-identical, and short recordings were unchanged
most often), so no small subset can stand for the corpus.  A mismatch raises with the number of changed
recordings, the first few names and the ``ffmpeg -version`` line.  Its output ``out_ffmpeg_binary`` is
a wrapper script that execs the checked binary; the encode jobs take that wrapper as their
``ffmpeg_binary``, so they cannot run before the check passes.  The job's hash holds the pin label, the
FLAC corpus and the reference list (through its sha256 ``hash_overwrite``), never the binary's real
path: every server whose ffmpeg passes the check shares the downstream hashes.  train-clean-100 is not
checked (about 100 h of audio); dev-other stands for it.

ANOTHER AUDIO GENERATION (``settings.py`` ``FFMPEG_PIN_ACCEPT = "<label>"``, passed as
``accept_label``).  The PCM comparison is bit-exact, so an ffmpeg 7.1.1 on another CPU architecture
(the reference is aarch64) may fail it: its audio really differs.  Failing is the default.  With a
label set, the job still encodes and decodes every recording and writes the PCM sha256 values and the
``ffmpeg -version`` output, but a mismatch does not raise: the report records the same numbers and
marks them ACCEPTED under the label.  The label enters this job's hash (unset, it is excluded, so the
default hash is unchanged), and through the wrapper it moves every downstream ogg, feature, unit and
model hash: results under a label are a different audio generation, never the banked audio, and cannot
share a job with it.

REFERENCE LIST.  Computed on 2026-09-24 on the JUPITER login node (aarch64, CPU) by this module's own
:func:`pcm_digests` (so :func:`encode_recording` and :func:`decode_pcm_sha256`) with
``/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/ffmpeg``
(sha256 ``c5eee15fde4da2bc63f783358a30874c272c5ecac4cf7232fe7312b14274f558``; ``ffmpeg -version``:
``ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers``), the binary that wrote the
banked audio, from the openslr dev-other FLAC files.  It took 72 s on 4
processes.  The same binary decoding the banked HF Ogg bytes of all 2864 recordings
(``TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb``, dev) gives the same 2864 sha256 values, and the
three constants of the previous 3-recording check (116-288045-0000..0002) equal their rows.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import shlex
import stat
import subprocess as sp
from typing import Dict, List, Optional, Sequence, Tuple

from sisyphus import Job, Task, tk

__all__ = [
    "ENCODE_KWARGS",
    "REFERENCE_LIST_FILE",
    "REFERENCE_LIST_SHA256",
    "NUM_REFERENCE_RECORDINGS",
    "REFERENCE_LIST",
    "CHECK_RQMT",
    "read_reference_list",
    "write_reference_list",
    "encode_recording",
    "decode_pcm_sha256",
    "pcm_digests",
    "FfmpegPinCheckJob",
]

#: the ``BlissChangeEncodingJob`` arguments of the package's encode (besides the corpus and binary);
#: :func:`.librispeech.get_bliss_corpus` passes exactly these, and the check encodes with them
ENCODE_KWARGS = {"output_format": "ogg", "codec": "libvorbis", "sample_rate": 16000}

#: the shipped reference list: ``<recording name> <PCM sha256>`` per dev-other recording, sorted by
#: name, gzip (module docstring)
REFERENCE_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "ffmpeg_pin_dev_other_pcm_sha256.txt.gz")
#: sha256 of the shipped file's bytes (the hash of :data:`REFERENCE_LIST`)
REFERENCE_LIST_SHA256 = "5dcff32b153c6c940230d741d77e95d583b427e55471389d2569fac60d1b8712"
#: the number of dev-other recordings in the list (all of dev-other)
NUM_REFERENCE_RECORDINGS = 2864
#: the reference list as the job's input; its hash is the file's sha256, not its path
REFERENCE_LIST = tk.Path(REFERENCE_LIST_FILE,
                         hash_overwrite=f"unsupervised_asr_ffmpeg_pin_dev_other_pcm_sha256_{REFERENCE_LIST_SHA256[:16]}")

#: the check job's requirements: encode plus decode runs at about 58x realtime per core, so the 5.1 h of
#: dev-other take about 5 min on one core
CHECK_RQMT = {"cpu": 4, "mem": 4, "time": 1}

_LINE = re.compile(r"^(\S+) ([0-9a-f]{64})$")


def read_reference_list(path: str) -> Dict[str, str]:
    """``{recording name: PCM sha256}`` from a reference list (gzip if ``path`` ends in ``.gz``);
    raises unless every line is ``<name> <sha256>`` and the names are unique and sorted."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        lines = fh.read().splitlines()
    out: Dict[str, str] = {}
    for i, line in enumerate(lines):
        m = _LINE.match(line)
        if m is None:
            raise ValueError(f"{path}:{i + 1}: not '<recording name> <sha256>': {line!r}")
        out[m.group(1)] = m.group(2)
    names = list(out)
    if len(names) != len(lines) or names != sorted(names):
        raise ValueError(f"{path}: the recording names must be unique and sorted")
    return out


def write_reference_list(digests: Dict[str, str], path: str) -> None:
    """Write ``digests`` in the reference-list format, sorted by name (gzip with a zero mtime, so the
    bytes depend on the content only, if ``path`` ends in ``.gz``)."""
    text = "".join(f"{name} {digests[name]}\n" for name in sorted(digests)).encode()
    if path.endswith(".gz"):
        with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                                                    compresslevel=9) as fh:
            fh.write(text)
    else:
        with open(path, "wb") as fh:
            fh.write(text)


def encode_recording(ffmpeg: str, corpus_file: tk.Path, recording, out_dir: str) -> str:
    """Encode one bliss recording with i6_core's ``BlissChangeEncodingJob`` command for
    :data:`ENCODE_KWARGS` and ``ffmpeg``, into ``out_dir``; returns the encoded file.

    The command is not restated here: an in-memory ``BlissChangeEncodingJob`` is built with the
    package's arguments (its ``__init__`` derives the ffmpeg options), its audio folder is pointed at
    ``out_dir``, and i6_core's own ``_perform_ffmpeg`` builds and runs the command.  The job object is
    never set up or run as a job.
    """
    from i6_core.audio.encoding import BlissChangeEncodingJob

    enc = BlissChangeEncodingJob(corpus_file=corpus_file, **ENCODE_KWARGS,
                                 ffmpeg_binary=tk.Path(ffmpeg), hash_binary=True)
    os.makedirs(out_dir, exist_ok=True)
    enc.out_audio_folder = tk.Path(os.path.abspath(out_dir))
    target = os.path.join(os.path.abspath(out_dir), enc._get_output_filename(recording))
    if os.path.exists(target):
        os.remove(target)  # _perform_ffmpeg skips an existing target
    try:
        enc._perform_ffmpeg(recording)
    except Exception as exc:
        raise RuntimeError(f"ffmpeg pin check: encoding {recording.name} with {ffmpeg} failed ({exc!r})") from exc
    if enc.failed_files or not os.path.isfile(target):
        raise RuntimeError(f"ffmpeg pin check: encoding {recording.name} with {ffmpeg} failed")
    return target


def decode_pcm_sha256(ffmpeg: str, audio: str) -> str:
    """sha256 of ``audio`` decoded by ``ffmpeg`` to raw s16le PCM (native channels and rate)."""
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-threads", "1", "-i", audio,
           "-f", "s16le", "-c:a", "pcm_s16le", "-"]
    res = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg pin check: decoding {audio} failed: {res.stderr.decode(errors='replace')[-2000:]}")
    return hashlib.sha256(res.stdout).hexdigest()


def _digest_one(args: Tuple[str, str, str, str, str]) -> Tuple[str, str]:
    """Worker of :func:`pcm_digests`: encode one recording, hash its PCM decode, remove the encode."""
    from i6_core.lib import corpus

    ffmpeg, corpus_path, name, audio, out_dir = args
    rec = corpus.Recording()
    rec.name, rec.audio = name, audio
    ogg = encode_recording(ffmpeg, tk.Path(corpus_path), rec, out_dir)
    try:
        return name, decode_pcm_sha256(ffmpeg, ogg)
    finally:
        os.remove(ogg)


def pcm_digests(ffmpeg: str, corpus_file: tk.Path, recordings: Sequence, out_dir: str,
                num_procs: int) -> Dict[str, str]:
    """``{recording name: PCM sha256}`` of each recording's encode (:func:`encode_recording` then
    :func:`decode_pcm_sha256`), over ``num_procs`` processes (as i6_core's own encode job, a
    ``multiprocessing`` pool).  Each encode is removed once hashed."""
    from multiprocessing import pool

    os.makedirs(out_dir, exist_ok=True)
    args = [(ffmpeg, corpus_file.get_path(), r.name, r.audio, out_dir) for r in recordings]
    with pool.Pool(num_procs) as p:
        return dict(p.map(_digest_one, args, chunksize=1))


class FfmpegPinCheckJob(Job):
    """Check that ``ffmpeg_binary`` writes the banked audio, then hand it on as ``out_ffmpeg_binary``.

    :param ffmpeg_binary: the ffmpeg to check (``default_tools.get_ffmpeg_binary()``; its hash is the
        fixed pin label, not the path).
    :param corpus_file: the dev-other FLAC bliss corpus, the one the dev-other encode job reads.
    :param reference_list: the per-recording reference list (default :data:`REFERENCE_LIST`, all of
        dev-other; hashed through its sha256 ``hash_overwrite``).  Its names must be exactly the
        corpus's recording names.
    :param accept_label: ``None`` (strict: a mismatch raises) or the ``FFMPEG_PIN_ACCEPT`` label of
        another audio generation (a mismatch is reported, not raised; the label is hashed).  See the
        module docstring.
    :param test_first_n: TESTS ONLY.  ``None`` (production: every recording) or n: check only the
        first n recordings of the corpus by sorted name, which must be the first n of the list
        (hashed when set).

    Outputs: ``out_ffmpeg_binary`` (an executable ``exec <binary> "$@"`` wrapper, written only after
    the check passes), ``out_version`` (the full ``ffmpeg -version`` output), ``out_pcm_sha256`` (every
    checked recording's PCM sha256, in the reference-list format) and ``out_report`` (the counts and
    every mismatching recording).
    """

    __sis_hash_exclude__ = {"accept_label": None, "test_first_n": None}

    def __init__(self, *, ffmpeg_binary: tk.Path, corpus_file: tk.Path,
                 reference_list: tk.Path = REFERENCE_LIST, accept_label: Optional[str] = None,
                 test_first_n: Optional[int] = None):
        assert accept_label is None or (isinstance(accept_label, str) and accept_label.strip()), (
            f"accept_label must be None or a non-empty string, got {accept_label!r}")
        assert test_first_n is None or (isinstance(test_first_n, int) and test_first_n > 0), test_first_n
        self.ffmpeg_binary = ffmpeg_binary
        self.corpus_file = corpus_file
        self.reference_list = reference_list
        self.accept_label = accept_label
        self.test_first_n = test_first_n
        self.rqmt = dict(CHECK_RQMT)
        self.out_version = self.output_path("ffmpeg_version.txt")
        self.out_pcm_sha256 = self.output_path("pcm_sha256.txt")
        self.out_report = self.output_path("pin_check.txt")
        self.out_ffmpeg_binary = self.output_path("ffmpeg")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from i6_core.lib import corpus

        ffmpeg = self.ffmpeg_binary.get_path()
        version = sp.run([ffmpeg, "-version"], stdout=sp.PIPE, stderr=sp.STDOUT, check=False).stdout
        with open(self.out_version.get_path(), "wb") as fh:
            fh.write(version)
        version_line = version.decode(errors="replace").split("\n", 1)[0].strip()

        reference = read_reference_list(self.reference_list.get_path())
        c = corpus.Corpus()
        c.load(self.corpus_file.get_path())
        recordings = sorted(c.all_recordings(), key=lambda r: r.name)
        names = [r.name for r in recordings]
        ref_names = list(reference)
        if self.test_first_n is not None:
            recordings, names, ref_names = (recordings[:self.test_first_n], names[:self.test_first_n],
                                            ref_names[:self.test_first_n])
        if names != ref_names:
            raise RuntimeError(
                f"ffmpeg pin check: the recordings of {self.corpus_file.get_path()} ({len(names)}, first "
                f"{names[:3]}) are not those of the reference list {self.reference_list.get_path()} "
                f"({len(ref_names)}, first {ref_names[:3]})")

        got = pcm_digests(ffmpeg, self.corpus_file, recordings, "encoded", self.rqmt["cpu"])
        write_reference_list(got, self.out_pcm_sha256.get_path())
        bad = [name for name in names if got[name] != reference[name]]
        summary = (f"{len(bad)} of {len(names)} recordings differ from the banked audio"
                   f"{' (first: ' + ', '.join(bad[:5]) + ')' if bad else ''}")
        lines: List[str] = [f"binary {ffmpeg}", f"version {version_line}", f"accept_label {self.accept_label}",
                            f"reference_list {self.reference_list.get_path()}",
                            f"checked {len(names)} match {len(names) - len(bad)} mismatch {len(bad)}"]
        lines += [f"MISMATCH {name} pcm_sha256 {got[name]} reference {reference[name]}" for name in bad]
        if bad and self.accept_label is not None:
            lines.append(f"ACCEPTED under FFMPEG_PIN_ACCEPT={self.accept_label!r}: {summary}.  Every result "
                         f"under this label is a different audio generation, NOT the banked audio.")
        elif not bad:
            lines.append("OK: every checked recording matches the banked audio")
        with open(self.out_report.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines[:5] + lines[-1:]), flush=True)
        if bad and self.accept_label is None:
            raise RuntimeError(
                f"ffmpeg pin check FAILED: {ffmpeg} ({version_line}) does not reproduce the banked audio: "
                f"{summary}; PCM sha256 per recording in {self.out_pcm_sha256.get_path()}.  This ffmpeg "
                f"would change the LibriSpeech audio, and with it every feature, without changing a job "
                f"hash.  Set FFMPEG_BINARY in settings.py to an ffmpeg that passes this check (the "
                f"reference is conda-forge ffmpeg 7.1.1), or, knowingly running on different audio, set "
                f"FFMPEG_PIN_ACCEPT to a label (see the README).")

        wrapper = self.out_ffmpeg_binary.get_path()
        with open(wrapper, "w") as fh:
            fh.write(f"#!/bin/sh\nexec {shlex.quote(ffmpeg)} \"$@\"\n")
        # set explicitly: the file must be executable whatever the umask or the write path
        os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
