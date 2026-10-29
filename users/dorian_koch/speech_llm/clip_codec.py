"""Clip audio codec: the ONE definition of how a clip store row holds its samples (2026-10-16).

A clip store row stores its mono samples either as ``audio`` (list<float32>, every store written before 2026-10-16)
or as ``audio_flac`` (FLAC PCM_16 bytes). FLAC is used only when it is EXACT: ``encode_exact`` returns None unless
every sample is an int16 value / 32768 and the FLAC decodes back bit for bit (then the writer keeps ``audio``). A
store may carry both columns, decided per row: a row's ``audio_flac`` is null when that row kept ``audio`` (and vice
versa), since a streaming writer cannot know in advance whether every clip will be exact.
So both layouts decode to identical float32 arrays and every reader stays exact (new behaviour must exactly match
old behaviour; guard check_clip_codec.py). Both layouts are read forever.

Dependencies: numpy and soundfile only. This file exists twice, byte-identical (check_clip_codec.py compares them):
``i6_experiments/users/dorian_koch/speech_llm/clip_codec.py`` (clip_store and the standalone worker scripts, which run
under job venvs without i6_experiments) and ``moshi_family/clip_codec.py`` (offline drivers, RL readers).
"""

import io

import numpy as np

COL_AUDIO = "audio"
COL_AUDIO_FLAC = "audio_flac"
COL_SR = "sampling_rate"


def audio_columns(column_names) -> list:
    """The audio columns a clip store has (``audio`` and/or ``audio_flac``), for ``select_columns``."""
    return [c for c in (COL_AUDIO, COL_AUDIO_FLAC) if c in column_names]


def numpy_rows(ds):
    """``ds`` in numpy format for sequential reads, with ``audio_flac`` left as Python ``bytes``.

    ⚠ ``datasets``' numpy format hands binary values over as ``np.bytes_``, which DROPS TRAILING NUL BYTES -- and a
    FLAC stream ends in a CRC-16, so ~1 in 256 clips ends in a NUL byte (found 2026-10-16: "flac decoder lost sync").
    Every numpy-format reader of a clip store goes through here.
    """
    cols = [c for c in ds.column_names if c != COL_AUDIO_FLAC]
    return ds.with_format("numpy", columns=cols, output_all_columns=True)


def decode_row(row) -> np.ndarray:
    """A row's samples as float32 -- for ``audio`` exactly ``np.asarray(row["audio"], dtype=np.float32)``."""
    flac = row.get(COL_AUDIO_FLAC)
    if isinstance(flac, np.bytes_):
        raise TypeError("audio_flac read in numpy format has lost its trailing NUL bytes: read via numpy_rows(ds)")
    if flac is not None:
        import soundfile as sf

        x, _ = sf.read(io.BytesIO(bytes(flac)), dtype="float32", always_2d=False)
        return x
    return np.asarray(row[COL_AUDIO], dtype=np.float32)


def encode_exact(samples, sample_rate: int) -> bytes | None:
    """FLAC PCM_16 bytes of ``samples`` if that is lossless (int16-valued, decodes back bit-exact), else None."""
    import soundfile as sf

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return None
    s = x.astype(np.float64) * 32768.0
    if np.any(s != np.round(s)) or s.min() < -32768 or s.max() > 32767:
        return None
    buf = io.BytesIO()
    sf.write(buf, x, int(sample_rate), format="FLAC", subtype="PCM_16")
    raw = buf.getvalue()
    back, _ = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return raw if back.shape == x.shape and np.array_equal(back, x) else None


def wav_to_flac(raw: bytes) -> bytes | None:
    """FLAC PCM_16 bytes holding exactly the samples of the PCM_16 WAV ``raw`` (any channel count), else None.

    For ``datasets`` ``Audio()`` cells (MoshiAnnotate, Chatterbox TTS): their WAV is PCM_16, so FLAC stores the same
    int16 frames and every decoder (soundfile, torchcodec) returns the identical array -- measured 2026-10-16 on 200
    real clips of four corpora through ``datasets``' own decode. Anything else (float WAV, not a WAV) stays as is.
    """
    import soundfile as sf

    try:
        info = sf.info(io.BytesIO(raw))
    except RuntimeError:  # soundfile.LibsndfileError: not audio libsndfile reads
        return None
    if info.format != "WAV" or info.subtype != "PCM_16" or info.frames == 0:
        return None
    x, sr = sf.read(io.BytesIO(raw), dtype="int16", always_2d=True)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
    out = buf.getvalue()
    back, sr2 = sf.read(io.BytesIO(out), dtype="int16", always_2d=True)
    return out if sr2 == sr and back.shape == x.shape and np.array_equal(back, x) else None
