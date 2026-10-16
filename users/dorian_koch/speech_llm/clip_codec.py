"""Clip audio codec: the ONE definition of how a clip store row holds its samples (2026-10-16).

A clip store row stores its mono samples either as ``audio`` (list<float32>, every store written before 2026-10-16)
or as ``audio_flac`` (FLAC PCM_16 bytes). FLAC is used only when it is EXACT: ``encode_exact`` returns None unless
every sample is an int16 value / 32768 and the FLAC decodes back bit for bit (then the writer keeps ``audio``).
So both layouts decode to identical float32 arrays and every reader stays exact (user: "new behaviour exactly matches
old behaviour"; guard check_clip_codec.py). Both layouts are read forever.

Dependencies: numpy and soundfile only. This file exists twice, byte-identical (check_clip_codec.py compares them):
``i6_experiments/users/dorian_koch/speech_llm/clip_codec.py`` (clip_store and the standalone worker scripts, which run
under job venvs without i6_experiments) and ``moshi_family/clip_codec.py`` (offline drivers, RL readers).
"""

import io

import numpy as np

COL_AUDIO = "audio"
COL_AUDIO_FLAC = "audio_flac"
COL_SR = "sampling_rate"


def audio_column(column_names) -> str:
    """The audio column a clip store has: ``audio_flac`` if present, else ``audio``."""
    return COL_AUDIO_FLAC if COL_AUDIO_FLAC in column_names else COL_AUDIO


def decode_row(row) -> np.ndarray:
    """A row's samples as float32 -- for ``audio`` exactly ``np.asarray(row["audio"], dtype=np.float32)``."""
    if COL_AUDIO_FLAC in row:
        import soundfile as sf

        x, _ = sf.read(io.BytesIO(bytes(row[COL_AUDIO_FLAC])), dtype="float32", always_2d=False)
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
