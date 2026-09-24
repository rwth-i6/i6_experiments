"""New in the port (no source counterpart): LibriSpeech segment names and the RETURNN ogg-zip layout,
for the CPU jobs that read the same audio the L15 forward reads.

The phase-4a audio is the i6 LibriSpeech ogg zip (``i6_experiments.common.datasets.librispeech``,
``BlissToOggZipJob``), whose seq tags are bliss segment names ``<corpus>/<spk-ch-utt>/<spk-ch-utt>``.
Every stored stream of the port (feature / unit / VAD HDFs, the unit store, the id lists, gold)
is keyed by the bare utterance id ``<spk-ch-utt>``, as the banked runs were; the mapping happens
where the audio enters (:func:`utt_id_of_segment`).

:func:`read_ogg_zip_index` and :func:`read_ogg_zip_audio` mirror RETURNN's ``OggZipDataset``
(``returnn/datasets/audio.py``: ``_collect_data_part``, ``_open_audio_file``) and its ``"raw"``
feature path (``returnn/datasets/util/feature_extraction.py``: ``soundfile.read`` of the member
bytes, then ``astype("float32")``; no peak normalisation, no pre-emphasis), so a CPU job decodes
bit-identical waveforms.  Pure python + numpy/soundfile; no sisyphus, no RETURNN import.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Tuple

__all__ = ["utt_id_of_segment", "read_ogg_zip_index", "read_ogg_zip_audio", "iter_ogg_zip_audio"]


def utt_id_of_segment(seq_name: str) -> str:
    """``"dev-other/116-288045-0000/116-288045-0000"`` -> ``"116-288045-0000"`` (asserts the form)."""
    parts = seq_name.split("/")
    if len(parts) != 3 or parts[1] != parts[2] or not parts[0] or not parts[1]:
        raise ValueError(f"not a LibriSpeech segment name <corpus>/<utt>/<utt>: {seq_name!r}")
    return parts[2]


def _zip_name(zip_path: str) -> str:
    # OggZipDataset: ``self._names = [os.path.splitext(os.path.basename(path))[0] ...]``
    return os.path.splitext(os.path.basename(zip_path))[0]


def read_ogg_zip_index(zip_path: str) -> List[Dict[str, Any]]:
    """The entries of ``<name>.txt`` in the zip (``seq_name``, ``file``, ``duration``, ``text``)."""
    import ast
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        data = ast.literal_eval(zf.read(f"{_zip_name(zip_path)}.txt").decode("utf8"))
    assert data and isinstance(data, list) and isinstance(data[0], dict), zip_path
    return data


def read_ogg_zip_audio(zf, zip_path: str, entry: Dict[str, Any]):
    """float32 waveform of one entry, decoded exactly as RETURNN's raw-audio ``OggZipDataset``."""
    import io

    import soundfile

    raw = zf.read(f"{_zip_name(zip_path)}/{entry['file']}")  # zip_audio_files_have_name_as_prefix=True
    audio, sample_rate = soundfile.read(io.BytesIO(raw))
    assert sample_rate == 16000, (entry.get("seq_name"), sample_rate)
    assert audio.ndim == 1, (entry.get("seq_name"), audio.shape)
    return audio.astype("float32")


def iter_ogg_zip_audio(zip_path: str) -> Iterator[Tuple[str, "object"]]:
    """Yield ``(utt_id, float32 waveform)`` for every entry of one ogg zip, in zip order."""
    import zipfile

    index = read_ogg_zip_index(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for entry in index:
            yield utt_id_of_segment(entry["seq_name"]), read_ogg_zip_audio(zf, zip_path, entry)
