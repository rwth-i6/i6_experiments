from dataclasses import dataclass
from sisyphus import tk

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments
from i6_experiments.users.hilmes.data.datastream import AudioFeatureDatastream

from ..data import get_tts_log_mel_datastream, get_ls100_silence_preprocess_ogg_zip


@dataclass(frozen=True)
class VocoderDataclass:
    """
    Dataclass for TTS Datasets
    """

    zip: tk.Path
    audio_opts: AudioFeatureDatastream
    train_segments: tk.Path
    dev_segments: tk.Path


def get_vocoder_data():
    train_segments, cv_segments = get_librispeech_tts_segments()
    zip_dataset = get_ls100_silence_preprocess_ogg_zip()
    log_mel_datastream = get_tts_log_mel_datastream()

    vocoder_data = VocoderDataclass(
        zip=zip_dataset,
        audio_opts=log_mel_datastream,
        train_segments=train_segments,
        dev_segments=cv_segments,
    )
    return vocoder_data