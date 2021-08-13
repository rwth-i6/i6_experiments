import os.path
from typing import *

from sisyphus import tk

from recipe.i6_core.audio.ffmpeg import BlissFfmpegJob


def ffmpeg_silence_remove(
        bliss_corpus : tk.Path,
        stop_threshold : int,
        window : float = 0.02,
        stop_duration : float = 0,
        stop_silence : float = 0,
        mode : str = "rms",
        ffmpeg_binary : Optional[tk.Path] = None,
        force_output_format=None,
        alias_path=""):
    """
    Applies the ffmpeg "silenceremove" audio filter.

    See https://ffmpeg.org/ffmpeg-filters.html#silenceremove

    start_periods defaults to -1, so that all silence is removed

    :param bliss_corpus:
    :param stop_threshold: in NEGATIVE dezibel, everything below this will be treated as silence
    :param stop_duration: keeps remaining silence of given value in seconds
    :param
    :return: bliss corpus
    :rtype: tk.Path
    """
    ffmpeg_binary = ffmpeg_binary if ffmpeg_binary else "ffmpeg"
    filter_string = 'silenceremove=detection=%s:stop_periods=-1' % mode

    assert stop_threshold <= 0, "positive values do not make sense"

    filter_string += ':stop_threshold=%ddB' % stop_threshold
    filter_string += ':window=%f' % window
    filter_string += ':stop_duration=%f' % stop_duration
    filter_string += ':stop_silence=%f' % stop_silence
    bliss_ffmpeg_job = BlissFfmpegJob(bliss_corpus,
                                      ffmpeg_options=["-af", filter_string],
                                      recover_duration=True,
                                      output_format=force_output_format,
                                      ffmpeg_binary=ffmpeg_binary)
    bliss_ffmpeg_job.add_alias(os.path.join(alias_path, "ffmpeg_silence_remove"))
    return bliss_ffmpeg_job.out_corpus