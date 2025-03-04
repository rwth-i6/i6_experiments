import os
import numpy as np
import subprocess
import multiprocessing as mp
from i6_experiments.users.rossenbach.experiments.jaist_project.pytorch_networks.tts_shared.corpus import Corpus, Recording, Segment

# global environment where thread count for ffmpeg is set to 2
ENVIRON = os.environ.copy()
ENVIRON["OMP_NUM_THREADS"] = "2"

def save_wav(wav, path, sr, peak_normalization=True):
    from scipy.io import wavfile
    if peak_normalization:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    else:
        wav *= 32767
    wavfile.write(path, sr, wav.astype(np.int16))


def save_ogg(args):
    """
    :param args: wav, path and sr
    """
    wav, path, sr = args
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    p1 = subprocess.Popen(["ffmpeg", "-y", "-f", "s16le", "-ar", "%i" % sr, "-i", "pipe:0", "-c:a", "libvorbis", "-q", "3.0", path],
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          env=ENVIRON)
    p1.communicate(input=wav.astype(np.int16).tobytes())
    p1.terminate()