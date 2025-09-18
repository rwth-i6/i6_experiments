"""
Data to store globally to make it easier to transfer arbitrarily between the different experiments
"""
from dataclasses import dataclass
from sisyphus import tk
from typing import Any, Dict

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.oggzip import BlissToOggZipJob
from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

# Synthetic data -------------------------------------------------------------------------------------------------------

synthetic_ogg_zip_data = {}
synthetic_bliss_data = {}


def add_synthetic_data(name: str, ogg_zip: tk.Path, bliss: tk.Path):
    global synthetic_ogg_zip_data
    global synthetic_bliss_data
    assert name not in synthetic_ogg_zip_data.keys()
    synthetic_ogg_zip_data[name] = ogg_zip
    synthetic_bliss_data[name] = bliss



# Add data from older experiments
librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls360 = tk.Path("/u/rossenbach/experiments/librispeech_tts/output/input_tts_experiments/gmm_0.5_fixed_phonemes/tts_outputs/librispeech-360_corpus.xml.gz")
librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls360_ogg = BlissToOggZipJob(
    bliss_corpus=librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls360,
    no_conversion=True,
    returnn_root=MINI_RETURNN_ROOT,
    returnn_python_exe=RETURNN_EXE,
).out_ogg_zip

librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls100 = tk.Path("/u/rossenbach/experiments/librispeech_tts/output/input_tts_experiments/gmm_0.5_fixed_phonemes/tts_outputs/librispeech-100_corpus.xml.gz")
librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls100_ogg = BlissToOggZipJob(
    bliss_corpus=librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls100,
    no_conversion=True,
    returnn_root=MINI_RETURNN_ROOT,
    returnn_python_exe=RETURNN_EXE,
).out_ogg_zip

synthetic_bliss_data["librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls100"] = librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls100
synthetic_ogg_zip_data["librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls100"] = librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls100_ogg

synthetic_bliss_data["librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls360"] = librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls360
synthetic_ogg_zip_data["librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls360"] = librispeech_tts_tacotron2_gmm_05_fixed_phonemes_ls360_ogg


# Alignments -----------------------------------------------------------------------------------------------------------

duration_alignments = {}

def add_duration(name: str, duration_hdf: tk.Path):
    global duration_alignments
    assert name not in duration_alignments.keys()
    duration_alignments[name] = duration_hdf


# Vocoder Models -------------------------------------------------------------------------------------------------------

@dataclass
class VocoderPackage:
    checkpoint: tk.Path
    config: Dict[str, Any]

vocoders: Dict[str, VocoderPackage] = {}

def add_vocoder(name: str, vocoder: VocoderPackage):
    global vocoders
    assert name not in vocoders.keys()
    vocoders[name] = vocoder


# CTC Models for RNN-T init --------------------------------------------------------------------------------------------

ctc_models: Dict[str, tk.Path] = {}

def add_ctc_model(name: str, checkpoint: tk.Path):
    global ctc_models
    assert name not in ctc_models.keys()
    ctc_models[name] = checkpoint


# ASR Systems

@dataclass
class ASRRecognizerSystem:
    config: ReturnnConfig
    checkpoint: tk.Path
    preemphasis: float
    peak_normalization: bool

asr_recognizer_systems: Dict[str, ASRRecognizerSystem] = {}

def add_asr_recognizer(name: str, system: ASRRecognizerSystem):
    global asr_recognizer_systems
    assert name not in asr_recognizer_systems.keys()
    asr_recognizer_systems[name] = system
