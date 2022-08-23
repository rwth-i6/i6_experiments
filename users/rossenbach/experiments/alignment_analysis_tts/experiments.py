from .tacotron2_aligner.experiments import run_tacotron2_aligner_training
from .ls_gmm_aligner_from_hilmes.baseline_config import run_librispeech_100_common_tts_baseline
from .gl_vocoder.default_vocoder import get_default_vocoder


def run_aat_experiments():
    run_librispeech_100_common_tts_baseline()
    run_tacotron2_aligner_training()
    get_default_vocoder("experiments/aat/ls100_aat_vocoder")