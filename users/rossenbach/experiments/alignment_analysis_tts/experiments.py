from .tacotron2_aligner.experiments import run_tacotron2_aligner_training
from .ls_gmm_aligner_from_hilmes.baseline_config import run_librispeech_100_common_tts_baseline
from .gl_vocoder.default_vocoder import get_default_vocoder
from .ctc_aligner.experiments import get_baseline_ctc_alignment, get_loss_scale_ctc_alignment, get_baseline_ctc_alignment_v2, get_ls460_ctc_alignment_v2, get_ls960_ctc_alignment_v2
from .default_tts.experiments import get_ctc_based_tts, get_optimized_tts_models, get_ls460_models, get_ls960_models
from .default_tts.random_audio import create_random_corpus



def run_aat_experiments():
    run_librispeech_100_common_tts_baseline()
    run_tacotron2_aligner_training()
    get_default_vocoder("experiments/aat/ls100_aat_vocoder")
    get_baseline_ctc_alignment()
    get_baseline_ctc_alignment_v2()
    get_ls460_ctc_alignment_v2()
    get_ls960_ctc_alignment_v2()
    get_loss_scale_ctc_alignment()
    get_ctc_based_tts()
    get_optimized_tts_models()
    create_random_corpus()
    get_ls460_models()
    get_ls960_models()