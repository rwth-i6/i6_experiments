__all__ = []

from sisyphus import tk

from .external.haotian_lm.experiments.lm_phon.count_ngram import build_phon_count_ngram_lm
from .external.haotian_lm.data.phon_lm import SilenceModel

VAD_silence_model = SilenceModel(
    between_word=0.05,
    sentence_bounds=0.05
)

def build_phon_ngram_lm(
    *,
    prefix: str = "phon_lm",
    librispeech_key: str = "train-other-960",
    order: int = 2,
    kenlm_max_order: int = 10,
    pruning: list[int] | None = None,
    interpolate_unigrams: bool = True,
    discount_fallback: list[float] | None = None,
    mem: float = 96.0,
    time: float = 48.0,
    use_eow_phonemes: bool = False,
    silence_model: SilenceModel = SilenceModel(
        between_word=0.75,
        sentence_bounds=1.0,
    ),
    g2p_model: tk.Path | None = None,
):
    if discount_fallback is None:
        discount_fallback = [0.5, 1.0, 1.5]

    ngram_res = build_phon_count_ngram_lm(
        prefix="phon_lm",
        librispeech_key="train-other-960",
        silence_model=silence_model,
        g2p_model=g2p_model,
    )

    tk.register_output(
        "phon_lm/train_data.txt",
        ngram_res["text"],
    )

    return ngram_res["arpa"]

def py():
    build_phon_ngram_lm(librispeech_key="train-clean-100")
