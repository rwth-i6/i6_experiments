"""SAE §1c — the wav2vec-U 2.0 GAN pipeline (fairseq trainer, BEST-RQ features).

Audio is **train-clean-100**, not LS960: the 100 h Ogg encode already exists, whereas the LS960 one
has never been built and its job name matches settings.py's `_HF_ONLINE_JOB_NAMES`, forcing it onto
the 4-slot login engine at cpu=min(16,4) -- ~a day of wall clock that would starve the other live
managers. GAN cost is step-bounded (150 k updates at batch 160) and therefore identical at 100 h and
960 h, so only the one-time dump is saved. The one thing 100 h costs us is that train-clean-100 is
*clean-only* while the gate is dev-other; that is measured rather than assumed (see the feature-stat
check in SAE_1c.md) because the generator's BatchNorm running stats are fit on this audio.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.phonemize import lm_corpus_lexicon_and_g2p
from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.features import (
    MergeW2vu2DataJob,
    MfccKmeansJob,
    W2vu2FeatureDumpJob,
)
from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.gan import (
    FairseqW2vu2TrainJob,
    w2vu2_overrides,
)
from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.text import (
    FairseqPreprocessTextJob,
    PhonemizeWithSilJob,
)

PREFIX = "sae/1c"

# The §1.0 selection LM (§1a, KenLM 4-gram, ppl 8.45, Spearman 0.89 vs gold PER). It is trained on
# SIL-*free* 𝒯_φ, which is the right LM here: fairseq strips <SIL> from hypotheses before scoring.
# (fairseq's own prepare_text.sh trains this LM on SIL-*augmented* text and strips SIL anyway -- a
# mismatch its own paper contradicts, so we deliberately keep the matched, already-validated LM.)
PHONEME_LM_BIN = tk.Path(
    "/e/project1/spell/wu24/2026-07-13_unsupervised/output/sae/1a/phoneme_lm_o4.bin",
    hash_overwrite="sae_1a_phoneme_lm_o4_bin",
)


def _audio_data(*, limit: Optional[int], encoder_layer: int) -> tk.Path:
    from speech_llm.prefix_lm.sis_recipe.exp2025_11_06_speech_llms.librispeech.data.huggingface import (
        get_librispeech_train_clean_100_hf_ogg,
    )

    hf = get_librispeech_train_clean_100_hf_ogg()

    mfcc = MfccKmeansJob(hf_data_dir=hf, split="train", num_clusters=64, limit=limit)
    mfcc.add_alias(f"{PREFIX}/mfcc_kmeans_k64")
    tk.register_output(f"{PREFIX}/mfcc_kmeans.stats.txt", mfcc.out_stats)

    dumps: Dict[str, tk.Path] = {}
    for fs_name, hf_split, t in (("train", "train", 8), ("valid", "dev", 4)):
        d = W2vu2FeatureDumpJob(
            hf_data_dir=hf, split=hf_split, name=fs_name, mfcc_centroids=mfcc.out_centroids,
            encoder_layer=encoder_layer, trim_silence=True, limit=limit, time_rqmt=t,
        )
        d.add_alias(f"{PREFIX}/feats_l{encoder_layer}_{fs_name}")
        tk.register_output(f"{PREFIX}/feats_l{encoder_layer}_{fs_name}.stats.txt", d.out_stats)
        dumps[fs_name] = d.out_dir

    merged = MergeW2vu2DataJob(splits=dumps)
    merged.add_alias(f"{PREFIX}/data_l{encoder_layer}")
    return merged.out_dir


def _text_data(*, sil_prob: float, max_lines: Optional[int], threshold: int) -> tk.Path:
    text, lex, g2p = lm_corpus_lexicon_and_g2p()
    sil = PhonemizeWithSilJob(
        text_file=text, bliss_lexicon=lex, g2p_lexicon=g2p,
        sil_prob=sil_prob, surround=True, seed=0, max_lines=max_lines,
    )
    sil.add_alias(f"{PREFIX}/text_sil{sil_prob}")
    tk.register_output(f"{PREFIX}/text_sil{sil_prob}.stats.txt", sil.out_stats)

    prep = FairseqPreprocessTextJob(text_file=sil.out_text, threshold=threshold)
    prep.add_alias(f"{PREFIX}/text_data_sil{sil_prob}")
    tk.register_output(f"{PREFIX}/text_data_sil{sil_prob}/dict.txt", prep.out_dict)
    return prep.out_dir


def build_sae_1c_gan(
    *,
    smoke: bool = False,
    encoder_layer: int = 5,
    sil_probs: Tuple[float, ...] = (0.5,),
    grid: Optional[List[Dict[str, Any]]] = None,
    max_update: Optional[int] = None,
) -> None:
    """Wire the §1c graph. `smoke=True` = a tiny end-to-end shakedown (few utts, 200 updates).

    `sil_probs` is swept outside `grid` because each value needs its own binarized text corpus.
    """
    limit = 200 if smoke else None
    max_lines = 200_000 if smoke else None
    if max_update is None:
        max_update = 200 if smoke else 150_000

    data = _audio_data(limit=limit, encoder_layer=encoder_layer)

    if grid is None:
        grid = [{}] if smoke else pilot_grid()

    for sil_prob in sil_probs:
        text = _text_data(sil_prob=sil_prob, max_lines=max_lines, threshold=1000)
        for cfg in grid:
            ov = w2vu2_overrides(max_update=max_update, **cfg)
            job = FairseqW2vu2TrainJob(
                data_dir=data, text_data=text, kenlm_path=PHONEME_LM_BIN, overrides=ov,
                time_rqmt=1 if smoke else 11.5,
            )
            tag = "smoke" if smoke else _tag(cfg)
            job.add_alias(f"{PREFIX}/gan_l{encoder_layer}_sil{sil_prob}/{tag}")
            tk.register_output(f"{PREFIX}/gan_l{encoder_layer}_sil{sil_prob}/{tag}/train.log", job.out_log)


def _tag(cfg: Dict[str, Any]) -> str:
    short = {"gradient_penalty": "gp", "smoothness_weight": "sm", "code_penalty": "pd",
             "mmi_weight": "ss", "generator_batch_norm": "bn", "generator_stride": "st", "seed": "s"}
    return "_".join(f"{short.get(k, k)}{v}" for k, v in sorted(cfg.items()))


def default_grid() -> List[Dict[str, Any]]:
    """The 2.0 sweep (paper p.6): lambda/gamma/eta/delta in {1.0,1.5}/{1.5,2.5}/{0,3}/{0.3,0.5} x 4 seeds."""
    out = []
    for gp in (1.0, 1.5):
        for sm in (1.5, 2.5):
            for pd in (0.0, 3.0):
                for ss in (0.3, 0.5):
                    for seed in range(4):
                        out.append({"gradient_penalty": gp, "smoothness_weight": sm,
                                    "code_penalty": pd, "mmi_weight": ss, "seed": seed})
    return out
