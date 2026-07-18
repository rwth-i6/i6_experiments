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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.phonemize import lm_corpus_lexicon_and_g2p
from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.features import (
    MergeW2vu2DataJob,
    MfccKmeansJob,
    W2vu2FeatureDumpJob,
)
from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.eval import (
    GoldPhonesJob,
    W2vu2PerEvalJob,
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


@dataclass(frozen=True)
class Encoder:
    """A frozen feature encoder + the generator geometry its frame rate / dim implies.

    generator_stride/kernel are the paper's values per feature rate (BEST-RQ 25 Hz -> stride 2,
    kernel 5 time-matched; wav2vec2 50 Hz -> the paper's stride 3, kernel 9). mfcc_downsample and
    vad_subframes both follow from fps (= 100 Hz / fps).
    """

    name: str
    encoder_type: str          # "bestrq" | "wav2vec2"
    layer: int
    fps: int
    input_dim: int
    generator_stride: int
    generator_kernel: int
    hf_model_id: str = ""      # wav2vec2 only

    @property
    def mfcc_downsample(self) -> int:
        return 100 // self.fps

    @property
    def vad_subframes(self) -> int:
        return 100 // self.fps


# BEST-RQ layer 5 @25 Hz -- the SAE §1c arm (name kept so its aliases/hashes are unchanged).
BESTRQ_L5 = Encoder("bestrq_l5", "bestrq", layer=5, fps=25, input_dim=512,
                    generator_stride=2, generator_kernel=5)
# wav2vec2-Large LV-60 (SSL, no finetune), HF port of the paper's model; hidden_states[15] = paper L15.
W2V2_LV60_L15 = Encoder("w2v2_lv60_l15", "wav2vec2", layer=15, fps=50, input_dim=1024,
                        generator_stride=3, generator_kernel=9,
                        hf_model_id="facebook/wav2vec2-large-lv60")


def _prefix(encoder: Encoder) -> str:
    """BEST-RQ keeps the bare `sae/1c` prefix (unchanged aliases); other encoders nest under it."""
    return PREFIX if encoder.encoder_type == "bestrq" else f"{PREFIX}/{encoder.name}"


def seed_grid(n: int = 5) -> List[Dict[str, Any]]:
    """Sweep only the seed -- the paper's protocol for a fixed config (README: common.seed=range(0,5))."""
    return [{"seed": s} for s in range(n)]

# The §1.0 selection LM (§1a, KenLM 4-gram, ppl 8.45, Spearman 0.89 vs gold PER). It is trained on
# SIL-*free* 𝒯_φ, which is the right LM here: fairseq strips <SIL> from hypotheses before scoring.
# (fairseq's own prepare_text.sh trains this LM on SIL-*augmented* text and strips SIL anyway -- a
# mismatch its own paper contradicts, so we deliberately keep the matched, already-validated LM.)
PHONEME_LM_BIN = tk.Path(
    "/e/project1/spell/wu24/2026-07-13_unsupervised/output/sae/1a/phoneme_lm_o4.bin",
    hash_overwrite="sae_1a_phoneme_lm_o4_bin",
)


def _audio_data(*, encoder: Encoder, limit: Optional[int]) -> tk.Path:
    from speech_llm.prefix_lm.sis_recipe.exp2025_11_06_speech_llms.librispeech.data.huggingface import (
        get_librispeech_train_clean_100_hf_ogg,
    )

    hf = get_librispeech_train_clean_100_hf_ogg()
    prefix = _prefix(encoder)
    mfcc_ds, vad_sf = encoder.mfcc_downsample, encoder.vad_subframes

    # wav2vec2 needs its HF checkpoint on disk for the offline GPU dump; the download job's name
    # matches settings._HF_ONLINE_JOB_NAMES -> it runs on the online login engine, and out_content_dir
    # is a from_pretrained-ready dir.
    hf_model_dir: Optional[tk.Path] = None
    if encoder.encoder_type == "wav2vec2":
        from i6_experiments.users.schmitt.external_models.huggingface import DownloadHuggingFaceRepoJobV2

        dl = DownloadHuggingFaceRepoJobV2(model_id=encoder.hf_model_id)
        dl.add_alias(f"{prefix}/dl_hf_model")
        hf_model_dir = dl.out_content_dir

    mfcc = MfccKmeansJob(hf_data_dir=hf, split="train", num_clusters=64,
                         mfcc_downsample=mfcc_ds, vad_subframes=vad_sf, limit=limit)
    mfcc.add_alias(f"{prefix}/mfcc_kmeans_k64")
    tk.register_output(f"{prefix}/mfcc_kmeans.stats.txt", mfcc.out_stats)

    dumps: Dict[str, tk.Path] = {}
    for fs_name, hf_split, t in (("train", "train", 8), ("valid", "dev", 4)):
        d = W2vu2FeatureDumpJob(
            hf_data_dir=hf, split=hf_split, name=fs_name, mfcc_centroids=mfcc.out_centroids,
            encoder_type=encoder.encoder_type, encoder_layer=encoder.layer, hf_model_dir=hf_model_dir,
            mfcc_downsample=mfcc_ds, vad_subframes=vad_sf, trim_silence=True, limit=limit, time_rqmt=t,
        )
        d.add_alias(f"{prefix}/feats_l{encoder.layer}_{fs_name}")
        tk.register_output(f"{prefix}/feats_l{encoder.layer}_{fs_name}.stats.txt", d.out_stats)
        dumps[fs_name] = d.out_dir

    merged = MergeW2vu2DataJob(splits=dumps)
    merged.add_alias(f"{prefix}/data_l{encoder.layer}")
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
    encoder: Encoder = BESTRQ_L5,
    sil_probs: Tuple[float, ...] = (0.5,),
    grid: Optional[List[Dict[str, Any]]] = None,
    max_update: Optional[int] = None,
) -> None:
    """Wire the §1c graph for one `encoder`. `smoke=True` = a tiny end-to-end shakedown.

    `sil_probs` is swept outside `grid` because each value needs its own binarized text corpus. The
    text side is encoder-independent, so its jobs are shared across encoders (same `PREFIX`).
    """
    limit = 200 if smoke else None
    max_lines = 200_000 if smoke else None
    if max_update is None:
        max_update = 200 if smoke else 150_000
    prefix = _prefix(encoder)

    data = _audio_data(encoder=encoder, limit=limit)

    if grid is None:
        grid = [{}] if smoke else pilot_grid()

    # Gold dev phones for greedy PER -- checkpoint-independent, computed once, shared by every eval.
    # Skipped in smoke: the dump is dev-clean-only (limit=200) so the split assignment is degenerate.
    gold = None if smoke else GoldPhonesJob().out_gold

    for sil_prob in sil_probs:
        text = _text_data(sil_prob=sil_prob, max_lines=max_lines, threshold=1000)
        for cfg in grid:
            ov = w2vu2_overrides(
                max_update=max_update, input_dim=encoder.input_dim,
                generator_stride=encoder.generator_stride, generator_kernel=encoder.generator_kernel,
                **cfg,
            )
            job = FairseqW2vu2TrainJob(
                data_dir=data, text_data=text, kenlm_path=PHONEME_LM_BIN, overrides=ov,
                time_rqmt=1 if smoke else 11.5,
            )
            tag = "smoke" if smoke else _tag(cfg)
            arm = f"{prefix}/gan_l{encoder.layer}_sil{sil_prob}/{tag}"
            job.add_alias(arm)
            tk.register_output(f"{arm}/train.log", job.out_log)

            if gold is not None:
                per = W2vu2PerEvalJob(
                    checkpoint=job.out_best, data_dir=data, text_data=text,
                    feats_dir=data, gold=gold,
                )
                per.add_alias(f"{arm}/per")
                tk.register_output(f"{arm}/per.json", per.out_per)


def _tag(cfg: Dict[str, Any]) -> str:
    short = {"gradient_penalty": "gp", "smoothness_weight": "sm", "code_penalty": "pd",
             "mmi_weight": "ss", "generator_batch_norm": "bn", "generator_stride": "st", "seed": "s"}
    return "_".join(f"{short.get(k, k)}{v}" for k, v in sorted(cfg.items()))


def pilot_grid() -> List[Dict[str, Any]]:
    """6 runs (x2 p_sil arms outside) -- what `default_grid` would cost is not available to us.

    SAE_PLAN §1c asks for the 2.0 sweep x 4-5 seeds *and* batch-norm {20,30,40} *and* stride {1,2};
    multiplied out that is 384 runs (~4400 GPU-h) and the project is rate-limited, so this pilot
    spends the budget only where the evidence says the outcome is actually uncertain:

      generator_batch_norm {20,30,40}  the one knob the paper calls "critical for convergence", and
                                       it tracks the feature distribution (30 for 1024-d wav2vec2-Large,
                                       35 for XLSR-53) -- so it cannot be inherited for 512-d BEST-RQ.
      p_sil {0.10, 0.5}                swept in build_sae_1c_gan: 0.10 matches our measured residual
                                       silence (0.044-0.054), 0.5 is the plan's inherited value
                                       (0.136, ~2.6x too much). See SAE_1c.md.

    Held fixed, with reasons:
      generator_stride = 2   25 Hz / 2 = 12.5 Hz vs our measured ~11.9 Hz phone rate. The plan's other
                             arm, stride 1, *is* 25 Hz -- the rate Table 1 reports diverging at >100
                             PER. Buying that arm costs half the pilot to confirm a published negative.
      loss weights           at the w2vu2.yaml defaults (lambda 1.0, gamma 1.5, eta 3.0, delta 0.5),
                             each of which is already one of the two values the plan's grid sweeps.
      seed = 0               convergence *rate* over seeds is only meaningful once a configuration
                             converges at all; seeds are stage 2.
    """
    return [{"generator_batch_norm": bn} for bn in (20, 30, 40)]


def default_grid() -> List[Dict[str, Any]]:
    """The 2.0 sweep (paper p.6): lambda/gamma/eta/delta in {1.0,1.5}/{1.5,2.5}/{0,3}/{0.3,0.5} x 4 seeds.

    64 runs x ~11.5 h. Kept for reference; not reachable from any config -- see `pilot_grid`.
    """
    out = []
    for gp in (1.0, 1.5):
        for sm in (1.5, 2.5):
            for pd in (0.0, 3.0):
                for ss in (0.3, 0.5):
                    for seed in range(4):
                        out.append({"gradient_penalty": gp, "smoothness_weight": sm,
                                    "code_penalty": pd, "mmi_weight": ss, "seed": seed})
    return out
