"""
Lexicon-free BPE CTC recognition with a **count** BPE n-gram (KenLM) LM, reusing the BPE CTC AM.

Mirror of the phon CTC lexfree count path (``ctc_phon.lexfree_baseline`` +
``phon_lexfree.lexfree_count_search``), with the BPE-specific differences, all passed as parameters to
the topology-agnostic ``rasr`` / ``phon_am`` helpers (so the phon path is untouched):

* the lexicon-free lexicon is built from the CTC BPE lexicon (``[BLANK]`` at index 0, ``special="blank"``),
  one lemma per BPE subword + ``<s>``/``</s>`` (:class:`...rasr.BuildPhonLexiconfreeLexiconJob`);
* the LM is a count BPE n-gram (:func:`...lm_bpe.count_ngram.build_bpe_count_ngram_lm`), trained on the
  same BPE-tokenized text as the BPE neural LMs and queried by the KenLM Python label scorer on the BPE
  subword tokens;
* the search collapses repeated labels around the blank (``collapse_repeated_labels=True``); and
* the metric is **WER** (the BPE lexfree decoder ``rasr_phmm_lexfree_ngram_bpe_v1`` de-BPEs the traceback
  to a word hypothesis), NOT PER -- BPE subwords are losslessly invertible to words, so WER is the native
  metric (unlike the phoneme lexfree path, whose phoneme-stream hypothesis is only scorable as PER). No
  phoneme/silence duration analysis (meaningless for BPE units).
"""
from typing import Iterable, List, Optional

from .baseline import bpe_ctc_ls960_base
from ...phon_am import lm_scale_sweep_dev_other, compute_phon_prior
from ...default_tools import RETURNN_EXE, RETURNN_ROOT, LIBRASR_WHEEL, kenlm_repo
from ...rasr import (
    BuildPhonLexiconfreeLexiconJob,
    CreateLibrasrVenvWithKenLMJob,
    build_lexiconfree_count_recognition_config,
)
from ..lm_bpe.count_ngram import build_bpe_count_ngram_lm
from ...pytorch_networks.phmm.decoder.rasr_phmm_lexfree_ngram_bpe_v1 import DecoderConfig as NgramDecoderConfig

# The CTC blank lemma's phoneme; index 0 of the lexfree lexicon (must exist in the lexicon).
_BLANK_LABEL = "[BLANK]"


def _scale_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def lexfree_count_bpe_ctc_ls960(
    *,
    bpe_size: int = 512,
    orders: Iterable[int] = (4,),
    kenlm_max_order: int = 10,
    pruning: Optional[List[int]] = None,
    lm_scales: Iterable[float] = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0),
    prior_scale: float = 0.3,
    sweep_epoch: int = 125,
    beam_size: int = 32,
    score_threshold: float = 14.0,
    dataset_keys: Iterable[str] = ("dev-other",),
):
    """
    Lexicon-free recognition with a count BPE n-gram LM, scored by WER.

    Focused recognition matching the lexical path: ALWAYS apply the acoustic prior
    (``prior_scale=0.3`` -- the count path decodes the SAME peaky CTC posteriors, so it needs the same
    ``/p(label)`` correction) and sweep the LM scale on dev-other at the final checkpoint. The count LM is
    reused (by content hash) from the BPE neural-LM text pipeline.

    :param bpe_size: MUST match the AM lexicon's bpe_size (shared subword codes -> matching LM tokens).
    :param orders: KenLM n-gram orders to build/sweep (4-gram default; BPE subwords carry near-word context).
    """
    am_artifacts = bpe_ctc_ls960_base(bpe_size=bpe_size)
    ctc_bpe_lexicon = am_artifacts["ctc_bpe_lexicon"]
    top_prefix = am_artifacts["prefix_name"]

    # BPE-only Bliss lexicon for lexicon-free search ([BLANK] at idx 0 marked special="blank", + <s>/</s>);
    # reuse the phon lexfree builder with the blank as the index-0 "silence". Build only; not registered.
    lexfree_lexicon = BuildPhonLexiconfreeLexiconJob(
        bliss_lexicon=ctc_bpe_lexicon, silence_phoneme=_BLANK_LABEL
    ).out_lexicon

    # Dedicated recognition venv with the `kenlm` python module built with MAX_ORDER set (so an order>6
    # model can be LOADED). Shared by content hash with the phon count path.
    kenlm_returnn_exe = CreateLibrasrVenvWithKenLMJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "dm-tree", "h5py"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
        kenlm_repository=kenlm_repo,
        kenlm_max_order=kenlm_max_order,
    ).out_python_bin
    kenlm_returnn = {"returnn_exe": kenlm_returnn_exe, "returnn_root": RETURNN_ROOT}

    # Acoustic prior for the sweep checkpoint (shared by content hash with the lexical path's prior).
    prior_file = compute_phon_prior(am_artifacts, epoch=sweep_epoch)
    prior_tag = _scale_tag(prior_scale)

    for order in orders:
        count_lm = build_bpe_count_ngram_lm(
            prefix=top_prefix + f"/lexicon-free-search/_lm/count_o{order}",
            librispeech_key="train-other-960",
            bpe_size=bpe_size,
            order=order,
            kenlm_max_order=kenlm_max_order,
            pruning=pruning,
        )

        def _make_count_recog(lm_scale, _count_lm=count_lm):
            rasr_cfg = build_lexiconfree_count_recognition_config(
                lexicon_path=lexfree_lexicon,
                lm_scale=lm_scale,
                am_scale=1.0,
                max_beam_size=beam_size,
                score_threshold=score_threshold,
                collapse_repeated_labels=True,  # CTC AM: collapse consecutive duplicate subwords
            )
            decoder_cfg = NgramDecoderConfig(
                rasr_config_file=rasr_cfg,
                lexicon=lexfree_lexicon,
                kenlm_file=_count_lm["binary"],
                prior_file=prior_file,
                prior_scale=prior_scale,
                silence_label=_BLANK_LABEL,
            )
            return {
                "decoder_module": "phmm.decoder.rasr_phmm_lexfree_ngram_bpe_v1",
                "decoder_config": decoder_cfg,
                "search_kwargs": {
                    "forward_config": {
                        "num_workers_per_gpu": 0,
                        "torch_dataloader_opts": {"num_workers": 0},
                    }
                },
            }

        lm_scale_sweep_dev_other(
            artifacts=am_artifacts,
            make_recog=_make_count_recog,
            lm_scales=lm_scales,
            epoch=sweep_epoch,
            search_type="lexicon-free",
            variant_prefix=f"count_o{order} prior{prior_tag}",
            variant_suffix=f"_beam{beam_size}_st{_scale_tag(score_threshold)}",
            returnn=kenlm_returnn,
            per_lexicon=None,  # WER is the native metric for BPE (de-BPE'd hypothesis)
            hyp_is_phonemes=False,
            report_wer=True,
            dataset_keys=dataset_keys,
        )

    return {"lexfree_lexicon": lexfree_lexicon}
