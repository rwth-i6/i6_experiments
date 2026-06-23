"""
EOW-phoneme CTC baseline (lexicon-constrained recognition).

This is the posterior-HMM baseline (:func:`...phmm_phon.baseline.eow_phon_phmm_ls960_base`) with the
**FSA topology switched from HMM to CTC** -- nothing else changes. Training reuses the exact same
shared path (:mod:`...phon_am`): the ``phmm.phmm_zhou`` network, the ``PhmmTrainStep`` (``i6_native_ops``
``fbw2`` full-sum loss over a RASR FSA built from the orthography), the raw-orth data pipeline, and the
same optimizer / OCLR-style /4 schedule / 4-GPU parameter-averaging / SpecAugment / aux losses / label
smoothing. The CTC loss is realized as ``fbw2`` over a ``topology="ctc"`` FSA (``fbw2`` v2 matches
``torch.nn.functional.ctc_loss``), so this isolates the effect of the topology against the pHMM baseline.

The only name-implied differences vs. the pHMM baseline:

* the lexicon has ``[BLANK]`` at index 0 (``special="blank"``) instead of ``[SILENCE]``
  (:func:`get_ctc_eow_lexicon`),
* the FSA exporter uses ``topology="ctc"``,
* recognition uses :func:`build_librasr_ctc_recognition_config` (CTC label collapsing around the blank)
  with the context-independent recog lexicon, and ``silence_label="[BLANK]"`` in the decoder,
* an auxiliary PER is reported (the CTC label stream is phoneme-native).
"""
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict

from ...data.phon import build_eow_phon_ctc_fsa_training_datasets, get_ctc_eow_lexicon
from ...lm import get_4gram_lm_rasr_config
from ...phon_am import (
    build_phon_train_settings,
    build_dev_test_dataset_tuples,
    make_librasr_returnn,
    train_and_lexicon_search,
    lm_scale_sweep_dev_other,
    compute_phon_prior,
    greedy_phon_per_dev_other,
)
from ...pytorch_networks.phmm.decoder.rasr_phmm_v1 import DecoderConfig as RasrDecoderConfig
from ...rasr import (
    AddSentenceBoundaryLemmataToPhmmLexiconJob,
    MakeLexiconContextIndependentJob,
    build_fsa_exporter_config,
    build_librasr_ctc_recognition_config,
)

# Focused recognition: ALWAYS apply the acoustic prior (prior_scale=0.3 -- the diagnosed fix for the
# CTC<<pHMM lexical gap; raw posteriors bias the peaky/blank-dominated CTC AM toward frequent words) and
# tune ONLY the 4-gram word-LM scale, on dev-other at one converged checkpoint. No no-prior recog is run.
_RECOG_EPOCH = 125  # final checkpoint (matches the pHMM sweep epoch)
# JOINT prior x lm sweep. The extended lm-only grid REFUTED the "truncated grid" idea: CTC word-WER
# bottoms at lm1.0 (8.75) then EXPLODES (9.88@1.3, 15.13@1.6, 33.62@2.0), while pHMM bottoms at lm1.0
# (6.74) and degrades gently. The residual CTC<pHMM gap is +1.76% SUBSTITUTIONS on short function words
# (a/the, in/and, is/was) -- the SAME error types as pHMM, ~50% more of them; del/ins are equal, so the
# search is not breaking segmentation. CTC can't buy LM help by raising lm_scale because its PEAKY
# posterior (blank marginal ~0.61 vs pHMM silence ~0.40) destabilizes search above lm1.0. A LARGER
# acoustic prior FLATTENS that posterior (penalizes the dominant blank, boosts rare phonemes) -> more
# HMM-like -> should let CTC absorb a stronger word LM and cut the function-word subs. prior and lm are
# COUPLED, so sweep them JOINTLY on dev-other @ep125. The prior=0.3 row reuses cached recogs; only the
# 0.5/0.7 rows are new.
_PRIOR_SCALES = (0.3, 0.5, 0.7)
_LM_SCALES = (0.7, 1.0, 1.3, 1.6)


def eow_phon_ctc_ls960_base():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_ctc_eow_phon"

    train_settings = build_phon_train_settings()
    train_data = build_eow_phon_ctc_fsa_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size  # = #EOW phonemes + 1 (incl. [BLANK] at idx 0)
    dev_dataset_tuples, test_dataset_tuples = build_dev_test_dataset_tuples(train_settings)

    # CTC lexicon: [BLANK] at index 0, then EOW phonemes. Used for both the training FSA and the
    # recognition (with sentence-boundary lemmata added for the LM pass).
    ctc_lexicon = get_ctc_eow_lexicon(g2p_librispeech_key="train-other-960")
    ctc_recog_lexicon = AddSentenceBoundaryLemmataToPhmmLexiconJob(ctc_lexicon).out_lexicon
    # The ``ctc`` tree builder requires context-independent phonemes; flip the variation flag for
    # recognition only. Single-state monophone tying => label/blank indices are unchanged, and the
    # training lexicon (ctc_lexicon, used by the topology="ctc" FSA) is left untouched.
    ctc_recog_lexicon = MakeLexiconContextIndependentJob(ctc_recog_lexicon).out_lexicon
    librispeech_corpus = get_bliss_corpus_dict(audio_format="ogg")["train-other-960"]

    # CTC training FSA: same exporter as the pHMM, but topology="ctc".
    fsa_exporter_config = build_fsa_exporter_config(
        lexicon_path=ctc_lexicon,
        corpus_path=librispeech_corpus,
        topology="ctc",
    )
    # Train only (the per-epoch no-prior lexicon search is gone -- recognition is the focused prior +
    # LM-scale sweep below).
    artifacts = train_and_lexicon_search(
        prefix_name=prefix_name,
        train_subdir="eow_phon",
        model_name_suffix="_ctc",
        train_data=train_data,
        vocab_size_without_blank=vocab_size_without_blank,
        fsa_exporter_config=fsa_exporter_config,
        returnn=make_librasr_returnn(),
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        run_lexicon_search=False,
    )
    # Consumed by the lexicon-free CTC recognition entry point (ctc_phon.lexfree_baseline).
    artifacts["ctc_lexicon"] = ctc_lexicon
    artifacts["ctc_recog_lexicon"] = ctc_recog_lexicon

    # --- Focused word-LM recognition (dev-other, ep{_RECOG_EPOCH}; recognition-only, training + train
    # lexicon untouched). Decode the scaled likelihood cost = -log p(l|x) + prior_scale*log p(l) and
    # sweep BOTH the acoustic-prior scale and the 4-gram word-LM scale (the two are coupled for a peaky
    # CTC AM). Reports WER (primary) + auxiliary PER. Variant family: "4gram-word-lm prior{p}_lm{s}".
    ctc_prior_file = compute_phon_prior(artifacts, epoch=_RECOG_EPOCH)

    def _build_word_lm_recog(lm_scale, prior_scale):
        return {
            "decoder_module": "phmm.decoder.rasr_phmm_v1",
            "decoder_config": RasrDecoderConfig(
                rasr_config_file=build_librasr_ctc_recognition_config(
                    lexicon_path=ctc_recog_lexicon,
                    lm_config=get_4gram_lm_rasr_config(lexicon_file=ctc_recog_lexicon, scale=lm_scale),
                    logfile_suffix="ctc_phon_recog",
                ),
                lexicon=ctc_recog_lexicon,
                silence_label="[BLANK]",
                prior_file=ctc_prior_file,
                prior_scale=prior_scale,
            ),
            "search_kwargs": {
                "forward_config": {
                    "num_workers_per_gpu": 0,
                    "torch_dataloader_opts": {"num_workers": 0},
                }
            },
        }

    for _prior_scale in _PRIOR_SCALES:
        _ptag = f"{_prior_scale:.2f}".replace(".", "p")
        lm_scale_sweep_dev_other(
            artifacts=artifacts,
            make_recog=lambda lm, ps=_prior_scale: _build_word_lm_recog(lm, ps),
            lm_scales=_LM_SCALES,
            epoch=_RECOG_EPOCH,
            search_type="lexicon",
            variant_prefix="4gram-word-lm",
            returnn=artifacts["default_returnn"],
            per_lexicon=ctc_lexicon,
            hyp_is_phonemes=False,
            report_wer=True,
            scale_label=f"prior{_ptag}_lm",
            dataset_keys=("dev-other", "test-other"),
        )

    # Greedy phoneme PER (frame-argmax, NO LM / NO lexicon / NO search) -- the clean AM-quality probe.
    # Directly comparable to the pHMM greedy PER: if CTC ~= pHMM here, the CTC AM is fine and the
    # word-constrained search degrades it; if CTC >> pHMM, the CTC AM itself is genuinely weaker.
    greedy_phon_per_dev_other(
        artifacts=artifacts,
        lexicon=ctc_lexicon,
        silence_label="[BLANK]",
        epoch=_RECOG_EPOCH,
        returnn=artifacts["default_returnn"],
        per_lexicon=ctc_lexicon,
    )
    return artifacts
