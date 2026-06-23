"""
BPE CTC baseline (lexicon-constrained recognition) -- the BPE analog of the EOW-phoneme CTC baseline
(:func:`...ctc_phon.baseline.eow_phon_ctc_ls960_base`), started at BPE-512.

It reuses the EXACT shared AM path (:mod:`...phon_am`): the ``phmm.phmm_zhou`` Conformer, the ``fbw2``
full-sum CTC loss over a RASR FSA with ``topology="ctc"``, the OCLR-style /4 schedule, 4-GPU
``torch_distributed`` parameter-averaging, bf16 AMP, the aux CTC losses, label smoothing and the data
settings. Only the name-implied pieces differ from the phon CTC baseline:

* the label inventory is BPE **subwords** (:class:`...rasr.CreateCorpusBpeFsaLexiconJob`), not EOW
  phonemes -- ``[BLANK]`` at index 0 (``special="blank"``, no empty ``""`` orth) followed by the BPE
  units; every corpus word decomposes to its subword-nmt pronunciation, so the one lexicon doubles as
  the training-FSA lexicon and the word-LM search tree (word boundaries are encoded by the ``@@``
  continuation markers, the BPE analog of the EOW ``#``);
* recognition is the same CTC tree search (:func:`...rasr.build_librasr_ctc_recognition_config`) + the
  official 4-gram **word** LM (the lexicon maps words -> BPE, the word LM scores the word orths at the
  tree leaves);
* the metric is **WER** throughout -- BPE is losslessly invertible to words, so there is no auxiliary
  PER and no phoneme/silence duration analysis (both phoneme-specific and meaningless for BPE units).

All recognition is recognition-only (training + train lexicon untouched). WERs feed
``ls960_ctc_bpe_<size>/summary.report`` (the same live combined report machinery as the phon configs).
"""
import copy
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict

from ...data.bpe import build_bpe_ctc_fsa_training_datasets, get_bpe_ctc_lexicon
from ...lm import get_4gram_lm_rasr_config
from ...phon_am import (
    build_phon_train_settings,
    build_dev_test_dataset_tuples,
    make_librasr_returnn,
    train_and_lexicon_search,
    lm_scale_sweep_dev_other,
    compute_phon_prior,
)
from ...pipeline import search
from ...results import add_result
from ...pytorch_networks.phmm.decoder.rasr_phmm_v1 import DecoderConfig as RasrDecoderConfig
from ...rasr import (
    AddSentenceBoundaryLemmataToPhmmLexiconJob,
    MakeLexiconContextIndependentJob,
    build_fsa_exporter_config,
    build_librasr_ctc_recognition_config,
)

_BPE_SIZE = 512
_RECOG_EPOCH = 125  # the final checkpoint (matches the phon CTC / pHMM sweep epoch)
# JOINT prior x word-LM-scale sweep, mirroring the phon CTC recipe (the acoustic prior flattens the
# peaky/blank-dominated CTC posterior, which couples with the word-LM scale). Moderate first-cut grid;
# extend as needed once the BPE-512 optimum is bracketed on dev-other. The prior=0.3 rows reuse the
# cached prior; only the lm/prior tail is new.
_PRIOR_SCALES = (0.3, 0.5)
_LM_SCALES = (0.5, 0.7, 1.0, 1.3)


def bpe_ctc_ls960_base(bpe_size: int = _BPE_SIZE):
    prefix_name = f"example_setups/librispeech/phmm_standalone_2024/ls960_ctc_bpe_{bpe_size}"

    train_settings = build_phon_train_settings()
    train_data = build_bpe_ctc_fsa_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=bpe_size,
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size  # = #BPE subwords + 1 (incl. [BLANK] at idx 0)
    dev_dataset_tuples, test_dataset_tuples = build_dev_test_dataset_tuples(train_settings)

    # CTC BPE lexicon: [BLANK] at index 0 (special="blank", no empty "" orth), then BPE subwords; each
    # word -> subword pron. Used for the training FSA and (with sentence-boundary lemmata added + the
    # phonemes flipped to context-independent) for recognition.
    ctc_bpe_lexicon = get_bpe_ctc_lexicon(librispeech_key="train-other-960", bpe_size=bpe_size)
    # Add <s>/</s> lemmata for the word-LM pass FIRST, then flip variation->none for the ctc tree builder
    # (the order matches the phon CTC recog lexicon). MakeLexiconContextIndependentJob is a NO-OP here --
    # CreateCorpusBpeFsaLexiconJob already emits variation="none" -- but is kept for structural parity and
    # to guarantee the ctc builder's context-independence assertion.
    ctc_bpe_recog_lexicon = AddSentenceBoundaryLemmataToPhmmLexiconJob(ctc_bpe_lexicon).out_lexicon
    ctc_bpe_recog_lexicon = MakeLexiconContextIndependentJob(ctc_bpe_recog_lexicon).out_lexicon
    librispeech_corpus = get_bliss_corpus_dict(audio_format="ogg")["train-other-960"]

    # CTC training FSA: same exporter as the phon CTC, topology="ctc", over the BPE lexicon. Default
    # orthographic-parser settings (NOT DEFAULT_CTC_BPE_RASR_CONFIG) -- topology is passed explicitly.
    fsa_exporter_config = build_fsa_exporter_config(
        lexicon_path=ctc_bpe_lexicon,
        corpus_path=librispeech_corpus,
        topology="ctc",
    )
    # Train only; recognition is the focused prior + word-LM-scale sweep below.
    artifacts = train_and_lexicon_search(
        prefix_name=prefix_name,
        train_subdir="bpe",
        model_name_suffix=f"_ctc_bpe{bpe_size}",
        train_data=train_data,
        vocab_size_without_blank=vocab_size_without_blank,
        fsa_exporter_config=fsa_exporter_config,
        returnn=make_librasr_returnn(),
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        run_lexicon_search=False,
    )
    # Consumed by the lexicon-free BPE CTC recognition entry point (ctc_bpe.lexfree_baseline).
    artifacts["ctc_bpe_lexicon"] = ctc_bpe_lexicon
    artifacts["ctc_bpe_recog_lexicon"] = ctc_bpe_recog_lexicon

    # --- Focused word-LM recognition (dev-other + test-other, ep{_RECOG_EPOCH}; recognition-only).
    # Decode the scaled likelihood cost = -log p(l|x) + prior_scale*log p(l) and sweep BOTH the acoustic
    # prior scale and the 4-gram word-LM scale (coupled for a peaky CTC AM). Reports WER. Variant family:
    # "4gram-word-lm prior{p}_lm{s}".
    ctc_prior_file = compute_phon_prior(artifacts, epoch=_RECOG_EPOCH)

    def _build_word_lm_recog(lm_scale, prior_scale):
        return {
            "decoder_module": "phmm.decoder.rasr_phmm_v1",
            "decoder_config": RasrDecoderConfig(
                rasr_config_file=build_librasr_ctc_recognition_config(
                    lexicon_path=ctc_bpe_recog_lexicon,
                    lm_config=get_4gram_lm_rasr_config(lexicon_file=ctc_bpe_recog_lexicon, scale=lm_scale),
                    logfile_suffix="ctc_bpe_recog",
                ),
                lexicon=ctc_bpe_recog_lexicon,
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
            per_lexicon=None,  # BPE WER is the native metric -- no auxiliary PER
            hyp_is_phonemes=False,
            report_wer=True,
            scale_label=f"prior{_ptag}_lm",
            dataset_keys=("dev-other", "test-other"),
        )

    # Greedy BPE WER (frame-argmax -> collapse repeats -> drop [BLANK] -> de-BPE -> words): the clean
    # AM-quality probe (NO LM / NO lexicon / NO search), the BPE analog of the phon greedy PER. Because
    # BPE is invertible to words it yields a WER directly (vs. the phoneme greedy's PER).
    _greedy_bpe_wer(artifacts, lexicon=ctc_bpe_lexicon, epoch=_RECOG_EPOCH)

    return artifacts


def _greedy_bpe_wer(artifacts, *, lexicon, epoch, dataset_key: str = "dev-other"):
    """Greedy BPE decode (real-RETURNN ``greedy_bpe_v1``) on one dev set, registered as a WER row."""
    asr_model = artifacts["asr_models_by_epoch"].get(epoch)
    if asr_model is None:
        print(f"[_greedy_bpe_wer] no asr_model for epoch {epoch}; skipping")
        return

    prefix_name = artifacts["prefix_name"]
    model_name = artifacts["model_name"]
    returnn = artifacts["default_returnn"]
    dev_dataset_tuples = artifacts["dev_dataset_tuples"]
    assert dataset_key in dev_dataset_tuples, f"{dataset_key!r} not in {list(dev_dataset_tuples)}"
    one_dataset = {dataset_key: dev_dataset_tuples[dataset_key]}

    search_name = f"{prefix_name}/lexicon-free-search/{model_name}/greedy/recog_ep{epoch}"
    asr_model_copy = copy.deepcopy(asr_model)
    asr_model_copy.prior_file = None  # greedy uses raw posteriors -- no prior, no LM
    search_jobs, wers = search(
        search_name,
        forward_config={},
        asr_model=asr_model_copy,
        decoder_module="phmm.decoder.greedy_bpe_v1",
        decoder_args={"config": {"lexicon": lexicon, "silence_label": "[BLANK]"}},
        test_dataset_tuples=one_dataset,
        use_gpu=True,
        **returnn,
    )
    for search_job in search_jobs:
        # device="gpu" with no gpu_mem defaults to the flaky gpu_11gb pool (cuInit fails); pin gpu_24gb.
        search_job.rqmt["gpu_mem"] = 24
    add_result(
        prefix_name,
        search_type="lexicon-free",
        model=model_name,
        variant="greedy",
        epoch=epoch,
        wers={name.split("/")[-1]: wer for name, wer in wers.items()},
    )
