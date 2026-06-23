"""
Posterior HMM baseline with NON-EOW (plain monophone) phonemes (lexicon-constrained recognition).

Identical to :func:`...phmm_phon.baseline.eow_phon_phmm_ls960_base` (same shared :mod:`...phon_am`
AM / schedule / 4-GPU param-averaging / SpecAugment / aux losses / HMM topology / recognition), with
the only name-implied difference being the phoneme inventory: plain monophones instead of the
EOW-augmented set. That means the lexicon is built without ``AddEowPhonemesToLexiconJob``
(:func:`get_phmm_phon_lexicon`) and the training data uses the matching non-EOW vocab
(:func:`build_phon_phmm_training_datasets`).
"""
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict

from ...data.phon import build_phon_phmm_training_datasets, get_phmm_phon_lexicon
from ...lm import get_4gram_lm_rasr_config
from ...phon_am import (
    build_phon_train_settings,
    build_dev_test_dataset_tuples,
    make_librasr_returnn,
    train_and_lexicon_search,
    lm_scale_sweep_dev_other,
    compute_phon_prior,
)
from ...pytorch_networks.phmm.decoder.rasr_phmm_v1 import DecoderConfig as RasrDecoderConfig
from ...rasr import (
    AddSentenceBoundaryLemmataToPhmmLexiconJob,
    build_fsa_exporter_config,
    build_librasr_phmm_recognition_config,
)

_RECOG_EPOCH = 125
_PRIOR_SCALE = 0.3
_PRIOR_TAG = "0p30"
_LM_SCALES = (0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5)


def phon_phmm_ls960_base():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phmm_phon"

    train_settings = build_phon_train_settings()
    train_data = build_phon_phmm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size
    dev_dataset_tuples, test_dataset_tuples = build_dev_test_dataset_tuples(train_settings)

    # Plain monophone lexicon ([SILENCE] at index 0, no EOW augmentation); recognition adds the
    # sentence-boundary lemmata.
    phmm_lexicon = get_phmm_phon_lexicon(g2p_librispeech_key="train-other-960")
    phmm_recog_lexicon = AddSentenceBoundaryLemmataToPhmmLexiconJob(phmm_lexicon).out_lexicon
    librispeech_corpus = get_bliss_corpus_dict(audio_format="ogg")["train-other-960"]
    fsa_exporter_config = build_fsa_exporter_config(
        lexicon_path=phmm_lexicon,
        corpus_path=librispeech_corpus,
    )
    artifacts = train_and_lexicon_search(
        prefix_name=prefix_name,
        train_subdir="phon",
        model_name_suffix="",
        train_data=train_data,
        vocab_size_without_blank=vocab_size_without_blank,
        fsa_exporter_config=fsa_exporter_config,
        returnn=make_librasr_returnn(),
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        run_lexicon_search=False,
    )
    artifacts["phmm_lexicon"] = phmm_lexicon

    prior_file = compute_phon_prior(artifacts, epoch=_RECOG_EPOCH)

    def _build_word_lm_recog(lm_scale):
        return {
            "decoder_module": "phmm.decoder.rasr_phmm_v1",
            "decoder_config": RasrDecoderConfig(
                rasr_config_file=build_librasr_phmm_recognition_config(
                    lexicon_path=phmm_recog_lexicon,
                    lm_config=get_4gram_lm_rasr_config(lexicon_file=phmm_recog_lexicon, scale=lm_scale),
                    logfile_suffix="phmm_phon_recog",
                ),
                lexicon=phmm_recog_lexicon,
                prior_file=prior_file,
                prior_scale=_PRIOR_SCALE,
            ),
            "search_kwargs": {
                "forward_config": {
                    "num_workers_per_gpu": 0,
                    "torch_dataloader_opts": {"num_workers": 0},
                }
            },
        }

    lm_scale_sweep_dev_other(
        artifacts=artifacts,
        make_recog=_build_word_lm_recog,
        lm_scales=_LM_SCALES,
        epoch=_RECOG_EPOCH,
        search_type="lexicon",
        variant_prefix=f"4gram-word-lm prior{_PRIOR_TAG}",
        returnn=artifacts["default_returnn"],
        per_lexicon=phmm_lexicon,
        hyp_is_phonemes=False,
        report_wer=True,
        dataset_keys=("dev-other", "test-other"),
    )
    return artifacts
