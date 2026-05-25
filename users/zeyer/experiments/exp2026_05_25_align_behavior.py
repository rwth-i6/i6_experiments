"""
Alignment behavior of CTC: full-sum training variants on LibriSpeech.

Project note: ``2026-05-25-align-behavior.md``.

Runs:

- Forced alignment of in-house CTC models with priors / transition probs.
- Normalized-gradient CTC variants (training-side change).
- Blank-separated CTC variants (training-side change).
- TSE / WER metrics vs. a GMM reference alignment on LibriSpeech.

Predecessor: :mod:`exp2024_09_16_grad_align`, from which most heavy lifting is
imported (model registry, forced-align job builders, prior estimation, metric
job). The grad-align extraction block from that recipe is **deliberately not
called here** -- it is driven by :mod:`exp2026_05_23_grad_align` instead.

Convention: ``prefix`` controls aliases / output paths only. Job hashes match
those that ``exp2024_09_16_grad_align`` would produce for the same variants,
so trained CTC checkpoints under ``exp2024_04_23_baselines.ctc`` are reused.
"""

from __future__ import annotations

from sisyphus import tk, Path

# Imported (do not modify): helpers + job classes from the predecessor recipes.
from .exp2024_09_09_grad_align import CalcAlignmentMetrics
from .exp2024_09_16_grad_align import (
    sis_get_ctc_model,
    ctc_forced_align,
    get_ctc_prior,
    get_ctc_ref_label_log_probs,
)


# (shortname_for_alias, fullname_of_trained_ctc_model, vocab_key)
# WERs (dev-other/test-other) from earlier experiments; comments are TSE-LR/center [ms].
_VARIANTS = [
    # SPM 10k baselines + normalized-gradient + blank-sep variants
    (
        "base",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001",
        "spm10k",
    ),  # 5.77/6.03; TSE 68.2/52.0
    (
        "lpNormedGradC05_11P1",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001-lpNormedGradC05_11P1",
        "spm10k",
    ),  # 5.71/5.87; batch est.
    (
        "lpNormedGradC05_11P1Seq",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001-lpNormedGradC05_11P1Seq",
        "spm10k",
    ),  # 5.83/5.91; seq est.
    (
        "lpNormedGradC01_11P1",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001-lpNormedGradC01_11P1",
        "spm10k",
    ),  # 6.21/6.55; tight clamp
    (
        "blankSep",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001-blankSep",
        "spm10k",
    ),  # 5.73/6.02
    (
        "lpNormedGradC05_11P1+blankSep",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001-blankSep-lpNormedGradInclBlank",
        "spm10k",
    ),  # 5.73/6.08
    # SPM 512 baselines + blank-sep
    (
        "base-spm512",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm512-bpeSample001",
        "spm512",
    ),  # 5.97/6.21
    (
        "base-spm512-blankSep",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm512-bpeSample001-blankSep",
        "spm512",
    ),  # 6.02/6.04
    # BPE 10k baseline + blank-sep
    (
        "base-bpe10k",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-bpe10k-bpeSample001",
        "bpe10k",
    ),  # 6.18/6.35
    (
        "base-bpe10k-blankSep",
        "v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-bpe10k-bpeSample001-blankSep",
        "bpe10k",
    ),  # 5.98/6.13
]


# LibriSpeech GMM reference alignment (Tina). Same as in exp2024_09_16_grad_align.
_GMM_ALIGNMENT_ALLOPHONES = Path(
    "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
)
_GMM_ALIGNMENT_SPRINT_CACHE = Path(
    "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"
)
_FEATURES_SPRINT_CACHE = Path(  # for exact timings
    "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/features/extraction/FeatureExtractionJob.VTLN.upmU2hTb8dNH/output/vtln.cache.bundle"
)
_SEQ_LIST_REF = Path(
    "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
)


def py():
    """Sisyphus entry point."""
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_raw_v2,
        get_vocab_by_str,
        seq_list_960_to_split_100_360_500,
    )
    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

    prefix = "exp2026_05_25_align_behavior/"

    vocabs = {
        "spm10k": ("spm", ExtractSentencePieceVocabJob(get_vocab_by_str("spm10k").model_file).out_vocab, 10_240),
        "spm512": ("spm", ExtractSentencePieceVocabJob(get_vocab_by_str("spm512").model_file).out_vocab, 512),
        "bpe10k": ("bpe", get_vocab_by_str("bpe10k").vocab, 10_025),
    }
    seq_list = seq_list_960_to_split_100_360_500(_SEQ_LIST_REF)

    for shortname, fullname, vocab in _VARIANTS:
        task = get_librispeech_task_raw_v2(vocab=vocab)
        train_dataset = task.train_dataset.copy_train_as_static()
        train_dataset.main_dataset["seq_list_filter_file"] = seq_list

        for epoch in [-1]:
            ctc_model = sis_get_ctc_model(fullname, epoch=epoch)

            # Prior on full train dataset.
            prior_stats = get_ctc_prior(ctc_model, task.train_dataset.copy_train_as_static(), {"fix_log_probs": True})
            prior_stats.mean.creator.add_alias(f"{prefix}ctc_prior/{shortname}-ep{epoch}/prior_stats_full")
            tk.register_output(f"{prefix}ctc_prior/{shortname}-ep{epoch}/prior_stats_full.mean.txt", prior_stats.mean)

            opts_variants = [{}]
            for shift in [0, -5, -10, -15, -18, -20, -25]:
                opts_variants.append({"fix_log_probs": True, "blank_logit_shift": shift})
            for am_scale, prior_scale in [(1.0, 1.0), (1.0, 1.5), (1.0, 2.0), (1.0, 3.0)]:
                opts_variants.append(
                    {
                        "fix_log_probs": True,
                        "ctc_prior_type": "static",
                        "static_prior": {"type": "prob", "file": prior_stats.mean},
                        "ctc_am_scale": am_scale,
                        "ctc_prior_scale": prior_scale,
                    }
                )
            for shift in [-5, -10, -15, -20]:
                opts_variants.append(
                    {
                        "fix_log_probs": True,
                        "blank_logit_shift": shift,
                        "ctc_prior_type": "static",
                        "static_prior": {"type": "prob", "file": prior_stats.mean},
                        "ctc_am_scale": 1.0,
                        "ctc_prior_scale": 1.0,
                    }
                )

            for opts in opts_variants:
                name = f"{shortname}-ep{epoch}"
                for k, v in opts.items():
                    if k == "fix_log_probs" and v:
                        name += "-fix"
                        continue
                    if k == "static_prior":
                        continue
                    name += f"-{k}{v}"

                ref_log_probs = get_ctc_ref_label_log_probs(ctc_model, train_dataset, opts)
                ref_log_probs.creator.add_alias(f"{prefix}ctc_ref_log_probs/{name}/log_probs")
                tk.register_output(f"{prefix}ctc_ref_log_probs/{name}/log_probs.hdf", ref_log_probs)

                prefix_ = f"{prefix}ctc_forced_align/{name}/"
                alignment = ctc_forced_align(ctc_model, train_dataset, opts)
                alignment.creator.add_alias(f"{prefix_}align")
                tk.register_output(f"{prefix_}align.hdf", alignment)

                job = CalcAlignmentMetrics(
                    seq_list=seq_list,
                    seq_list_ref=_SEQ_LIST_REF,
                    alignment_hdf=alignment,
                    alignment_label_topology="ctc",
                    alignment_bpe_vocab=vocabs[vocab][1],
                    alignment_bpe_style=vocabs[vocab][0],
                    alignment_blank_idx=vocabs[vocab][2],
                    features_sprint_cache=_FEATURES_SPRINT_CACHE,
                    ref_alignment_sprint_cache=_GMM_ALIGNMENT_SPRINT_CACHE,
                    ref_alignment_allophones=_GMM_ALIGNMENT_ALLOPHONES,
                    ref_alignment_len_factor=6,
                )
                job.add_alias(f"{prefix_}align-metrics")
                tk.register_output(f"{prefix_}align-metrics.json", job.out_scores)
                tk.register_output(f"{prefix_}align-metrics.short_report.txt", job.out_short_report_str)

    # TODO Switchboard phoneme-based HMM/CTC variants (Tina): not driven from this
    # recipe yet -- prior numbers come from a separate i6-Aachen setup.

    # TODO Convergence-rate appendix figure: see exp2024_09_16_grad_align.visualize_train_scores
    # (standalone, not Sis). Re-run from there for the relevant shortnames.

    # NOTE Synthetic-framework experiments (FFNN / Conv / LSTM on artificial data)
    # run from the standalone repo ``2024-alignment-analysis`` -- laptop-CPU, no Sis.
