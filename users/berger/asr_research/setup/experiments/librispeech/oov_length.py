"""6e07a2e5: beta look-ahead, gated by delayed-cost and first-piece probes.

The historical run paths below deliberately identify completed 5083edde jobs.
They are immutable inputs for the scoring-balance audit, not new-build baselines.
Full-corpus recognition must use new-build controls after reviewing preflight.
"""

from dataclasses import replace
import math

from sisyphus import tk

from ... import tools
from ...data.huggingface import get_huggingface_repo_dir
from ...data.librispeech import datasets
from ...data.oov_diagnostics import GraniteWordStartMarginJob, OovScoringBalanceJob
from ...data.oov_length_smoke import (
    OovLengthDecoderSmokeJob,
    OovDelayedPenaltyProbeJob,
    OovFirstPiecePruningProbeJob,
    RequireOovPruningProbeJob,
)
from .recognition.ctc_granite import GRANITE_CTC_REPO_ID, GRANITE_CTC_REVISION


REVISION = "6e07a2e5"
PREFIX = f"datasets/LibriSpeech/oov-{REVISION}"
TOOLS_PREFIX = f"tools/rasr-oov-{REVISION}"
WORK = "/work/asr4/berger/sisyphus_work_dirs/librispeech/20241106_seq2seq_template_setups"


def run_preflight():
    """Schedule new-build acceptance checks; reuse completed AM diagnostics.

    No dev-reference words are used to expand the lexicon. No ASR WER sweep is
    scheduled before the acoustic margins and alpha_0 have been reviewed.
    """
    from ...data.huggingface import get_hf_ctc_vocab_file
    from .oov_prototype import _validated_lexicon

    binary = tools.get_oov_lookahead_rasr_binary_path()
    smoke = OovLengthDecoderSmokeJob(binary, tools.returnn_python_exe)
    smoke.add_alias(f"{TOOLS_PREFIX}/decoder-smoke")
    tk.register_output(f"{TOOLS_PREFIX}/decoder-smoke-report", smoke.out_report)
    tk.register_output(f"{TOOLS_PREFIX}/decoder-smoke-logs", smoke.out_logs)
    probe = OovDelayedPenaltyProbeJob(binary, tools.returnn_python_exe)
    probe.add_alias(f"{TOOLS_PREFIX}/delayed-penalty-probe")
    tk.register_output(f"{TOOLS_PREFIX}/delayed-penalty-probe", probe.out_report)
    tk.register_output(f"{TOOLS_PREFIX}/delayed-penalty-logs", probe.out_logs)
    first_piece = OovFirstPiecePruningProbeJob(binary, tools.returnn_python_exe)
    first_piece.add_alias(f"{TOOLS_PREFIX}/first-piece-probe")
    tk.register_output(f"{TOOLS_PREFIX}/first-piece-report", first_piece.out_report)
    tk.register_output(f"{TOOLS_PREFIX}/first-piece-logs", first_piece.out_logs)
    pruning = RequireOovPruningProbeJob(probe.out_report, first_piece_report=first_piece.out_report)
    pruning.add_alias(f"{TOOLS_PREFIX}/pruning-acceptance")
    tk.register_output(f"{TOOLS_PREFIX}/pruning-acceptance", pruning.out_report)
    lexicon = tk.Path(
        f"{WORK}/i6_experiments/example_setups/seq2seq_rasr_2025/data/oov_prototype/"
        "ValidateOovPrototypeInputsJob.7PHGwFjw2wIz/output/lexicon.xml.gz"
    )
    audit = OovScoringBalanceJob(
        lexicon=lexicon,
        baseline_log=tk.Path(f"{WORK}/i6_core/returnn/forward/ReturnnForwardJobV2.LnDeBy7OB35x/log.run.1"),
        baseline_config=tk.Path(f"{WORK}/i6_core/rasr/config/WriteRasrConfigJob.HbArxYUxtqVH/output/rasr.config"),
        collapsed_log=tk.Path(f"{WORK}/i6_core/returnn/forward/ReturnnForwardJobV2.4er9wq6S9th7/log.run.1"),
        collapsed_config=tk.Path(f"{WORK}/i6_core/rasr/config/WriteRasrConfigJob.YKdnTkbZddvS/output/rasr.config"),
    )
    audit.add_alias(f"{PREFIX}/scoring-balance")
    tk.register_output(f"{PREFIX}/scoring-balance", audit.out_report)
    tk.register_output(f"{PREFIX}/alpha0", audit.out_alpha0)
    repo = get_huggingface_repo_dir(GRANITE_CTC_REPO_ID, revision=GRANITE_CTC_REVISION)
    state_tying = _validated_lexicon(
        lexicon,
        get_hf_ctc_vocab_file(repo, blank_index=0),
        binary,
        f"{PREFIX}/granite-boundary-indices",
        decoder_smoke_report=smoke.out_report,
    )
    # The margin measurement uses only the AM/data, not RASR. Reconstruct its
    # original dependency so the completed GPU job is reused without rehashing.
    old_smoke = OovLengthDecoderSmokeJob(tools.get_oov_length_rasr_binary_path(), tools.returnn_python_exe)
    margins = GraniteWordStartMarginJob(
        repo=repo,
        data_config=replace(datasets.get_default_recog_data("dev-other"), peak_normalization=False, preemphasis=None),
        returnn_root=tools.returnn_root,
        correctness_report=old_smoke.out_report,
    )
    margins.add_alias(f"{PREFIX}/word-start-margins")
    tk.register_output(f"{PREFIX}/word-start-margins", margins.out_report)
    tk.register_output(f"{PREFIX}/word-start-samples", margins.out_samples)
    return dict(
        binary=binary,
        correctness=smoke,
        delayed_penalty_probe=probe,
        first_piece_probe=first_piece,
        pruning=pruning,
        state_tying=state_tying.creator,
        scoring_balance=audit,
        margins=margins,
    )


def _guard_serializers(lexicon, tokenization, repo=None):
    from i6_experiments.common.setups.serialization import Call, Import
    from sisyphus.delayed_ops import DelayedFormat
    from ...model_pipelines.common.oov_guard import GuardedOovTracebackFormatter

    kwargs = [("lexicon", DelayedFormat('tk.Path("{}")', lexicon)), ("tokenization", repr(tokenization))]
    if repo is not None:
        kwargs.append(("huggingface_repo_dir", DelayedFormat('tk.Path("{}")', repo)))
    return [
        Import(f"{GuardedOovTracebackFormatter.__module__}.{GuardedOovTracebackFormatter.__name__}"),
        Call("GuardedOovTracebackFormatter", kwargs=kwargs, return_assign_variables="traceback_formatter"),
    ]


def length_variants(
    kind,
    lexicons,
    binary,
    *,
    lm_conditions,
    alpha0_by_lm=None,
    phase="comparison",
    repo=None,
    betas=(-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0),
    alpha_factors=(0.5, 0.75, 1.0, 1.5, 2.0),
    alphas=None,
):
    """Opt-in A/B/C or BPE legacy/excluding grids after reviewing diagnostics.

    lm_conditions maps descriptive names to ONE word-LM config (or None).
    Granite defaults to alpha=0 unless an explicit `alphas` grid is supplied.
    For BPE, alpha0_by_lm must be supplied explicitly
    and calibrated separately for each model/LM scale, except in beta-only mode,
    which fixes alpha=0 to isolate the upfront-beta change.
    A sweep can instead supply explicit `alphas` in cost units; these are
    exploratory settings, not a claim of measured alpha0 calibration.
    `lexicons` maps closed/A-100k/A-300k/A-full/B/C-100k/C-300k/C-full for Granite,
    or closed/legacy/known-excluding for BPE. No subword scorer is added.
    """
    from .recognition import ctc_bpe, ctc_granite, ffnn_transducer_bpe

    if kind not in {"bpe", "transducer", "granite"} or phase not in {"comparison", "sweep", "beta-only"}:
        raise ValueError((kind, phase))
    if alphas is not None:
        if phase != "sweep" or alpha0_by_lm is not None:
            raise ValueError("Explicit alphas require sweep mode and no alpha0 calibration")
        alphas = tuple(float(alpha) for alpha in alphas)
        if not alphas or any(not math.isfinite(alpha) or alpha < 0 for alpha in alphas):
            raise ValueError("Explicit alphas must be nonempty, finite and nonnegative")
    module = {"bpe": ctc_bpe, "transducer": ffnn_transducer_bpe, "granite": ctc_granite}[kind]
    tokenization = "word-start-marked" if kind == "granite" else "continuation-marked"
    variants, seen = [], set()
    for lm_name, lm in lm_conditions.items():
        if kind == "granite":
            if alpha0_by_lm and any(float(value) != 0 for value in alpha0_by_lm.values()):
                raise ValueError("Use explicit alphas for Granite; do not reuse the old alpha0 calibration")
            alpha0 = 0.0
        elif phase == "beta-only":
            if alpha0_by_lm and any(float(value) != 0 for value in alpha0_by_lm.values()):
                raise ValueError("The beta-only phase fixes alpha=0")
            alpha0 = 0.0
        elif alphas is not None:
            alpha0 = 0.0  # Not used: explicit grid below.
        else:
            alpha0 = float((alpha0_by_lm or {})[lm_name])
        if (
            not math.isfinite(alpha0)
            or alpha0 < 0
            or (kind != "granite" and phase == "sweep" and alphas is None and alpha0 == 0)
        ):
            raise ValueError("Provide a finite alpha calibration (strictly positive for a sweep)")
        for approach, lexicon in lexicons.items():
            if kind == "granite":
                if approach != "closed" and not (approach == "B" or approach.startswith(("A-", "C-"))):
                    raise ValueError(approach)
                fallback = approach == "B" or approach.startswith("C-")
                mode = "known-excluding" if fallback else None
            else:
                if approach not in {"closed", "legacy", "known-excluding"}:
                    raise ValueError(approach)
                fallback = approach != "closed"
                mode = "disabled" if approach == "closed" else approach
            piece_cost = fallback and mode == "known-excluding"
            grid_betas = (0.0,) if phase == "comparison" or approach == "closed" else betas
            grid_alphas = (
                ((alpha0,) if phase != "sweep" else tuple(alpha0 * f for f in alpha_factors)) if piece_cost else (0.0,)
            )
            if piece_cost and alphas is not None:
                grid_alphas = alphas
            for beta in grid_betas:
                for alpha in grid_alphas:
                    if not math.isfinite(beta) or not math.isfinite(alpha):
                        raise ValueError("Nonfinite cost")
                    # One matched-budget and one wider-budget point. The latter
                    # is a convergence probe, not a claim of matched quality.
                    beams = (256, 1024) if phase == "comparison" else (256,)
                    for beam in beams:
                        key = (approach, lm_name, beta, alpha, beam)
                        if key in seen:
                            continue
                        seen.add(key)
                        v = module.default_offline_tree_recog_variant()
                        v.descriptor = (
                            f"oov-{REVISION}_{approach}_{lm_name}_beta{beta:g}_alpha{alpha:g}" f"_beam{beam}_thr16"
                        )
                        v.blank_penalty = 0.0
                        if hasattr(v, "prior_scale"):
                            v.prior_scale = 0.0
                        v.rasr_binary_path_override = binary
                        v.prepared_lexicon_file = lexicon
                        v.search_algorithm_params = replace(
                            v.search_algorithm_params,
                            word_lm_params=lm,
                            max_beam_sizes=[beam],
                            score_thresholds=[16.0],
                            max_word_end_beam_size=beam,
                            word_end_score_threshold=16.0,
                            sentence_end_fallback=False,
                            recombination_mode="on",
                            unknown_word_fallback=mode,
                            unknown_word_tokenization=tokenization if fallback else None,
                            unknown_word_penalty=beta,
                            unknown_piece_penalty=alpha if piece_cost else None,
                        )
                        v.search_mode_params = replace(v.search_mode_params, mem_rqmt=32)
                        if fallback:
                            v.prototype_traceback_formatter_serializers = _guard_serializers(
                                lexicon, tokenization, repo
                            )
                        variants.append(v)
    return variants


def setup_granite_lexicons(preflight, approaches=None):
    """Build the train-text-only A/B/C lexica; all depend on successful preflight."""
    from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
    from i6_experiments.common.datasets.librispeech.language_model import (
        get_arpa_lm_dict,
        get_librispeech_normalized_lm_data,
    )
    from ...data.huggingface import HuggingFaceBpeBlissLexiconJob, get_hf_ctc_vocab_file
    from ...data.oov_extended_lexicon import (
        PrepareOovLengthLexiconJob,
        RankOovLexiconCandidatesJob,
        ExtendGraniteOovLexiconJob,
    )
    from ...data.oov_prototype import (
        InspectWordStartTokenizerJob,
        BuildGraniteOovLexiconJob,
        RemoveEmptyUnknownPronunciationsJob,
    )
    from .oov_prototype import _validated_lexicon

    if approaches is None:
        approaches = ("closed", "A-100k", "A-300k", "A-full", "B", "C-100k", "C-300k", "C-full")
    if not set(approaches) <= {"closed", "A-100k", "A-300k", "A-full", "B", "C-100k", "C-300k", "C-full"}:
        raise ValueError(approaches)
    repo = get_huggingface_repo_dir(GRANITE_CTC_REPO_ID, revision=GRANITE_CTC_REVISION)
    table = InspectWordStartTokenizerJob(repo, datasets.get_default_score_corpus("dev-other").bliss_corpus_file)
    base = RemoveEmptyUnknownPronunciationsJob(
        HuggingFaceBpeBlissLexiconJob(get_bliss_lexicon(), repo, blank_index=0, add_unknown_lemma=True).out_lexicon
    ).out_lexicon
    restricted = BuildGraniteOovLexiconJob(base, table.out_table)
    closed = PrepareOovLengthLexiconJob(
        restricted.out_lexicon,
        preflight["correctness"].out_report,
        preflight["margins"].out_report,
        preflight["scoring_balance"].out_report,
        strip_fallback=True,
    ).out_lexicon
    rank = RankOovLexiconCandidatesJob(get_librispeech_normalized_lm_data(), closed, get_arpa_lm_dict()["4gram"])
    tk.register_output(f"{PREFIX}/extension-candidates", rank.out_words)
    tk.register_output(f"{PREFIX}/extension-candidate-report", rank.out_report)
    files = {"closed": closed} if "closed" in approaches else {}
    if "B" in approaches:
        files["B"] = BuildGraniteOovLexiconJob(closed, table.out_table).out_lexicon
    for limit, name in ((100000, "100k"), (300000, "300k"), (None, "full")):
        if f"A-{name}" not in approaches and f"C-{name}" not in approaches:
            continue
        extension = ExtendGraniteOovLexiconJob(closed, rank.out_words, table.out_table, repo, max_candidates=limit)
        if f"A-{name}" in approaches:
            files[f"A-{name}"] = extension.out_lexicon
        if f"C-{name}" in approaches:
            files[f"C-{name}"] = BuildGraniteOovLexiconJob(extension.out_lexicon, table.out_table).out_lexicon
        tk.register_output(f"{PREFIX}/A-{name}/preparation", extension.out_report)
        tk.register_output(f"{PREFIX}/A-{name}/rejected", extension.out_rejected)
    for name, path in files.items():
        if name == "B" or name.startswith("C-"):
            files[name] = PrepareOovLengthLexiconJob(
                path,
                preflight["correctness"].out_report,
                preflight["margins"].out_report,
                preflight["scoring_balance"].out_report,
                pruning_report=preflight["pruning"].out_report,
            ).out_lexicon
    return repo, {
        name: _validated_lexicon(
            path,
            get_hf_ctc_vocab_file(repo, blank_index=0),
            preflight["binary"],
            f"{PREFIX}/granite/{name}",
            decoder_smoke_report=preflight["correctness"].out_report,
        )
        for name, path in files.items()
    }


def run_granite_phase(
    *,
    lm_conditions,
    alpha0_by_lm=None,
    phase="comparison",
    approaches=None,
    betas=(-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0),
):
    """Explicit phase driver on dev-other; broad grids remain opt-in."""
    from .recognition import ctc_granite

    preflight = run_preflight()
    repo, lexicons = setup_granite_lexicons(preflight, approaches=approaches)
    variants = length_variants(
        "granite",
        lexicons,
        preflight["binary"],
        lm_conditions=lm_conditions,
        alpha0_by_lm=alpha0_by_lm,
        phase=phase,
        repo=repo,
        betas=betas,
    )
    if phase == "comparison":
        greedy = ctc_granite.default_offline_lexfree_recog_variant()
        greedy.descriptor = f"oov-{REVISION}_lexfree-greedy"
        greedy.rasr_binary_path_override = preflight["binary"]
        variants.append(greedy)
    return ctc_granite.run(huggingface_repo_dir=repo, variants=variants, corpora=["dev-other"])


def run_approach_a():
    """Rerun the non-fallback comparison and controls on the selected build."""
    from ...data.librispeech.lm import ArpaLmParams

    return run_granite_phase(
        lm_conditions={"no-lm": None, "4gram-lm0.3": ArpaLmParams(scale=0.3)},
        approaches=("closed", "A-100k"),
    )


def limited_fallback_variants(kind, lexicons, binary, *, repo=None):
    """First full-corpus check of the accepted fix, not the full OOV matrix.

    Use the standard lexicon, alpha=0, beam=256/threshold=16 throughout. The
    beta=100 endpoint tests approach toward the matched closed-tree result.
    """
    from ...data.librispeech.lm import ArpaLmParams

    if kind == "bpe":
        conditions = {"4gram-lm0.6": ArpaLmParams(scale=0.6)}
        betas = (10.0, 12.0, 20.0, 100.0)
    elif kind == "granite":
        conditions = {"no-lm": None, "4gram-lm0.3": ArpaLmParams(scale=0.3)}
        betas = (0.0, 2.0, 4.0, 8.0, 16.0, 100.0)
    else:
        raise ValueError(kind)
    return length_variants(
        kind,
        lexicons,
        binary,
        lm_conditions=conditions,
        phase="beta-only",
        repo=repo,
        betas=betas,
    )


def setup_bpe_baseline_lexicon(ctc_model, preflight):
    """Share the exact accepted standard-lexicon preparation across both grids."""
    from ...data.librispeech.bpe import get_bpe_vocab_file, vocab_to_bpe_size
    from ...data.librispeech.lexicon import get_bpe_bliss_lexicon
    from ...data.oov_extended_lexicon import PrepareOovLengthLexiconJob
    from .oov_prototype import _validated_lexicon

    bpe_size = vocab_to_bpe_size(ctc_model.model_config.target_size - 1)
    lexicon = PrepareOovLengthLexiconJob(
        get_bpe_bliss_lexicon(bpe_size, add_blank=True, add_sentence_end_pron=False, add_oov_fallback=True),
        preflight["correctness"].out_report,
        preflight["margins"].out_report,
        preflight["scoring_balance"].out_report,
        pruning_report=preflight["pruning"].out_report,
    ).out_lexicon
    return _validated_lexicon(
        lexicon,
        get_bpe_vocab_file(bpe_size, add_blank=True),
        preflight["binary"],
        f"{PREFIX}/bpe/baseline",
        label_aliases={"</s>": "<s>"},
        decoder_smoke_report=preflight["correctness"].out_report,
    )


def run_limited_fallback(ctc_model):
    """Retain 17 dev-other regression runs and their completed job identities.

    run_all also calls run_approach_a(), which registers the matching Granite
    closed controls and greedy baseline. BPE uses the identical fallback XML
    for its closed control. No reduced lexica or adapted/subword LMs here.
    """
    from .recognition import ctc_bpe, ctc_granite

    preflight = run_preflight()
    lexicon = setup_bpe_baseline_lexicon(ctc_model, preflight)
    variants = limited_fallback_variants(
        "bpe",
        {mode: lexicon for mode in ("closed", "known-excluding")},
        preflight["binary"],
    )
    for variant in variants:
        variant.descriptor += "_lexicon-baseline"
    results = ctc_bpe.run(ctc_model, variants=variants, corpora=["dev-other"])

    repo, lexicons = setup_granite_lexicons(preflight, approaches=("B",))
    variants = limited_fallback_variants("granite", lexicons, preflight["binary"], repo=repo)
    results.extend(ctc_granite.run(huggingface_repo_dir=repo, variants=variants, corpora=["dev-other"]))
    return results


def tuning_variants(kind, lexicons, binary, *, repo=None, include_completed=False):
    """Empirical dev tuning after acceptance: BPE alpha/beta, Granite beta only."""
    from ...data.librispeech.lm import ArpaLmParams

    if kind == "bpe":
        variants = length_variants(
            kind,
            lexicons,
            binary,
            lm_conditions={"4gram-lm0.6": ArpaLmParams(scale=0.6)},
            phase="sweep",
            betas=(6.0, 8.0, 10.0, 12.0, 14.0),
            alphas=(0.0, 0.25, 0.5, 1.0, 2.0),
        )
    elif kind == "granite":
        variants = []
        for lm_name, lm, betas in (
            ("no-lm", None, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
            ("4gram-lm0.3", ArpaLmParams(scale=0.3), (4.0, 6.0, 8.0, 10.0, 12.0)),
        ):
            variants.extend(
                length_variants(
                    kind,
                    lexicons,
                    binary,
                    lm_conditions={lm_name: lm},
                    phase="beta-only",
                    repo=repo,
                    betas=betas,
                )
            )
    else:
        raise ValueError(kind)
    if include_completed:
        return variants
    completed = {v.descriptor for v in limited_fallback_variants(kind, lexicons, binary, repo=repo)}
    return [v for v in variants if v.descriptor not in completed]


def run_tuning(ctc_model):
    """Append 30 new runs; old points remain in run_limited_fallback's report."""
    from .recognition import ctc_bpe, ctc_granite

    preflight = run_preflight()
    lexicon = setup_bpe_baseline_lexicon(ctc_model, preflight)
    variants = tuning_variants("bpe", {"known-excluding": lexicon}, preflight["binary"])
    for variant in variants:
        variant.descriptor += "_lexicon-baseline"
    results = ctc_bpe.run(ctc_model, variants=variants, corpora=["dev-other"])
    repo, lexicons = setup_granite_lexicons(preflight, approaches=("B",))
    variants = tuning_variants("granite", lexicons, preflight["binary"], repo=repo)
    results.extend(ctc_granite.run(huggingface_repo_dir=repo, variants=variants, corpora=["dev-other"]))
    return results


def granite_alpha_tuning_variants(lexicons, binary, *, repo, include_completed=False):
    """Joint alpha/beta tuning, with fixed LM scales and pruning, on dev-other.

    Lower beta values allow the length cost to replace some of the per-word
    penalty. Alpha=0 is a control, not an established optimum. Completed zero
    points remain registered by the preceding regression/beta tuning stages.
    """
    from ...data.librispeech.lm import ArpaLmParams

    variants = []
    for lm_name, lm, betas in (
        ("no-lm", None, (0.0, 1.0, 2.0, 3.0, 4.0)),
        ("4gram-lm0.3", ArpaLmParams(scale=0.3), (0.0, 2.0, 4.0, 6.0, 8.0)),
    ):
        variants.extend(
            length_variants(
                "granite",
                lexicons,
                binary,
                lm_conditions={lm_name: lm},
                phase="sweep",
                repo=repo,
                betas=betas,
                alphas=(0.0, 0.25, 0.5, 1.0, 2.0),
            )
        )
    if include_completed:
        return variants
    completed = {
        v.descriptor
        for v in (
            limited_fallback_variants("granite", lexicons, binary, repo=repo)
            + tuning_variants("granite", lexicons, binary, repo=repo, include_completed=True)
        )
    }
    return [v for v in variants if v.descriptor not in completed]


def run_granite_alpha_tuning():
    """Append 40 Granite dev-other runs; defer beam tuning until their review."""
    from .recognition import ctc_granite

    preflight = run_preflight()
    repo, lexicons = setup_granite_lexicons(preflight, approaches=("B",))
    variants = granite_alpha_tuning_variants(lexicons, preflight["binary"], repo=repo)
    return ctc_granite.run(huggingface_repo_dir=repo, variants=variants, corpora=["dev-other"])


def granite_beam_tuning_variants(lexicons, binary, *, repo, include_completed=False):
    """Pruning grid at the best observed alpha/beta pairs, not retuned costs.

    Vary beam caps independently of thresholds. Within-word and word-end
    settings are tied here, so neither stage silently limits the wide probe.
    This is a convergence/runtime study, not a claim that wider means lower WER.
    """
    from ...data.librispeech.lm import ArpaLmParams

    variants = []
    for lm_name, lm, beta in (
        ("no-lm", None, 1.0),
        ("4gram-lm0.3", ArpaLmParams(scale=0.3), 6.0),
    ):
        (base,) = length_variants(
            "granite",
            lexicons,
            binary,
            lm_conditions={lm_name: lm},
            phase="sweep",
            repo=repo,
            betas=(beta,),
            alphas=(2.0,),
        )
        for beam in (64, 256, 1024):
            for threshold in (8.0, 16.0, 32.0, 64.0):
                if not include_completed and (beam, threshold) == (256, 16.0):
                    continue  # Exact existing alpha-sweep point.
                variants.append(
                    replace(
                        base,
                        descriptor=base.descriptor.removesuffix("_beam256_thr16") + f"_beam{beam}_thr{threshold:g}",
                        search_algorithm_params=replace(
                            base.search_algorithm_params,
                            max_beam_sizes=[beam],
                            max_word_end_beam_size=beam,
                            score_thresholds=[threshold],
                            word_end_score_threshold=threshold,
                        ),
                    )
                )
    return variants


def run_granite_beam_tuning():
    """Append 22 dev-other pruning runs; reuse the two completed operating points."""
    from .recognition import ctc_granite

    preflight = run_preflight()
    repo, lexicons = setup_granite_lexicons(preflight, approaches=("B",))
    variants = granite_beam_tuning_variants(lexicons, preflight["binary"], repo=repo)
    return ctc_granite.run(huggingface_repo_dir=repo, variants=variants, corpora=["dev-other"])


def granite_transformer_tuning_variants(lexicons, binary, *, repo, lm_params):
    """18 dev-only GPU/stateless conditions; keep the original pruning and cost grid."""
    from ...data.librispeech.lm import TransformerLmParams

    if not isinstance(lm_params, TransformerLmParams):
        raise TypeError("This comparison requires one original word Transformer LM")
    if lm_params.use_kv_cache or not lm_params.use_gpu:
        raise ValueError("Provide a stateless ONNX export with use_kv_cache=False and use_gpu=True")
    variants = length_variants(
        "granite",
        lexicons,
        binary,
        repo=repo,
        lm_conditions={f"trafo-lm{scale:g}": replace(lm_params, scale=scale) for scale in (0.3, 0.5)},
        phase="sweep",
        betas=(0.0, 4.0, 8.0, 12.0),
        alphas=(0.0, 2.0),
    )
    for v in variants:
        v.descriptor += "_gpu-stateless"
        v.search_mode_params = replace(v.search_mode_params, gpu_mem_rqmt=24)
        if "_closed_" in v.descriptor:
            v.search_algorithm_params = replace(v.search_algorithm_params, unknown_word_fallback="disabled")
    return variants


def run_granite_transformer_tuning():
    """Replace CPU/KV runs by GPU full-history runs of the same checkpoint."""
    from ...data.oov_neural_lm import get_gpu_word_transformer_params
    from .recognition import ctc_granite

    preflight = run_preflight()
    repo, lexicons = setup_granite_lexicons(preflight, approaches=("closed", "B"))
    variants = granite_transformer_tuning_variants(
        lexicons,
        preflight["binary"],
        repo=repo,
        lm_params=get_gpu_word_transformer_params(lexicons.values(), TOOLS_PREFIX),
    )
    return ctc_granite.run(huggingface_repo_dir=repo, variants=variants, corpora=["dev-other"])


def granite_transformer_pruning_variants(lexicons, binary, *, repo, lm_params, include_completed=False):
    """Independent beam caps at the selected LM=0.3, beta=4, alpha=2 point.

    Keep both thresholds at 16: tighter thresholds have previously triggered
    Granite fallback degeneracy. Compare WER and RTF before threshold tuning.
    The 256/256 reference is reused with its exact original descriptor/config.
    """
    (base,) = [
        v
        for v in granite_transformer_tuning_variants({"B": lexicons["B"]}, binary, repo=repo, lm_params=lm_params)
        if v.search_algorithm_params.word_lm_params.scale == 0.3
        and v.search_algorithm_params.unknown_word_penalty == 4.0
        and v.search_algorithm_params.unknown_piece_penalty == 2.0
    ]
    stem = base.descriptor.removesuffix("_beam256_thr16_gpu-stateless")
    variants = []
    for beam in (64, 128, 256):
        for word_beam in (4, 16, 64, 256):
            if (beam, word_beam) == (256, 256):
                if include_completed:
                    variants.append(base)
                continue
            variants.append(
                replace(
                    base,
                    descriptor=f"{stem}_beam{beam}_wordbeam{word_beam}_thr16_gpu-stateless",
                    search_algorithm_params=replace(
                        base.search_algorithm_params,
                        max_beam_sizes=[beam],
                        max_word_end_beam_size=word_beam,
                    ),
                )
            )
    return variants


def run_granite_transformer_pruning_tuning():
    """Rerun the 13 dev-only pruning/control conditions without speculative LM queuing."""
    from ...data.oov_neural_lm import get_gpu_word_transformer_params
    from .recognition import ctc_granite

    preflight = run_preflight()
    # Same lexicon order and gate inputs as the completed LM sweep.
    repo, lexicons = setup_granite_lexicons(preflight, approaches=("closed", "B"))
    lm_params = get_gpu_word_transformer_params(lexicons.values(), TOOLS_PREFIX)
    variants = granite_transformer_pruning_variants(
        lexicons,
        preflight["binary"],
        repo=repo,
        lm_params=lm_params,
        include_completed=True,
    )
    variants += [
        v
        for v in granite_transformer_tuning_variants(
            {"closed": lexicons["closed"]}, preflight["binary"], repo=repo, lm_params=lm_params
        )
        if v.search_algorithm_params.word_lm_params.scale == 0.3
    ]
    # OnnxStatelessLm retains speculative histories across segments when its
    # queue is not drained. Batch size 1 bypasses that queue entirely, without
    # changing model, histories, scores or GPU/full-history inference mode.
    variants = [
        replace(
            v,
            descriptor=v.descriptor + "_lm-batch1",
            search_algorithm_params=replace(
                v.search_algorithm_params,
                word_lm_params=replace(v.search_algorithm_params.word_lm_params, max_batch_size=1),
            ),
        )
        for v in variants
    ]
    return ctc_granite.run(huggingface_repo_dir=repo, variants=variants, corpora=["dev-other"])


def retained_granite_variants(lexicons, binary, *, repo):
    """Seven completed references, preserving descriptors and job hashes."""
    from ...data.librispeech.lm import ArpaLmParams
    from .recognition import ctc_granite

    closed = length_variants(
        "granite",
        {"closed": lexicons["closed"]},
        binary,
        repo=repo,
        lm_conditions={"no-lm": None, "4gram-lm0.3": ArpaLmParams(scale=0.3)},
    )
    variants = [v for v in closed if v.search_algorithm_params.max_beam_sizes == [256]]
    fallback = {"B": lexicons["B"]}
    for v in granite_beam_tuning_variants(fallback, binary, repo=repo, include_completed=True):
        p = v.search_algorithm_params
        if (p.word_lm_params is None and p.max_beam_sizes == [256] and p.score_thresholds == [8.0]) or (
            p.word_lm_params is not None and p.max_beam_sizes == [64] and p.score_thresholds == [16.0]
        ):
            variants.append(v)
    for v in tuning_variants("granite", fallback, binary, repo=repo, include_completed=True):
        p = v.search_algorithm_params
        if p.unknown_word_penalty == (3.0 if p.word_lm_params is None else 8.0):
            variants.append(v)
    greedy = ctc_granite.default_offline_lexfree_recog_variant()
    greedy.descriptor = f"oov-{REVISION}_lexfree-greedy"
    greedy.rasr_binary_path_override = binary
    variants.append(greedy)
    return variants


def run_retained_comparisons(ctc_model):
    """Keep selected BPE/Granite references; historical sweep drivers stay opt-in."""
    from .recognition import ctc_bpe, ctc_granite

    preflight = run_preflight()
    lex = setup_bpe_baseline_lexicon(ctc_model, preflight)
    variants = limited_fallback_variants("bpe", {"closed": lex}, preflight["binary"])
    variants += [
        v
        for v in tuning_variants("bpe", {"known-excluding": lex}, preflight["binary"])
        if v.search_algorithm_params.unknown_word_penalty == 10 and v.search_algorithm_params.unknown_piece_penalty == 1
    ]
    for v in variants:
        v.descriptor += "_lexicon-baseline"
    results = ctc_bpe.run(ctc_model, variants=variants, corpora=["dev-other"])
    repo, lexicons = setup_granite_lexicons(preflight, approaches=("closed", "B"))
    variants = retained_granite_variants(lexicons, preflight["binary"], repo=repo)
    results.extend(ctc_granite.run(huggingface_repo_dir=repo, variants=variants, corpora=["dev-other"]))
    return results


def run_bpe_phase(
    model, *, alpha0_by_lm, lm_conditions, phase="comparison", kind="bpe", betas=(-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
):
    """Same-file legacy/excluding tests for standard and 1/1.5/2/3% OOV lexica."""
    from ...data.librispeech.bpe import get_bpe_vocab_file, vocab_to_bpe_size
    from ...data.librispeech.lexicon import get_bpe_bliss_lexicon
    from ...data.oov_extended_lexicon import PrepareOovLengthLexiconJob
    from .lexicon_oov import setup_oov_lexicons
    from .oov_prototype import _validated_lexicon
    from .recognition import ctc_bpe, ffnn_transducer_bpe

    if kind not in {"bpe", "transducer"}:
        raise ValueError(kind)
    module = ctc_bpe if kind == "bpe" else ffnn_transducer_bpe
    preflight = run_preflight()
    bpe_size = vocab_to_bpe_size(model.model_config.target_size - 1)
    variants = []
    for rate, source in setup_oov_lexicons().items():
        lex = get_bpe_bliss_lexicon(
            bpe_size, add_blank=True, add_sentence_end_pron=False, base_lexicon_path=source, add_oov_fallback=True
        )
        lex = PrepareOovLengthLexiconJob(
            lex,
            preflight["correctness"].out_report,
            preflight["margins"].out_report,
            preflight["scoring_balance"].out_report,
            pruning_report=preflight["pruning"].out_report,
        ).out_lexicon
        lex = _validated_lexicon(
            lex,
            get_bpe_vocab_file(bpe_size, add_blank=True),
            preflight["binary"],
            f"{PREFIX}/bpe/{rate}",
            label_aliases={"</s>": "<s>"},
            decoder_smoke_report=preflight["correctness"].out_report,
        )
        grid = length_variants(
            kind,
            {k: lex for k in ("closed", "legacy", "known-excluding")},
            preflight["binary"],
            lm_conditions=lm_conditions,
            alpha0_by_lm=alpha0_by_lm,
            phase=phase,
            betas=betas,
        )
        for variant in grid:
            variant.descriptor += f"_lexicon-{rate}"
        variants.extend(grid)
    if phase == "comparison":
        greedy = module.default_offline_lexfree_recog_variant()
        greedy.descriptor = f"oov-{REVISION}_lexfree-greedy"
        greedy.rasr_binary_path_override = preflight["binary"]
        variants.append(greedy)
    return module.run(model, variants=variants, corpora=["dev-other"])
