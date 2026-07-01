"""Resolve the paper's result tables to ``data.json`` (numbers only), wired into the Sis graph.

``build_tables(results)`` is called at the end of the main recipe.
``results`` is the ``{registered-output-name: Variable}`` dict
captured by the module-level ``reg`` wrapper during ``py()``
(sis_graph is NOT introspectable mid-py()).
For each paper table it looks up the relevant metric outputs
and emits one :class:`WriteTableDataJob`, registered under ``output/tables-data/``.

This is the *data* half of the table pipeline:
only the numbers, which genuinely depend on upstream jobs, live here.
All presentation -- captions, column headers, units, layout, col-align --
lives in authored spec files in the paper repo (``tables-spec/<name>.spec.json`` + ``<name>.caption.tex``),
and is applied by the local ``scripts/render_tables.py``.
So a caption or header tweak is a one-second local re-render, never a manager restart;
the manager only runs when an actual number changes.

Lookup is by registered output NAME (this recipe's, never stale on-disk aliases).
A "<base>-wbe.txt" output IS the scalar out_wbe Variable;
its ``.creator`` is the metric job (whose ``out_metrics`` holds the full dict incl. acc@collar).
So WBE works for every aligner; acc needs out_metrics.
"""

import sys
from sisyphus import tk
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.table_data import WriteTableDataJob

_RESULTS = {}
_CAPTURE = None  # when a list, _emit captures (name, columns, rows, source) -- preview mode
_MISSING = set()  # output names a builder referenced that are NOT registered this recipe -- always a bug
# Tables whose cells are wired ONLY in the torch-2.12 companion recipe (exp2026_05_23_grad_align_p212).
# Their preview manifests are written by THAT recipe (build_preview_external_tables),
# so the main recipe's build_preview_tables must NOT prune them
# (it never sees their Variables -> would always drop them).
_EXTERNAL_PREVIEW_TABLES = {"time-stretch", "cost"}


def _require(name):
    """Look up a registered-output Variable by name; record a missing one instead of hiding it.

    Every name a builder references MUST have been registered via ``reg`` (so it is in ``_RESULTS``).
    A registered-but-unfinished output is still present -- its Variable is just unresolved on disk --
    so a missing key is never a 'pending' cell; it is a typo / rename / unwired job.
    We collect all such names and raise once at the end of ``build_tables``,
    rather than silently returning an empty cell (the old ``_RESULTS.get`` behaviour).
    """
    if name not in _RESULTS:
        _MISSING.add(name)
        return None
    return _RESULTS[name]


# Streaming model display names carry a forced LaTeX line break
# before "(streaming)",
# so the long name wraps onto two lines;
# render_tables wraps any line-break-bearing label/cell in a makecell.
_M_EMFORMER = "Emformer\\\\(streaming)"
_M_FC_CTC = "FastConf.-CTC\\\\(streaming)"
_M_FC_RNNT = "FastConf.-RNN-T\\\\(streaming)"


def build_tables(results):
    global _RESULTS
    _RESULTS = results
    _MISSING.clear()
    _compare_table()  # ground-truth forced-mode only (separate hyp table below)
    _compare_table(with_hyp=True)  # variant: hyp-mode columns merged onto the grad rows
    _hyp_table()
    _owsm_layer_table()  # OWSM grad-align per inter-CTC emit block (side table)
    _prompt_splice_table()  # generalized prompt-splice: 3 LLMs x grad + self-attn (Buckeye-segA)
    _streaming_offset_table()  # streaming start/end signed offset (grad vs native viterbi)
    # _time_stretch_table() is built by the torch-2.12 companion recipe exp2026_05_23_grad_align_p212
    # (its wav2vec2-CTC ts cells need the compiled-scan split, torch-2.12 only); see build_time_stretch_table.
    _word_length_table()  # word-duration accuracy (signed word-width error) per model
    _char_subword_family_table()  # char vs subword, one rep model per family (grad + native), replaces matched-tok
    _wav2vec_resolution_table()  # wav2vec2-CTC grad-align vs internal time-resolution level
    _encoder_depth_table()  # grad-align vs encoder depth (Whisper-large + FastConformer side by side)
    _ablation_table()  # T3a grad-score reduction (cross-model)
    _attribution_table()  # T3b attribution method (cross-model, fast models)
    _abc_table()  # Buckeye segmentation-protocol robustness (A vs B vs C)
    _alignopts_silence_table()  # T3b-i align-opts: silence/topology axis (grad-only silence fix)
    _alignopts_energy_table()  # T3b-i' audio_energy_pow sweep x blank scheme (complements silence)
    _alignopts_dtw_table()  # T3b-ii align-opts: apply-log/DTW equivalence (grad vs cross-attn)
    if _MISSING:
        # Loud, aggregated failure instead of the old silent _RESULTS.get -> None empty cell.
        raise AssertionError(
            "grad_align_tables references output names not registered in this recipe -- a missing key is\n"
            "always a bug (typo / rename / unwired job), never a pending cell, since a registered-but-\n"
            "unfinished output is still in _RESULTS. Fix the name or wire the job:\n  " + "\n  ".join(sorted(_MISSING))
        )


def _cost_cell(name, metric):
    # A cell of the 4-way cost table: index the single benchmark metrics dict by "<model>|<metric>".
    src = "speedcmp-p212/cost-4way-metrics.txt"
    return {"var": _require(src), "key": f"{name}|{metric}", "src": src}


def _cost_table():
    # 4-way per-second cost breakdown (Forward / Prefix-fwd / Prefix-bwd / Backward),
    # one model per family, measured under torch 2.12 (TIMIT-test).
    # The benchmark job runs in the p212 companion recipe;
    # the table is emitted here so it rides the SAME data + preview pipeline as every other table
    # (no special-casing).
    # CTC (wav2vec2) drives split_prefix_backward -> all 4 stages;
    # AED/Sp.LLM have no lattice (prefix cols n/a).
    # (metrics-file key, display label). The p212 benchmark job emits the CTC row under the internal
    # key "CTC (wav2vec2)"; the table displays it as MMS-FA (the wav2vec2 forced-alignment model).
    models = [
        ("CTC (wav2vec2)", "CTC (MMS-FA)"),
        ("AED (whisper-large)", "AED (whisper-large)"),
        ("Sp. LLM (phi4)", "Sp. LLM (phi4)"),
    ]
    metric_of = {
        "forward": "forward_ms_per_s",
        "prefix_fwd": "prefix_fwd_ms_per_s",
        "prefix_bwd": "prefix_bwd_ms_per_s",
        "backward": "backward_ms_per_s",
    }
    columns = ["forward", "prefix_fwd", "prefix_bwd", "backward"]
    rows = [{"label": disp, "cells": {c: _cost_cell(key, metric_of[c]) for c in columns}} for key, disp in models]
    _emit("cost", columns, rows)


def build_external_tables(results):
    """Emit the p212-owned tables (time-stretch, cost) as real WriteTableDataJobs,
    from a results dict the caller owns.
    The torch-2.12 companion recipe (exp2026_05_23_grad_align_p212) wires these tables' cells
    -- the slow wav2vec2-CTC time-stretch extracts (split_prefix_backward, compiled scan),
    and the cost benchmark --
    so they are built there, NOT in build_tables();
    the main 2.7 recipe never references them.
    Preview manifests for the same set come from build_preview_external_tables
    (see _EXTERNAL_PREVIEW_TABLES)."""
    global _RESULTS
    _RESULTS = results
    _MISSING.clear()
    _time_stretch_table()
    _cost_table()
    if _MISSING:
        raise AssertionError("external tables reference unregistered outputs:\n  " + "\n  ".join(sorted(_MISSING)))


def _emit(name, columns, rows):
    # `source` records this builder (file :: function) for a provenance comment in the rendered table;
    # the per-cell `src` (the registered output name) is what actually pins each number's origin.
    source = f"{__name__.replace('.', '/')}.py :: {sys._getframe(1).f_code.co_name}"
    if _CAPTURE is not None:  # preview mode: capture the definition instead of emitting a job
        _CAPTURE.append((name, columns, rows, source))
        return
    job = WriteTableDataJob(columns=columns, rows=rows, source=source)
    tk.register_output(f"tables-data/{name}.data.json", job.out_data)


# Cell resolve-specs: var (+ optional dict key),
# plus `src` = the Sisyphus registered output name the number comes from.
# units/format are applied by the render layer;
# `src` becomes the per-row provenance comment.
def _wbe(base):
    # The "<base>-wbe.txt" output is the scalar out_wbe (raw seconds).
    name = f"{base}-wbe.txt"
    return {"var": _require(name), "src": name}


def _metrics_var(base):
    # The metrics dict behind a "<base>-wbe.txt": creator.out_metrics,
    # or out_word_metrics for the CTC forced-align / phoneme baselines
    # (which name their aggregate dict differently).
    v = _require(f"{base}-wbe.txt")
    j = getattr(v, "creator", None) if v is not None else None
    m = getattr(j, "out_metrics", None) if j is not None else None
    if m is None and j is not None:
        m = getattr(j, "out_word_metrics", None)
    return m


def _metric(base, key):
    # Value comes from the metrics dict,
    # but the registered output that pins it is the "<base>-wbe.txt".
    return {"var": _metrics_var(base), "key": key, "src": f"{base}-wbe.txt"}


def _hyp(name, key):
    out = f"hyp-align/{name}-metrics.txt"
    return {"var": _require(out), "key": key, "src": out}


# ----------------------------------------------------------------------------------------
# Streaming signed-offset: per streaming model, grad vs native viterbi,
#     SIGNED start/end boundary offset (pred - ref, ms; + = late).
#     Native chunked viterbi starts late / ends early (emission delay); grad ~0.
#     Data = start_signed_mean / end_signed_mean (align_metrics out_metrics).
# ----------------------------------------------------------------------------------------
def _streaming_offset_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def g(stem_sa, stem_ti):
        return (f"align/{stem_sa}", f"align/{stem_ti}")

    MODELS = [
        (
            _M_EMFORMER,
            [
                (
                    "Gradients",
                    *g(
                        f"emformer-rnnt-prefix-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"emformer-rnnt-prefix-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-emformer-rnnt-native-viterbi-{S}",
                    f"baseline-emformer-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            _M_FC_CTC,
            [
                (
                    "Gradients",
                    *g(
                        f"fastconformer-stream-ctc-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"fastconformer-stream-ctc-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                ("Posteriors", f"baseline-fastconformer-stream-ctc-{S}", f"baseline-fastconformer-stream-ctc-{T}"),
            ],
        ),
        (
            _M_FC_RNNT,
            [
                (
                    "Gradients",
                    *g(
                        f"fastconformer-stream-rnnt-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"fastconformer-stream-rnnt-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{S}",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "Parakeet RNN-T\\\\(offline)",
            [
                (
                    "Gradients",
                    *g(
                        f"parakeet-rnnt-1.1b-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"parakeet-rnnt-1.1b-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{S}",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{T}",
                ),
            ],
        ),
    ]
    # Architecture per row (Emformer is an RNN-T transducer, not CTC),
    # plus a full-context (offline) Parakeet RNN-T as the non-streaming reference:
    # its native viterbi sees the whole utterance, so it has no emission-delay bias.
    STYPE = {
        _M_EMFORMER: "Transd.",
        _M_FC_CTC: "CTC",
        _M_FC_RNNT: "Transd.",
        "Parakeet RNN-T\\\\(offline)": "Transd.",
    }

    columns = ["type", "model", "method", "t_start", "t_end", "s_start", "s_end"]
    # Group by architecture: all CTC first, then all transducers (double rule between the two families).
    _TYPE_ORDER = {"CTC": 0, "Transd.": 1}
    rows = []
    prev_type = None
    for model, methods in sorted(MODELS, key=lambda m: _TYPE_ORDER[STYPE[m[0]]]):
        if prev_type is not None and STYPE[model] != prev_type:
            rows.append({"hline2": True})
        prev_type = STYPE[model]
        for mlabel, sa, ti in methods:
            rows.append(
                {
                    "cells": {
                        "type": STYPE[model],
                        "model": model,
                        "method": mlabel,
                        "t_start": _metric(ti, "start_signed_mean"),
                        "t_end": _metric(ti, "end_signed_mean"),
                        "s_start": _metric(sa, "start_signed_mean"),
                        "s_end": _metric(sa, "end_signed_mean"),
                    },
                }
            )
    _emit("streaming-offset", columns, rows)


# ----------------------------------------------------------------------------------------
# Length robustness redesigned (Buckeye-segA, vocoder time-stretch, factors 0.5..3.0).
#     The same mechanism, opposite sign across the model's grad-resolution:
#     wav2vec2-CTC grad (fine grid) only degrades with stretch (robustness limit),
#     Voxtral grad (coarse projected-embedding grid) IMPROVES at mild stretch (upsamples the grid),
#     MMS-FA forced-align re-runs Viterbi on the stretched audio, so it stays flat (reference).
# ----------------------------------------------------------------------------------------
def _time_stretch_table():
    S = "buckeye-segA-5h"
    AO = "asotTrue-bs-5-en0.5-sil2.0-wordtopo"
    FACTORS = [
        ("0.5", "f05"),
        ("0.75", "f075"),
        ("1.0", "f10"),
        ("1.2", "f12"),
        ("1.5", "f15"),
        ("2.0", "f20"),
        ("3.0", "f30"),
    ]
    # Type | Model | Align method | grid (mirrors the per-model table).
    # The grid is the effective alignment resolution (encoder/audio-token rate),
    # NOT the fine log-mel leaf -- the fine-vs-coarse axis behind the opposite-sign behaviour.
    # Each model x {vocoder, resample} stretch method.
    MODELS = [
        (
            "CTC",
            "MMS-FA",
            "Gradients",
            "50",
            lambda ts, m: f"align/wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-ts{ts}{m}-{AO}",
        ),
        ("CTC", "MMS-FA", "Posteriors", "50", lambda ts, m: f"baseline-mms_fa-{S}-ts{ts}{m}"),
        (
            "Sp. LLM",
            "Voxtral",
            "Gradients",
            "12.5",
            lambda ts, m: f"align/voxtral-charlevlogmel-{S}-L2_grad-pertoken-ts{ts}{m}-{AO}",
        ),
        (
            "Sp. LLM",
            "Voxtral",
            "Self-att.",
            "12.5",
            lambda ts, m: f"align/baseline-voxtral-selfattn-{S}-ts{ts}{m}-{AO}",
        ),
    ]
    STRETCH = [("vocoder", ""), ("resample", "-resample")]
    columns = ["type", "model", "method", "fps", "stretch"] + [k for _, k in FACTORS]
    rows = []
    for typ, model, meth, fps, basefn in MODELS:
        for slabel, msfx in STRETCH:
            cells = {"type": typ, "model": model, "method": meth, "fps": fps, "stretch": slabel}
            for ts, k in FACTORS:
                # Voxtral's 30s encoder window caps usable stretch: >=2x on an 18s segment exceeds it
                # (would split into 2 encoder chunks) -> mark the cell rather than leaving it blank.
                if model == "Voxtral" and float(ts) >= 2.0:
                    cells[k] = "too long"
                    continue
                # MMS-FA grad at ts3.0 does not fit the 24h GPU wall even with the compiled-scan split
                # (3x-stretched audio, B=1) -> mark it (distinct from Voxtral's encoder-window "too long").
                # ts1.2/1.5/2.0 ARE computed via the split under the torch-2.12 companion recipe (exp..._p212).
                if model == "MMS-FA" and float(ts) >= 3.0:
                    cells[k] = "too costly"
                    continue
                cells[k] = _wbe(basefn(ts, msfx))
            rows.append({"cells": cells})
    _emit("time-stretch", columns, rows)


# ----------------------------------------------------------------------------------------
# Word-length (duration) accuracy, full breakdown (Buckeye-segA):
#     WBE (mean abs boundary error),
#     signed start/end offset (pred - ref; + = late),
#     and word-WIDTH signed error (= end - start; 0 = correct duration).
#     The CTC/forced-align posteriors bias start LATE + end EARLY
#     -> the word shrinks from both ends, so |width| ~ WBE;
#     grad shifts start+end the SAME direction (a positional lead, not a width change),
#     so width is far below WBE -> grad is more accurate on word duration.
#     TIMIT-test shows the same pattern, milder (grad's lead is largest on spontaneous Buckeye).
# ----------------------------------------------------------------------------------------
def _word_length_table():
    S = "buckeye-segA-5h"
    MODELS = [
        (
            "MMS-FA",
            [
                (
                    "Gradients",
                    f"align/wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
                ("Posteriors", f"baseline-mms_fa-{S}"),
            ],
        ),
        (
            "XLS-R (Phoneme)",
            [
                (
                    "Gradients",
                    f"align/phoneme-vitouphy-prefixfwd-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
                ("Posteriors", f"baseline-phoneme-fa-{S}-word"),
            ],
        ),
        (
            "Parakeet CTC",
            [
                (
                    "Gradients",
                    f"align/parakeet-ctc-1.1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
                ("Posteriors", f"baseline-parakeet-ctc-1.1b-{S}"),
            ],
        ),
        (
            "OWSM-CTC",
            [
                (
                    # block 6 (best emit block), matching the per-model + owsm-per-layer tables.
                    "Gradients",
                    f"align/owsm-ctc-v4-1b-lyr6-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
                ("Posteriors", f"baseline-owsm-ctc-v4-1b-lyr6-{S}"),
            ],
        ),
        (
            "Whisper-base",
            [
                (
                    "Gradients",
                    f"align/whisper-base-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
                ("Cross-att.", f"align/baseline-whisper-base-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil2.0-wordtopo"),
            ],
        ),
    ]

    # Type | Model | Align method (mirrors the per-model table); double rule between the CTC and AED families.
    WTYPE = {
        "MMS-FA": "CTC",
        "XLS-R (Phoneme)": "CTC",
        "Parakeet CTC": "CTC",
        "OWSM-CTC": "CTC",
        "Whisper-base": "AED",
    }
    columns = ["type", "model", "method", "wbe", "start", "end", "width"]
    rows = []
    prev_type = None
    for model, methods in MODELS:
        if prev_type is not None and WTYPE[model] != prev_type:
            rows.append({"hline2": True})
        prev_type = WTYPE[model]
        for mlabel, base in methods:
            rows.append(
                {
                    "cells": {
                        "type": WTYPE[model],
                        "model": model,
                        "method": mlabel,
                        "wbe": _metric(base, "wbe"),
                        "start": _metric(base, "start_signed_mean"),
                        "end": _metric(base, "end_signed_mean"),
                        "width": _metric(base, "width_signed_err"),
                    },
                }
            )
    _emit("word-length", columns, rows)


# ----------------------------------------------------------------------------------------
# Char vs subword, ONE representative model per family (CTC / transducer / AED / Sp. LLM), each shown
#     with its gradient alignment AND its model-native alternative (CTC posteriors, transducer
#     native-Viterbi, cross-attention, self-attention). Fixed DP (Buckeye-segA, en0.5-sil2.0,
#     word-topology); only the tokenization differs between the Char and Subword column groups.
#     Same conventions as the per-model-methods table (multirow Type/Name, "Gradients"/"Posteriors").
# ----------------------------------------------------------------------------------------
def _char_subword_family_table():
    S = "buckeye-segA-5h"
    SFX = "asotTrue-bs-5-en0.5-sil2.0-wordtopo"

    def gr(stem):  # gradient / attention align (WordAlign-based): "align/<stem>-<SFX>"
        return f"align/{stem}-{SFX}"

    # (type, model, method, char base, subword base). Grad / cross-attn / self-attn rows are WordAlign
    # outputs (gr(...)); the CTC/transducer posteriors rows are the model's forced-align baseline.
    ROWS = [
        (
            "CTC",
            "Parakeet CTC",
            "Gradients",
            gr(f"parakeet-ctc-1.1b-prefixfwd-char-abl-{S}-L2_grad-pertoken"),
            gr(f"parakeet-ctc-1.1b-prefixfwd-abl-{S}-L2_grad-pertoken"),
        ),
        (
            "CTC",
            "Parakeet CTC",
            "Posteriors",
            f"baseline-parakeet-ctc-1.1b-char-{S}",
            f"baseline-parakeet-ctc-1.1b-{S}",
        ),
        (
            "Transd.",
            "Parakeet RNN-T",
            "Gradients",
            gr(f"parakeet-rnnt-1.1b-logmel-char-abl-{S}-L2_grad-pertoken"),
            gr(f"parakeet-rnnt-1.1b-logmel-abl-{S}-L2_grad-pertoken"),
        ),
        (
            "Transd.",
            "Parakeet RNN-T",
            "Posteriors",
            f"baseline-parakeet-rnnt-1.1b-char-native-viterbi-{S}",
            f"baseline-parakeet-rnnt-1.1b-native-viterbi-{S}",
        ),
        (
            "AED",
            "Whisper-large-v3",
            "Gradients",
            gr(f"whisper-large-v3-logmel-{S}-L2_grad-pertoken-charlev-spc"),
            gr(f"whisper-large-v3-logmel-{S}-L2_grad-pertoken-subword"),
        ),
        (
            "AED",
            "Whisper-large-v3",
            "Cross-att.",
            gr(f"baseline-whisper-large-v3-crossattn-charlev-{S}"),
            gr(f"baseline-whisper-large-v3-crossattn-auto-{S}"),
        ),
        (
            "Sp. LLM",
            "Voxtral",
            "Gradients",
            gr(f"voxtral-charlevlogmel-{S}-L2_grad-pertoken"),
            gr(f"voxtral-logmel-{S}-L2_grad-pertoken-subword"),
        ),
        (
            "Sp. LLM",
            "Voxtral",
            "Self-att.",
            gr(f"baseline-voxtral-char-selfattn-{S}"),
            gr(f"baseline-voxtral-selfattn-{S}"),
        ),
    ]
    columns = ["type", "model", "method", "c_wbe", "c_a50", "s_wbe", "s_a50"]
    rows = []
    prev = None
    for typ, model, method, cbase, sbase in ROWS:
        if prev is not None and typ != prev:
            rows.append({"hline2": True})  # double rule between families
        prev = typ
        rows.append(
            {
                "cells": {
                    "type": typ,
                    "model": model,
                    "method": method,
                    "c_wbe": _wbe(cbase),
                    "c_a50": _metric(cbase, "acc_50ms"),
                    "s_wbe": _wbe(sbase),
                    "s_a50": _metric(sbase, "acc_50ms"),
                },
            }
        )
    _emit("char-vs-subword-family", columns, rows)


# ----------------------------------------------------------------------------------------
# wav2vec2-CTC time-resolution sweep: which internal level the gradient is taken at (Buckeye-segA).
#     The 7-conv feature encoder (conv0-5), the feature-projection LayerNorm/Linear, and the raw
#     waveform at several pooling rates -- an internally consistent within-model resolution comparison
#     (all plain L2 grad, same en0.5-sil1.0 word-topology DP), with ms/frame at 16 kHz for context.
# ----------------------------------------------------------------------------------------
def _wav2vec_resolution_table():
    S = "buckeye-segA-5h"
    # headline decoder (prefix_fwd score + sil2.0 word-topology): feat-proj-out row == per-model MMS-FA grad.
    SFX = "asotTrue-bs-5-en0.5-sil2.0-wordtopo"
    # (tag, label, ms/frame at 16 kHz): conv strides 5/10/20/40/80/160 -> 0.31..10 ms; feat-proj and
    # the default raw pool (320) = 20 ms; raw pools 80/16/1 = 5/1/0.0625 ms (sample-level).
    # (tag, label, ms/frame at 16 kHz, grid Hz = 16000/stride): the two are reciprocals, shown side by
    # side. conv strides 5/10/20/40/80/160; feat-proj + default raw pool 320 = 20 ms / 50 Hz; raw pools
    # 80/16/1 = 5/1/0.0625 ms.
    LEVELS = [
        ("conv0", "Conv 0", "0.31", "3200"),
        ("conv1", "Conv 1", "0.62", "1600"),
        ("conv2", "Conv 2", "1.25", "800"),
        ("conv3", "Conv 3", "2.5", "400"),
        ("conv4", "Conv 4", "5", "200"),
        ("conv5", "Conv 5", "10", "100"),
        ("fproj_ln", "Feat-proj LayerNorm", "20", "50"),
        ("fproj_out", "Feat-proj Linear", "20", "50"),
        ("rawwav", "Raw waveform, pool 320", "20", "50"),
        ("rawwav-pool80", "Raw waveform, pool 80", "5", "200"),
        ("rawwav-pool16", "Raw waveform, pool 16", "1", "1000"),
        ("rawwav-pool1", "Raw waveform, pool 1", "0.0625", "16000"),
    ]
    columns = ["level", "ms_frame", "grid", "wbe", "a50"]
    rows = [
        {
            "cells": {
                "level": label,
                "ms_frame": ms,
                "grid": hz,
                "wbe": _wbe(f"align/wav2vec2ctc-{tag}-{S}-L2_grad-pertoken-{SFX}"),
                "a50": _metric(f"align/wav2vec2ctc-{tag}-{S}-L2_grad-pertoken-{SFX}", "acc_50ms"),
            },
        }
        for tag, label, ms, hz in LEVELS
    ]
    _emit("wav2vec-resolution", columns, rows)


# ----------------------------------------------------------------------------------------
# Encoder-depth grad sweep (Buckeye-segA): grad-align WBE as a function of WHERE in the network the
#     gradient is taken -- log-mel input, encoder input, encoder layers (1/4, 1/2, 3/4), and the final
#     encoder output (just before the decoder / CTC linear). Whisper-large (char) and FastConformer-CTC
#     (subword) side by side. The CTC encoder output should approach the CTC-posteriors forced-align.
# ----------------------------------------------------------------------------------------
def _encoder_depth_table():
    S = "buckeye-segA-5h"
    # headline decoder (sil2.0 word-topology): the log-mel-input row equals the per-model grad.
    SFX = "asotTrue-bs-5-en0.5-sil2.0-wordtopo"

    def cell(pre, tag):
        stem = f"align/{pre}-{tag}-encdepth-{S}-L2_grad-pertoken-{SFX}"
        return _wbe(stem), _metric(stem, "acc_50ms")

    wp = "whisper-large-v3-charlev-spc"
    fc = "fastconformer-stream-ctc"
    # (level label, whisper depth tag, fastconformer depth tag)
    # Frame-grid rate (Hz) of the representation the gradient is taken w.r.t.: log-mel input is 100 Hz
    # (10 ms) for both; Whisper's encoder is 50 Hz (20 ms), FastConformer's is 12.5 Hz (80 ms, 8x
    # subsampling). Shows the grid coarsens with depth yet WBE can still improve -- it is the
    # representation, not the grid resolution, that carries the alignment.
    ROWS = [
        # (level label, whisper depth tag, fastconformer depth tag, whisper grid Hz, fastconformer grid Hz)
        ("Log-mel input", "logmel", "logmel", 100, 100),
        ("Encoder input", "encin", "encin", 50, 12.5),
        ("Encoder 1/4", "encL8", "encL4", 50, 12.5),
        ("Encoder 1/2", "encL16", "encL9", 50, 12.5),
        ("Encoder 3/4", "encL24", "encL13", 50, 12.5),
        ("Encoder output", "encout", "encout", 50, 12.5),
    ]
    columns = ["level", "w_grid", "w_wbe", "w_a50", "fc_grid", "fc_wbe", "fc_a50"]
    rows = []
    for label, w_tag, fc_tag, w_hz, fc_hz in ROWS:
        w_wbe, w_a50 = cell(wp, w_tag)
        fc_wbe, fc_a50 = cell(fc, fc_tag)
        rows.append(
            {
                "cells": {
                    "level": label,
                    "w_grid": w_hz,
                    "w_wbe": w_wbe,
                    "w_a50": w_a50,
                    "fc_grid": fc_hz,
                    "fc_wbe": fc_wbe,
                    "fc_a50": fc_a50,
                }
            }
        )
    _emit("encoder-depth", columns, rows)


# ----------------------------------------------------------------------------------------
# T3a grad-score ablation (cross-model, Buckeye-segA, word-topology fixed).
#     Two orthogonal axes:
#     the feature-axis reduction (L0.5 / L1 / L2 / dot = signed sum)
#     and mult_grad_by_inputs (plain gradient vs gradient x input, the "x input" group).
#     A few ms apart -> neither choice is critical; the saliency signal, not the norm, carries it.
#     The signed "dot" scores align with apply_log off (a signed score has no meaningful log).
# ----------------------------------------------------------------------------------------
def _ablation_table():
    S = "buckeye-segA-5h"
    SFX = "asotTrue-bs-5-en0.5-sil2.0-wordtopo"
    MODELS = [
        ("CTC", "MMS-FA", "wav2vec2ctc-fproj_out-prefixfwd"),
        ("CTC", "Parakeet CTC", "parakeet-ctc-1.1b-prefixfwd"),
        ("Transd.", "Parakeet RNN-T", "parakeet-rnnt-1.1b-logmel"),
        ("Transd.", _M_EMFORMER, "emformer-rnnt-prefix-logmel"),
        ("AED", "Whisper-base", "whisper-base-logmel-charlev-spc"),
        ("Sp. LLM", "Voxtral", "voxtral-charlevlogmel"),
        ("Sp. LLM", "Phi-4-MM", "phi4mm-charlev-spc"),
        ("Sp. LLM", "Canary-Qwen", "canary-qwen-charlev-spc-logmel-st15"),
    ]
    # Signed "dot"/"dot_e": abs-mean norm with an eps floor + clip, with a constant blank in place of
    # the energy-silence blank (that blank log()s the score -> undefined when signed), on a plain CTC
    # topology (word-topology works on the signed score but aligns it worse, so the signed columns keep CTC).
    DOT_SFX = "nsabsmeanS-nse0.05-cs1e-05_None-asotTrue-bs-6"
    # (column key, align suffix). Plain-gradient group then the gradient-x-input ("_e") group.
    COLS = [
        ("L0.5_grad", SFX),
        ("L1_grad", SFX),
        ("L2_grad", SFX),
        ("dot_grad", DOT_SFX),
        ("L2_e_grad", SFX),
        ("dot_e_grad", DOT_SFX),
    ]
    columns = ["type", "model"] + [c for c, _ in COLS]
    # Every cell points at its WBE;
    # the renderer flags WBE > broken_above as "broken",
    # so a residual degenerate (or a not-yet-finished re-align) is never shown as a real number.
    rows = [
        {
            "cells": {
                "type": typ,
                "model": mlabel,
                **{c: _wbe(f"align/{mpre}-abl-{S}-{c}-pertoken-{sfx}") for c, sfx in COLS},
            },
        }
        for typ, mlabel, mpre in MODELS
    ]
    _emit("grad-score-ablation", columns, rows)


# ----------------------------------------------------------------------------------------
# T3b attribution-method ablation (cross-model, Buckeye-segA, word-topology fixed):
#     plain L2 input-grad vs SmoothGrad / VarGrad / Integrated-Gradients / Expected-Gradients,
#     on the two fast models.
#     The single-pass gradient is already a good attribution
#     -> the multi-pass methods are not worth it
#     (the slow families are omitted for the same cost reason).
# ----------------------------------------------------------------------------------------
def _attribution_table():
    # On a 0.25h Buckeye-segA subsample (~112 seqs), NOT the full set: the multi-pass methods x the full
    # 2234-seq set exceed the 24h walltime and never complete. See the caption note.
    S = "buckeye-segA-sub025"
    SFX = "asotTrue-bs-5-en0.5-sil2.0-wordtopo"
    MODELS = [
        ("MMS-FA", "wav2vec2ctc-fproj_out-prefixfwd"),
        ("Whisper-base", "whisper-base-logmel-charlev-spc"),
    ]
    METHODS = [
        "L2_grad",
        "L2-smoothgrad-std0.01-n8",
        "L2-vargrad-std0.02-n16",
        "L2-integrated-grad-ig16",
        "L2-expected-grad-eg16-bstd0.05",
    ]
    columns = list(METHODS)
    rows = [
        {
            "label": mlabel,
            "cells": {mk: _wbe(f"align/{mpre}-abl-{S}-{mk}-pertoken-{SFX}") for mk in METHODS},
        }
        for mlabel, mpre in MODELS
    ]
    _emit("attribution", columns, rows)


# ----------------------------------------------------------------------------------------
# Buckeye segmentation-protocol robustness (A vs B vs C, 5h each):
#     does the resegmentation protocol shift grad-align / forced-align WBE?
#     A = resegment-gap 1s + split @18s (the headline Buckeye); B = split only; C = resegment + drop @18s.
#     Representative AED / CTC / Transducer grad rows + the forced-align references.
#     The speech-LLMs are segA-only (each takes ~hours/segment), so they are not in this robustness table.
# ----------------------------------------------------------------------------------------
def _abc_table():
    AO = "asotTrue-bs-5-en0.5-sil2.0-wordtopo"
    SEGS = ["segA", "segB", "segC"]
    # (type, model, method, name template over {seg}, is_grad). Same taxonomy AND method labels as the
    # per-model table: each model shows its gradient align and its own native aligner (CTC forced-align =
    # "Posteriors", Whisper = "Cross-att."); typed by architecture; MFA = GMM-HMM reference ("Likelihoods").
    ROWS = [
        ("GMM-HMM", "MFA", "Likelihoods", "baseline-mfa-buckeye-{seg}-5h", False),
        ("CTC", "MMS-FA", "Gradients", "wav2vec2ctc-fproj_out-prefixfwd-buckeye-{seg}-5h-L2_grad-pertoken", True),
        ("CTC", "MMS-FA", "Posteriors", "baseline-mms_fa-buckeye-{seg}-5h", False),
        ("CTC", "Parakeet CTC", "Gradients", "parakeet-ctc-1.1b-prefixfwd-buckeye-{seg}-5h-L2_grad-pertoken", True),
        ("CTC", "Parakeet CTC", "Posteriors", "baseline-parakeet-ctc-1.1b-buckeye-{seg}-5h", False),
        ("Transd.", "Parakeet RNN-T", "Gradients", "parakeet-rnnt-1.1b-logmel-buckeye-{seg}-5h-L2_grad-pertoken", True),
        (
            "Transd.",
            "Parakeet RNN-T",
            "Posteriors",
            "baseline-parakeet-rnnt-1.1b-native-viterbi-buckeye-{seg}-5h",
            False,
        ),
        ("AED", "Whisper-base", "Gradients", "whisper-base-logmel-buckeye-{seg}-5h-L2_grad-pertoken-charlev-spc", True),
        ("AED", "Whisper-base", "Cross-att.", "baseline-whisper-base-crossattn-auto-buckeye-{seg}-5h", True),
    ]
    columns = ["type", "model", "method"] + SEGS
    rows = []
    prev = None
    for typ, model, meth, tmpl, is_grad in ROWS:
        if prev is not None and typ != prev:
            rows.append({"hline2": True})
        prev = typ
        cells = {"type": typ, "model": model, "method": meth}
        for seg in SEGS:
            base = tmpl.format(seg=seg)
            cells[seg] = _wbe(f"align/{base}-{AO}" if is_grad else base)
        rows.append({"cells": cells})
    _emit("buckeye-abc", columns, rows)


# ----------------------------------------------------------------------------------------
# T3b-i align-OPTS, silence & topology (Buckeye-segA, whisper grad-char).
#     Grouped "Blank scoring" {Type | Opts} + Topology;
#     grad-only (attention has ~no mass in silence).
# ----------------------------------------------------------------------------------------
def _alignopts_silence_table():
    S = "buckeye-segA-5h"
    # Blank-scoring schemes (ctc topology), across SIGNALS:
    # grad (3 models) + cross-attn (Whisper) + self-attn (Voxtral).
    # Energy token-weighting held at en0.5; only the blank score varies,
    # so this isolates const vs energy-silence vs z-score per signal.
    # (col key, full align-name stem incl. reduction + extract variant).
    # All models use their L2 headline extract (uniform reduction across every table);
    # each speech-LLM reuses the same non-abl base extract as the per-model table,
    # so every number agrees across tables.
    GRAD = [
        ("mms-fa", f"wav2vec2ctc-fproj_out-prefixfwd-abl-{S}-L2_grad-pertoken"),
        ("whisper", f"whisper-large-v3-logmel-charlev-spc-abl-{S}-L2_grad-pertoken"),
        ("owls", f"owls-1B-180K-charlev-logmel-{S}-L2_grad-pertoken"),
        ("voxtral", f"voxtral-charlevlogmel-{S}-L2_grad-pertoken"),
        ("canary", f"canary-qwen-charlev-spc-logmel-st15-{S}-L2_grad-pertoken"),
        ("phi4", f"phi4mm-{S}-L2_grad-pertoken-charlev-spc"),
        ("emformer", f"emformer-rnnt-prefix-logmel-{S}-L2_grad-pertoken"),
    ]
    ATTN = [
        ("wh_a", "baseline-whisper-large-v3-crossattn-auto"),
        ("owls_a", "baseline-owls-1B-180K-crossattn-auto"),
        ("vx_a", "baseline-voxtral-selfattn"),
        ("cn_a", "baseline-canary-qwen-selfattn"),
        ("ph_a", "baseline-phi4mm-selfattn"),
    ]
    # Blank-scoring schemes swept around the defaults (gamma / s / kappa), so the metric is seen to
    # degrade away from the chosen values; shown on the word-topology DP (our production setting).
    SCHEMES = [
        ("constant", "$\\gamma = -3$", "asotTrue-bs-3-en0.5"),
        ("constant", "$\\gamma = -5$", "asotTrue-bs-5-en0.5"),
        ("constant", "$\\gamma = -8$", "asotTrue-bs-8-en0.5"),
        ("energy", "$s = 0.5$", "asotTrue-bs-5-en0.5-sil0.5"),
        ("energy", "$s = 1$", "asotTrue-bs-5-en0.5-sil1.0"),
        ("energy", "$s = 2$", "asotTrue-bs-5-en0.5-sil2.0"),
        ("energy", "$s = 3$", "asotTrue-bs-5-en0.5-sil3.0"),
        ("z-score", "$\\kappa = 0.5$", "asotTrue-bs-5-en0.5-zsk0.5"),
        ("z-score", "$\\kappa = 1$", "asotTrue-bs-5-en0.5-zsk1.0"),
        ("z-score", "$\\kappa = 2$", "asotTrue-bs-5-en0.5-zsk2.0"),
    ]
    # Word-topology only (blank only between words, our production DP); the CTC-topology variant is dropped.
    columns = ["type", "opts"] + [k for k, _ in GRAD] + [k for k, _ in ATTN]
    rows = []
    for typ, opts, tail in SCHEMES:
        cells = {"type": typ, "opts": opts}
        for k, stem in GRAD:
            cells[k] = _wbe(f"align/{stem}-{tail}-wordtopo")
        for k, mpre in ATTN:
            cells[k] = _wbe(f"align/{mpre}-{S}-{tail}-wordtopo")
        rows.append({"cells": cells})
    _emit("alignopts-silence", columns, rows)


# ----------------------------------------------------------------------------------------
# T3b-ii align-OPTS, apply-log / DTW equivalence
#     (Buckeye-segA, whisper grad-char vs cross-attn).
#     Boolean option columns first (checkmark = on), then a short note, then the numbers.
#     Blank state off = a monotonic DTW;
#     additionally log off = the exact OpenAI Whisper timestamp DTW (bottom row),
#     where grad and cross-attn converge.
# ----------------------------------------------------------------------------------------
def _alignopts_dtw_table():
    # Per-difference ablation Whisper find_alignment <-> our aligner, BOTH directions, + a grad block.
    # Cross-attn rows (whisper-base, Buckeye-segA-5h) from WhisperDtwAblationJob (genuine openai-whisper attn;
    # the faithful row reproduces find_alignment exactly, verified).
    # Grad rows (whisper-char, Buckeye-segA) from the production grad aligner
    # -- does true-DTW help grad as it does cross-attn?
    # mono: checkmark = our monotonic DP (no vertical step); cross = whisper DTW (vertical allowed).
    # silence: word (ours full: blank only between words) / none (DTW has no blank states).
    # (label, src=(kind,key), heads, z-norm, median-filter, log, mono, silence, energy)
    ck, cx, na = "\\checkmark", "$\\times$", "--"
    WH, OU, B1 = "Whisper", "ours", "1-best"
    R = "\\ $\\hookrightarrow$ "
    # softmax = apply_softmax_over_time (the Aligner's softmax-over-time score transform).
    # x = off, NOT na: it is a real setting that would change the result.
    # The whisper-DTW rows have it OFF (x) -- with log on they use log-softmax instead (the log column).
    # It is OFF for faithful+mono/sil too (z-norm yields a DTW cost, not a softmax-able distribution),
    # and ON for the other our-DP rows + grad.
    # All rows read boundaries off the same way (span: each word = its own first-token start .. last-token
    # end, max+1). Whisper's native jump read-off (word end = next word's onset) is a boundary-definition
    # bug -- not used here; every row uses the WordAlign span read-off.
    # (label, src=(kind,key), heads, z-norm, median-filter, log, mono, silence, energy, softmax)
    ROWS = [
        ("Cross-attn: Whisper (faithful)", ("xa", "faithful"), WH, ck, ck, cx, cx, "none", cx, cx),
        (R + "$+$ energy", ("xa", "faithful_energy"), WH, ck, ck, cx, cx, "none", ck, cx),
        (R + "$+$ mono DP", ("xa", "faithful_mono"), WH, ck, ck, cx, ck, "none", cx, cx),
        (R + "\\quad $+$ silence", ("xa", "faithful_silence"), WH, ck, ck, cx, ck, "word", cx, cx),
        (R + "no median-filter", ("xa", "nomedfilt"), WH, ck, cx, cx, cx, "none", cx, cx),
        (R + "no z-norm", ("xa", "noznorm"), WH, cx, ck, cx, cx, "none", cx, cx),
        (R + "no z-norm $+$ log", ("xa", "noznorm_log"), WH, cx, ck, ck, cx, "none", cx, cx),
        (R + "\\quad $+$ mono DP", ("xa", "noznorm_log_mono"), WH, cx, ck, ck, ck, "none", cx, ck),
        (R + "\\qquad $+$ silence", ("xa", "noznorm_log_mono_sil"), WH, cx, ck, ck, ck, "word", cx, ck),
        (R + "our heads", ("xa", "ourheads"), OU, ck, ck, cx, cx, "none", cx, cx),
        (R + "single best head", ("xa", "best1head"), B1, ck, ck, cx, cx, "none", cx, cx),
        ("Cross-attn: ours (full)", ("xa", "ours_full"), OU, cx, cx, ck, ck, "word", ck, ck),
        (R + "DTW DP", ("xa", "ours_dtw"), OU, cx, cx, ck, cx, "none", ck, cx),
        (R + "DTW DP, no energy", ("xa", "ours_dtw_noen"), OU, cx, cx, ck, cx, "none", cx, cx),
        (R + "no silence", ("xa", "ours_none"), OU, cx, cx, ck, ck, "none", ck, ck),
        ("Grad: ours (full)", ("grad", "en0.5-sil2.0-wordtopo"), na, na, na, ck, ck, "word", ck, ck),
        (R + "DTW DP, word-topo", ("grad", "en0.5-sil2.0-dtwword"), na, na, na, ck, cx, "word", ck, ck),
        (R + "DTW DP", ("grad", "en0.5-truedtw"), na, na, na, ck, cx, "none", ck, ck),
        (R + "DTW DP, no energy", ("grad", "en0.0-truedtw"), na, na, na, ck, cx, "none", cx, ck),
    ]
    GP = "align/whisper-base-logmel-buckeye-segA-5h-L2_grad-pertoken-charlev-spc-asotTrue-bs-5"
    columns = ["row", "heads", "znorm", "medfilt", "log", "softmax", "mono", "silence", "energy", "wbe"]
    rows = []
    for label, src, heads, znorm, mf, log, mono, sil, en, sm in ROWS:
        wbe = _wbe(f"dtw-abl/whisper-base-buckeye-segA-5h-{src[1]}") if src[0] == "xa" else _wbe(f"{GP}-{src[1]}")
        rows.append(
            {
                "cells": {
                    "row": label,
                    "heads": heads,
                    "znorm": znorm,
                    "medfilt": mf,
                    "log": log,
                    "softmax": sm,
                    "mono": mono,
                    "silence": sil,
                    "energy": en,
                    "wbe": wbe,
                }
            }
        )
    _emit("alignopts-dtw", columns, rows)


def _alignopts_energy_table():
    # audio_energy_pow (token-weighting exponent) swept at three blank schemes, word-topology DP, Buckeye-segA.
    # Complements alignopts-silence (which sweeps the blank at fixed en0.5):
    # here the blank scheme is fixed per row-group and the energy exponent varies down the rows,
    # so en0.5 can be judged and its interplay with the blank scheme is visible.
    S = "buckeye-segA-5h"
    GRAD = [
        ("mms-fa", f"wav2vec2ctc-fproj_out-prefixfwd-abl-{S}-L2_grad-pertoken"),
        ("whisper", f"whisper-large-v3-logmel-charlev-spc-abl-{S}-L2_grad-pertoken"),
        ("owls", f"owls-1B-180K-charlev-logmel-{S}-L2_grad-pertoken"),
        ("voxtral", f"voxtral-charlevlogmel-{S}-L2_grad-pertoken"),
        ("canary", f"canary-qwen-charlev-spc-logmel-st15-{S}-L2_grad-pertoken"),
        ("phi4", f"phi4mm-{S}-L2_grad-pertoken-charlev-spc"),
    ]
    ATTN = [
        ("wh_a", "baseline-whisper-large-v3-crossattn-auto"),
        ("owls_a", "baseline-owls-1B-180K-crossattn-auto"),
        ("vx_a", "baseline-voxtral-selfattn"),
        ("cn_a", "baseline-canary-qwen-selfattn"),
        ("ph_a", "baseline-phi4mm-selfattn"),
    ]
    # (blank-scheme label, name infix); the energy exponent varies down each group.
    SCHEMES = [
        ("constant", ""),
        ("energy $s = 1$", "-sil1.0"),
        ("z-score $\\kappa = 1$", "-zsk1.0"),
    ]
    ENS = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    columns = ["type", "en"] + [k for k, _ in GRAD] + [k for k, _ in ATTN]
    rows = []
    prev = None
    for typ, infix in SCHEMES:
        if prev is not None:
            rows.append({"hline2": True})
        prev = typ
        for en in ENS:
            tail = f"asotTrue-bs-5-en{en}{infix}-wordtopo"
            cells = {"type": typ, "en": en}
            for k, stem in GRAD:
                cells[k] = _wbe(f"align/{stem}-{tail}")
            for k, mpre in ATTN:
                cells[k] = _wbe(f"align/{mpre}-{S}-{tail}")
            rows.append({"cells": cells})
    _emit("alignopts-energy", columns, rows)


# ----------------------------------------------------------------------------------------
# T1: per-model grad-align vs the model's own alternative aligner (same model, different method).
#     Rows grouped by model; columns = align method + per-dataset metrics (TIMIT, Buckeye).
# ----------------------------------------------------------------------------------------
def _compare_table(with_hyp=False):
    S, T = "buckeye-segA-5h", "timit-test"

    def g(stem_sa, stem_ti):  # grad-align align/ output bases (segA, TIMIT)
        return (f"align/{stem_sa}", f"align/{stem_ti}" if stem_ti else None)

    # (Model, [(align-method label, segA base, TIMIT base), ...])  -- grad first, then its alternative.
    # The method label names the alignment SIGNAL:
    # grad / posteriors (CTC, transducer, GMM-HMM) / cross-attn weights / self-attn weights.
    MODELS = [
        (
            "MMS-FA",
            [
                (
                    "Gradients",
                    *g(
                        f"wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"wav2vec2ctc-fproj_out-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                ("Posteriors", f"baseline-mms_fa-{S}", f"baseline-mms_fa-{T}"),
            ],
        ),
        (
            "XLS-R (Phoneme)",
            [
                (
                    "Gradients",
                    *g(
                        f"phoneme-vitouphy-prefixfwd-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"phoneme-vitouphy-prefixfwd-{T}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-phoneme-fa-{S}-word",
                    "align/w2v-phoneme-timit-test-ctc-forced-align-word",
                ),
            ],
        ),
        (
            "Parakeet CTC",
            [
                (
                    "Gradients",
                    *g(
                        f"parakeet-ctc-1.1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"parakeet-ctc-1.1b-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                ("Posteriors", f"baseline-parakeet-ctc-1.1b-{S}", f"baseline-parakeet-ctc-1.1b-{T}"),
            ],
        ),
        (
            # OWSM: general graphemic CTC, our least-favourable model for grad-align (still runs,
            # just imprecise). Shown at its BEST emit block (6 = lowest WBE of all blocks; the final block
            # 27 is worse -- see tab:owsm-per-layer). Grad AND posteriors both from block 6, so the row is
            # internally consistent and equals the block-6 row of tab:owsm-per-layer.
            "OWSM",
            [
                (
                    "Gradients",
                    *g(
                        f"owsm-ctc-v4-1b-lyr6-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"owsm-ctc-v4-1b-lyr6-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                # posteriors = CTC forced-align of the model's emission, at the same block-6 emission.
                ("Posteriors", f"baseline-owsm-ctc-v4-1b-lyr6-{S}", f"baseline-owsm-ctc-v4-1b-lyr6-{T}"),
            ],
        ),
        (
            "Whisper-base",
            [
                (
                    "Gradients",
                    *g(
                        f"whisper-base-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"whisper-base-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Cross-att.",
                    f"align/baseline-whisper-base-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    f"align/baseline-whisper-base-crossattn-auto-{T}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
            ],
        ),
        (
            "Whisper-large-v3",
            [
                (
                    "Gradients",
                    *g(
                        f"whisper-large-v3-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"whisper-large-v3-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    # Gradients at the OPTIMAL encoder depth (encoder 3/4 = layer 24; the best of the
                    # encoder-depth sweep, tab:encoder-depth), which beats the cross-attention DTW.
                    # Buckeye-segA only for now -- no TIMIT encoder-depth extract yet (TIMIT cols empty).
                    "Gradients*",
                    f"align/whisper-large-v3-charlev-spc-encL24-encdepth-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    None,
                ),
                (
                    "Cross-att.",
                    f"align/baseline-whisper-large-v3-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    f"align/baseline-whisper-large-v3-crossattn-auto-{T}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
            ],
        ),
        (
            "CrisperWhisper",
            [
                (
                    "Gradients",
                    *g(
                        f"crisperwhisper-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"crisperwhisper-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Cross-att.",
                    f"align/baseline-crisperwhisper-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    f"align/baseline-crisperwhisper-crossattn-auto-{T}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
            ],
        ),
        (
            "OWLS-1B",
            [
                (
                    "Gradients",
                    *g(
                        f"owls-1B-180K-charlev-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"owls-1B-180K-charlev-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Cross-att.",
                    f"align/baseline-owls-1B-180K-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    f"align/baseline-owls-1B-180K-crossattn-auto-{T}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
            ],
        ),
        (
            "Parakeet",
            [
                (
                    "Gradients",
                    *g(
                        f"parakeet-rnnt-1.1b-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"parakeet-rnnt-1.1b-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{S}",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "Parakeet TDT",
            [
                (
                    "Gradients",
                    *g(
                        f"parakeet-tdt-0.6b-v2-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"parakeet-tdt-0.6b-v2-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-parakeet-tdt-0.6b-v2-native-viterbi-{S}",
                    f"baseline-parakeet-tdt-0.6b-v2-native-viterbi-{T}",
                ),
            ],
        ),
        (
            # Emformer RNN-T: the streaming transducer (vs offline Parakeet). grad resists the
            # streaming emission delay that the native forced alignment carries.
            _M_EMFORMER,
            [
                (
                    "Gradients",
                    *g(
                        f"emformer-rnnt-prefix-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"emformer-rnnt-prefix-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-emformer-rnnt-native-viterbi-{S}",
                    f"baseline-emformer-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            _M_FC_CTC,
            [
                (
                    "Gradients",
                    *g(
                        f"fastconformer-stream-ctc-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"fastconformer-stream-ctc-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-fastconformer-stream-ctc-{S}",
                    f"baseline-fastconformer-stream-ctc-{T}",
                ),
            ],
        ),
        (
            _M_FC_RNNT,
            [
                (
                    "Gradients",
                    *g(
                        f"fastconformer-stream-rnnt-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"fastconformer-stream-rnnt-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{S}",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "Voxtral",
            [
                (
                    "Gradients",
                    *g(
                        f"voxtral-charlevlogmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"voxtral-charlevlogmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Self-att.",
                    f"align/baseline-voxtral-selfattn-{S}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    f"align/baseline-voxtral-selfattn-{T}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
            ],
        ),
        (
            "Phi-4-MM",
            [
                (
                    "Gradients",
                    *g(
                        f"phi4mm-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"phi4mm-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Self-att.",
                    f"align/baseline-phi4mm-selfattn-{S}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    f"align/baseline-phi4mm-selfattn-{T}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
            ],
        ),
        (
            "Canary-Qwen",
            [
                (
                    "Gradients",
                    *g(
                        f"canary-qwen-charlev-spc-logmel-st15-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                        f"canary-qwen-charlev-spc-logmel-st15-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    ),
                ),
                (
                    "Self-att.",
                    f"align/baseline-canary-qwen-selfattn-{S}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                    f"align/baseline-canary-qwen-selfattn-{T}-asotTrue-bs-5-en0.5-sil2.0-wordtopo",
                ),
            ],
        ),
    ]
    # Dedicated-aligner ceiling (model-agnostic), shown for reference.
    REFERENCE = [
        ("MFA", [("Likelihoods", f"baseline-mfa-{S}", f"baseline-mfa-{T}")]),
    ]

    # Hyp-mode (each model aligns its OWN recognition, Buckeye only) -> grad rows only, where it exists.
    HYP = {
        "Whisper-base": f"whisper-base-charlev-{S}-grad",
        "Voxtral": f"voxtral-charlevlogmel-{S}-grad",
        "Phi-4-MM": f"phi4mm-charlev-spc-{S}-grad",
        "Canary-Qwen": f"canary-qwen-charlev-spc-logmel-st15-{S}-grad",
        "Parakeet": f"parakeet-rnnt-1.1b-logmel-{S}-grad",
        "Parakeet TDT": f"parakeet-tdt-0.6b-v2-logmel-{S}-grad",
        "Whisper-large-v3": f"whisper-large-v3-charlev-{S}-grad",
        "Parakeet CTC": f"parakeet-ctc-1.1b-{S}-grad",
        "OWSM": f"owsm-ctc-v4-1b-{S}-grad",
        _M_FC_CTC: f"fastconformer-stream-ctc-{S}-grad",
        _M_FC_RNNT: f"fastconformer-stream-rnnt-{S}-grad",
        _M_EMFORMER: f"emformer-rnnt-prefix-logmel-{S}-grad",
        "CrisperWhisper": f"crisperwhisper-charlev-{S}-grad",
        "OWLS-1B": f"owls-1B-180K-charlev-{S}-grad",
    }
    # Models with no word-level own-recognition -> hyp-mode is structurally n/a (not "unrun"):
    # MMS-FA + Phoneme emit no word boundaries;
    # MFA is a forced-aligner, not a recognizer.
    HYP_NA = {"MMS-FA", "XLS-R (Phoneme)", "MFA"}

    # Native aligner in hyp-mode (own recognition), shown on the model's ALTERNATIVE row:
    # cross-attn (whisper), self-attn (LLMs), native viterbi (transducers), CTC forced-align (CTCs).
    # whisper-base keeps its original whisper-crossattn alias; the CTC entries fill once their
    # forced-align job emits a boundary HDF (pending), so they render empty until then.
    NATIVE_HYP = {
        "Whisper-base": f"whisper-crossattn-{S}",
        "Whisper-large-v3": f"whisper-large-v3-charlev-{S}-native",
        "Voxtral": f"voxtral-charlevlogmel-{S}-native",
        "Phi-4-MM": f"phi4mm-charlev-spc-{S}-native",
        "Canary-Qwen": f"canary-qwen-charlev-spc-logmel-st15-{S}-native",
        "Parakeet": f"parakeet-rnnt-1.1b-logmel-{S}-native",
        "Parakeet TDT": f"parakeet-tdt-0.6b-v2-logmel-{S}-native",
        _M_EMFORMER: f"emformer-rnnt-prefix-logmel-{S}-native",
        _M_FC_RNNT: f"fastconformer-stream-rnnt-{S}-native",
        "Parakeet CTC": f"parakeet-ctc-1.1b-{S}-native",
        "OWSM": f"owsm-ctc-v4-1b-{S}-native",
        _M_FC_CTC: f"fastconformer-stream-ctc-{S}-native",
        "CrisperWhisper": f"crisperwhisper-charlev-{S}-native",
        "OWLS-1B": f"owls-1B-180K-charlev-{S}-native",
    }

    # Rows grouped by model family; the leftmost (row-label) column is the family Type, shown once
    # per family; Model is the second column, shown once per model; double rule between families.
    TYPE = {
        "MMS-FA": "CTC",
        "XLS-R (Phoneme)": "CTC",
        "Parakeet CTC": "CTC",
        "OWSM": "CTC",
        _M_FC_CTC: "CTC",
        "Whisper-base": "AED",
        "Whisper-large-v3": "AED",
        "CrisperWhisper": "AED",
        "OWLS-1B": "AED",
        "Parakeet": "Transd.",
        "Parakeet TDT": "Transd.",
        _M_EMFORMER: "Transd.",
        _M_FC_RNNT: "Transd.",
        "Voxtral": "Sp. LLM",
        "Phi-4-MM": "Sp. LLM",
        "Canary-Qwen": "Sp. LLM",
    }
    TYPE_ORDER = ["CTC", "Transd.", "AED", "Sp. LLM"]
    by_type = {t: [] for t in TYPE_ORDER}
    for model, methods in MODELS:
        by_type[TYPE[model]].append((model, methods))
    # Family order: GMM-HMM reference first, then CTC, Transducer, AED, Speech LLM.
    blocks = [("GMM-HMM", REFERENCE)] + [(t, by_type[t]) for t in TYPE_ORDER]

    columns = ["type", "model", "method", "t_wbe", "t_a50", "s_wbe", "s_a50"]
    if with_hyp:
        columns += ["h_mwbe", "h_f50"]

    # Display names: the Type column already shows the family, but we keep the explicit head for
    # same-family siblings (Parakeet RNN-T vs TDT, FastConformer CTC vs RNN-T) and a clearer model name,
    # and use the SAME names as the hyp / streaming tables for consistency.
    # The dict KEY stays the full name (TYPE/HYP/NATIVE_HYP are keyed by it); only the shown label changes.
    DISPLAY = {"Parakeet": "Parakeet RNN-T", "OWSM": "OWSM-CTC"}
    rows = []
    for bi, (typ, fam_models) in enumerate(blocks):
        if bi > 0:
            rows.append({"hline2": True})  # double rule between families
        for mi2, (model, methods) in enumerate(fam_models):
            for mj, (mlabel, sa, ti) in enumerate(methods):
                cells = {
                    "type": typ,
                    "model": DISPLAY.get(model, model),
                    "method": mlabel,
                    "t_wbe": _wbe(ti) if ti else None,
                    "t_a50": _metric(ti, "acc_50ms") if ti else None,
                    "s_wbe": _wbe(sa) if sa else None,
                    "s_a50": _metric(sa, "acc_50ms") if sa else None,
                }
                if with_hyp:
                    # grad hyp on the plain gradients row; the model's native-aligner hyp on its
                    # alternative row (cross-att / self-att / posteriors / native) -- each aligns the
                    # model's OWN recognition. Matched by LABEL (not row index) so the opt-depth
                    # "Gradients*" variant, which has no hyp-mode extract, stays empty.
                    if mlabel == "Gradients":
                        hn = HYP.get(model)
                    elif mlabel.endswith("*"):
                        hn = None
                    else:
                        hn = NATIVE_HYP.get(model)
                    if hn:
                        cells["h_mwbe"] = _hyp(hn, "matched_wbe")
                        cells["h_f50"] = _hyp(hn, "f1_50ms")
                    elif model in HYP_NA and not mlabel.endswith("*"):
                        # structurally no word-level own-recognition -> mark, don't leave empty
                        cells["h_mwbe"] = "n/a"
                        cells["h_f50"] = "n/a"
                rows.append({"cells": cells})
    _emit("per-model-" + ("merged" if with_hyp else "methods"), columns, rows)


# ----------------------------------------------------------------------------------------
# T6: hypothesis-mode -- each model aligns its OWN recognition. Identity-gated F1 + matched-WBE.
# ----------------------------------------------------------------------------------------
def _hyp_table():
    S = "buckeye-segA-5h"

    def _cells(name):
        return {
            "mwbe": _hyp(name, "matched_wbe"),
            "f50": _hyp(name, "f1_50ms"),
            "f100": _hyp(name, "f1_100ms"),
            "f200": _hyp(name, "f1_200ms"),
            "rm": _hyp(name, "match_rate_ref"),
        }

    # (type, model, method, hyp-metrics name). Same taxonomy/labels as the per-model table: each model
    # aligns its OWN recognition. Every recognizing model gets its gradient row plus its alternative
    # aligner (CTC/transducer posteriors, AED cross-attention, speech-LLM self-attention), so the
    # table is consistent with the per-model table. Grouped by architecture, double rule between families.
    ROWS = [
        ("CTC", "Parakeet CTC", "Gradients", f"parakeet-ctc-1.1b-{S}-grad"),
        ("CTC", "Parakeet CTC", "Posteriors", f"parakeet-ctc-1.1b-{S}-native"),
        ("CTC", "OWSM-CTC", "Gradients", f"owsm-ctc-v4-1b-{S}-grad"),
        ("CTC", "OWSM-CTC", "Posteriors", f"owsm-ctc-v4-1b-{S}-native"),
        ("CTC", _M_FC_CTC, "Gradients", f"fastconformer-stream-ctc-{S}-grad"),
        ("CTC", _M_FC_CTC, "Posteriors", f"fastconformer-stream-ctc-{S}-native"),
        ("Transd.", "Parakeet RNN-T", "Gradients", f"parakeet-rnnt-1.1b-logmel-{S}-grad"),
        ("Transd.", "Parakeet RNN-T", "Posteriors", f"parakeet-rnnt-1.1b-logmel-{S}-native"),
        ("Transd.", "Parakeet TDT", "Gradients", f"parakeet-tdt-0.6b-v2-logmel-{S}-grad"),
        ("Transd.", "Parakeet TDT", "Posteriors", f"parakeet-tdt-0.6b-v2-logmel-{S}-native"),
        ("Transd.", _M_EMFORMER, "Gradients", f"emformer-rnnt-prefix-logmel-{S}-grad"),
        ("Transd.", _M_EMFORMER, "Posteriors", f"emformer-rnnt-prefix-logmel-{S}-native"),
        ("Transd.", _M_FC_RNNT, "Gradients", f"fastconformer-stream-rnnt-{S}-grad"),
        ("Transd.", _M_FC_RNNT, "Posteriors", f"fastconformer-stream-rnnt-{S}-native"),
        ("AED", "Whisper-base", "Gradients", f"whisper-base-charlev-{S}-grad"),
        ("AED", "Whisper-base", "Cross-att.", f"whisper-crossattn-{S}"),
        ("AED", "Whisper-large-v3", "Gradients", f"whisper-large-v3-charlev-{S}-grad"),
        ("AED", "Whisper-large-v3", "Cross-att.", f"whisper-large-v3-charlev-{S}-native"),
        ("AED", "CrisperWhisper", "Gradients", f"crisperwhisper-charlev-{S}-grad"),
        ("AED", "CrisperWhisper", "Cross-att.", f"crisperwhisper-charlev-{S}-native"),
        ("AED", "CrisperWhisper", "Official", f"crisperwhisper-official-{S}"),
        ("AED", "OWLS-1B", "Gradients", f"owls-1B-180K-charlev-{S}-grad"),
        ("AED", "OWLS-1B", "Cross-att.", f"owls-1B-180K-charlev-{S}-native"),
        ("Sp. LLM", "Voxtral", "Gradients", f"voxtral-charlevlogmel-{S}-grad"),
        ("Sp. LLM", "Voxtral", "Self-att.", f"voxtral-charlevlogmel-{S}-native"),
        ("Sp. LLM", "Phi-4-MM", "Gradients", f"phi4mm-charlev-spc-{S}-grad"),
        ("Sp. LLM", "Phi-4-MM", "Self-att.", f"phi4mm-charlev-spc-{S}-native"),
        ("Sp. LLM", "Canary-Qwen", "Gradients", f"canary-qwen-charlev-spc-logmel-st15-{S}-grad"),
        ("Sp. LLM", "Canary-Qwen", "Self-att.", f"canary-qwen-charlev-spc-logmel-st15-{S}-native"),
    ]
    columns = ["type", "model", "method", "mwbe", "f50", "f100", "f200", "rm"]
    rows = []
    prev = None
    for typ, model, meth, name in ROWS:
        if prev is not None and typ != prev:
            rows.append({"hline2": True})
        prev = typ
        rows.append({"cells": {"type": typ, "model": model, "method": meth, **_cells(name)}})
    _emit("hyp", columns, rows)


# ----------------------------------------------------------------------------------------
# T7: OWSM grad-align emitted from each inter-CTC self-conditioning block (side table).
# ----------------------------------------------------------------------------------------
def _owsm_layer_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def _tag(layer):
        # blocks 6/12/15/21 carry an `lyr{N}-` tag; block 27 is the default final-emission base.
        return "" if layer == 27 else f"lyr{layer}-"

    def gbase(layer, ds):  # grad-align output base
        # layer 27 (final emission) is the headline prefix_fwd extract (layer=None);
        # the inter-CTC blocks use their own prefix_fwd configs.
        # The per-model table shows OWSM at its BEST block, which is the lyr6 row here.
        if layer == 27:
            return f"align/owsm-ctc-v4-1b-prefixfwd-{ds}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo"
        return f"align/owsm-ctc-v4-1b-lyr{layer}-{ds}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil2.0-wordtopo"

    def fbase(layer, ds):  # forced-align (posteriors) baseline base
        return f"baseline-owsm-ctc-v4-1b-{_tag(layer)}{ds}"

    columns = ["t_grad", "t_fa", "s_grad", "s_fa"]
    rows = []
    for layer in [6, 12, 15, 21, 27]:
        rows.append(
            {
                "label": f"{layer} (final)" if layer == 27 else f"{layer}",
                "cells": {
                    "t_grad": _wbe(gbase(layer, T)),
                    "t_fa": _wbe(fbase(layer, T)),
                    "s_grad": _wbe(gbase(layer, S)),
                    "s_fa": _wbe(fbase(layer, S)),
                },
            }
        )
    _emit("owsm-per-layer", columns, rows)


# ----------------------------------------------------------------------------------------
# T8b: prompt-splice sensitivity generalized across the speech LLMs, grad AND self-attn.
#     Rows = the 4 spliced instructions;
#     columns grouped per model into Grad / Self-att. (char-level, Buckeye-segA).
#     Shows prompt-robustness holds across models and across the grad-vs-attention signal.
# ----------------------------------------------------------------------------------------
def _prompt_splice_table():
    S = "buckeye-segA-5h"
    AO_GRAD = "asotTrue-bs-5-en0.5-sil2.0-wordtopo"
    AO_ATTN = "asotTrue-bs-5-en0.5-sil2.0-wordtopo"
    MODELS = [("phi4mm", "Phi-4-MM"), ("voxtral", "Voxtral"), ("canary-qwen", "Canary-Qwen")]
    # (tag, row label, the actual spliced instruction text).
    # The "len" column = the character count of the text, computed dynamically below (not hardcoded),
    # so it tracks any edit to the instruction.
    PROMPTS = [
        ("default", "neutral (into text)", "Transcribe the audio clip into text."),
        ("verbatim", "verbatim, as spoken", "Please transcribe the audio verbatim, exactly as spoken."),
        ("word", "exactly, word for word", "Transcribe the following exactly and accurately, word for word:"),
        (
            "char",
            "at the character level",
            "Transcribe the audio at the character level, spelling out each word letter by letter.",
        ),
    ]
    columns = ["len"]
    for mk, _ in MODELS:
        columns += [f"{mk}_g", f"{mk}_a"]
    rows = []
    # Official ASR prompt per model FIRST, as the reference row (Phi-4: its default instruction;
    # Canary: "Transcribe the following:"; Voxtral: native transcription request, no free-text prompt
    # slot). The instruction text is model-specific, so its length varies -> "(var)".
    off_cells = {"len": "(var)"}
    for mk, _ in MODELS:
        off_cells[f"{mk}_g"] = _wbe(f"align/{mk}-promptsplice-official-char-{S}-grad-pertoken-{AO_GRAD}")
        off_cells[f"{mk}_a"] = _wbe(f"align/baseline-{mk}-promptsplice-official-selfattn-{S}-{AO_ATTN}")
    rows.append({"label": "official ASR prompt", "cells": off_cells})
    rows.append({"hline2": True})
    for tag, label, prompt_text in PROMPTS:
        cells = {"len": len(prompt_text)}
        for mk, _ in MODELS:
            cells[f"{mk}_g"] = _wbe(f"align/{mk}-promptsplice-{tag}-char-{S}-grad-pertoken-{AO_GRAD}")
            cells[f"{mk}_a"] = _wbe(f"align/baseline-{mk}-promptsplice-{tag}-selfattn-{S}-{AO_ATTN}")
        rows.append({"label": label, "cells": cells})
    _emit("prompt-splice", columns, rows)


def build_preview_tables(results):
    """Write per-table preview MANIFESTS to output/tables-data-preview/<name>.manifest.pkl.

    Each gzip-pickles (columns, rows, source); the rows carry the real cell Variables, so the manifest
    holds everything needed to resolve the numbers later. Written once per config load (this runs in py()).
    The numbers are NOT resolved here -- jobs/table_data.py --refresh-preview output/tables-data-preview
    re-resolves them from current disk state any time, with no manager restart
    (scripts/sync_tables_preview.sh runs that refresher before rendering), so a pending cell never blocks."""
    import os
    from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.table_data import (
        write_preview_manifest,
    )

    global _CAPTURE
    _CAPTURE = []
    try:
        build_tables(results)  # same builders, but _emit captures the defs instead of emitting jobs
        defs = list(_CAPTURE)
    finally:
        _CAPTURE = None

    out_dir = "output/tables-data-preview"
    os.makedirs(out_dir, exist_ok=True)
    # Drop snapshots of renamed/removed tables so the preview dir holds only the current set.
    current = {name for name, *_ in defs}
    for fn in os.listdir(out_dir):
        if fn.endswith(".manifest.pkl") and fn[: -len(".manifest.pkl")] not in current:
            if fn[: -len(".manifest.pkl")] in _EXTERNAL_PREVIEW_TABLES:
                continue  # owned by the p212 recipe -- leave its manifest alone
            os.remove(os.path.join(out_dir, fn))
        elif fn.endswith(".data.json") and fn[: -len(".data.json")] not in current:
            if fn[: -len(".data.json")] in _EXTERNAL_PREVIEW_TABLES:
                continue
            os.remove(os.path.join(out_dir, fn))
    for name, columns, rows, source in defs:
        write_preview_manifest(name, columns, rows, source, out_dir)


def build_preview_external_tables(results):
    """Write preview manifests for the p212-owned tables (_EXTERNAL_PREVIEW_TABLES),
    from a results dict the caller owns.
    Called from the torch-2.12 companion recipe's py() (exp2026_05_23_grad_align_p212),
    where those tables' cells (the split-backward wav2vec2 extracts) are wired.
    The manifests land in the SAME output/tables-data-preview dir as the main recipe's,
    so scripts/sync_tables.sh's --refresh-preview resolves partial progress for them too
    -- a pending split cell shows the dot glyph, never blocks the render,
    and the main recipe leaves these names alone (see _EXTERNAL_PREVIEW_TABLES)."""
    import os
    from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.table_data import (
        write_preview_manifest,
    )

    global _RESULTS, _CAPTURE
    _RESULTS = results
    _MISSING.clear()
    _CAPTURE = []
    try:
        _time_stretch_table()
        _cost_table()
        defs = list(_CAPTURE)
    finally:
        _CAPTURE = None
    if _MISSING:
        raise AssertionError(
            "external preview tables reference unregistered outputs:\n  " + "\n  ".join(sorted(_MISSING))
        )
    out_dir = "output/tables-data-preview"
    os.makedirs(out_dir, exist_ok=True)
    for name, columns, rows, source in defs:
        write_preview_manifest(name, columns, rows, source, out_dir)
