"""Auto-generated LaTeX result tables, wired into the Sis graph.

``build_tables(results)`` is called at the end of the main recipe. ``results`` is the
``{registered-output-name: Variable}`` dict captured by the module-level ``reg`` wrapper during
``py()`` (sis_graph is NOT introspectable mid-py()). For each paper table it looks up the relevant
metric outputs and emits one :class:`WriteLatexTableJob`, registered under ``output/tables/``.

Lookup is by registered output NAME (this recipe's, never stale on-disk aliases). A "<base>-wbe.txt"
output IS the scalar out_wbe Variable; its ``.creator`` is the metric job (whose ``out_metrics``
holds the full dict incl. acc@collar). So WBE works for every aligner; acc needs out_metrics.
"""

from sisyphus import tk
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.latex_table import WriteLatexTableJob

_RESULTS = {}


def build_tables(results):
    global _RESULTS
    _RESULTS = results
    _compare_table()
    _hyp_table()


def _wbe(base):
    # The "<base>-wbe.txt" output is the scalar out_wbe (seconds); -> ms.
    return {"var": _RESULTS.get(f"{base}-wbe.txt"), "mul": 1000.0, "fmt": "{:.1f}"}


def _acc(base, key):
    v = _RESULTS.get(f"{base}-wbe.txt")
    j = getattr(v, "creator", None) if v is not None else None
    m = getattr(j, "out_metrics", None) if j is not None else None
    if m is None and j is not None:
        # ForcedAlignPhonemeBaselineJob names its (same aggregate_corpus) metrics dict out_word_metrics.
        m = getattr(j, "out_word_metrics", None)
    return {"var": m, "key": key, "mul": 100.0, "fmt": "{:.1f}"}


# ----------------------------------------------------------------------------------------
# T1: per-model grad-align vs the model's own alternative aligner (same model, different method).
#     Rows grouped by model; columns = align method + per-dataset metrics (TIMIT, Buckeye).
# ----------------------------------------------------------------------------------------
def _compare_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def g(stem_sa, stem_ti):  # grad-align align/ output bases (segA, TIMIT)
        return (f"align/{stem_sa}", f"align/{stem_ti}" if stem_ti else None)

    # (Model, [(align-method label, segA base, TIMIT base), ...])  -- grad first, then its alternative.
    # The method label names the alignment SIGNAL: grad / posteriors (CTC, transducer, GMM-HMM) /
    # cross-attn weights / self-attn weights.
    MODELS = [
        (
            "Wav2Vec2-CTC",
            [
                (
                    "grad",
                    *g(
                        f"wav2vec2ctc-fproj_out-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"wav2vec2ctc-fproj_out-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                ("posteriors", f"baseline-mms_fa-{S}", f"baseline-mms_fa-{T}"),
            ],
        ),
        (
            "Phoneme-CTC",
            [
                (
                    "grad",
                    *g(
                        f"phoneme-vitouphy-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0",
                        "w2v-phoneme-timit-test-L2_grad-perphone-en0.5-sil1.0-word",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-phoneme-fa-{S}-word",
                    "align/w2v-phoneme-timit-test-ctc-forced-align-word",
                ),
            ],
        ),
        (
            "Whisper-base",
            [
                (
                    "grad",
                    *g(
                        f"whisper-base-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                        f"whisper-base-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "cross-attn weights",
                    f"align/baseline-whisper-base-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-whisper-base-crossattn-auto-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
        (
            "Whisper-large-v3",
            [
                (
                    "grad",
                    *g(
                        f"whisper-large-v3-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                        f"whisper-large-v3-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "cross-attn weights",
                    f"baseline-whisper-large-v3-crossattn-{S}",
                    f"baseline-whisper-large-v3-crossattn-{T}",
                ),
            ],
        ),
        (
            "Parakeet RNN-T",
            [
                (
                    "grad",
                    *g(
                        f"parakeet-rnnt-1.1b-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"parakeet-rnnt-1.1b-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{S}",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "Parakeet TDT",
            [
                (
                    "grad",
                    *g(
                        f"parakeet-tdt-0.6b-v2-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"parakeet-tdt-0.6b-v2-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-parakeet-tdt-0.6b-v2-native-viterbi-{S}",
                    f"baseline-parakeet-tdt-0.6b-v2-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "Voxtral",
            [
                (
                    "grad",
                    *g(
                        f"voxtral-charlevlogmel-{S}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"voxtral-charlevlogmel-{T}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "self-attn weights",
                    f"align/baseline-voxtral-selfattn-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-voxtral-selfattn-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
        (
            "Phi-4-MM",
            [
                (
                    "grad",
                    *g(
                        f"phi4mm-{S}-L2_e_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                        f"phi4mm-{T}-L2_e_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "self-attn weights",
                    f"align/baseline-phi4mm-selfattn-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-phi4mm-selfattn-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
        (
            "Canary-Qwen",
            [
                (
                    "grad",
                    *g(
                        f"canary-qwen-charlev-spc-logmel-st15-{S}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"canary-qwen-charlev-spc-logmel-st15-{T}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "self-attn weights",
                    f"align/baseline-canary-qwen-selfattn-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-canary-qwen-selfattn-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
    ]
    # Dedicated-aligner ceiling (model-agnostic), shown for reference.
    REFERENCE = [
        ("MFA (GMM-HMM)", [("posteriors", f"baseline-mfa-{S}", f"baseline-mfa-{T}")]),
    ]

    # Two-level header: TIMIT / Buckeye groups, each with WBE + acc@{50,100}.
    cols = [
        {"key": "method", "header": "Align \\\\ method"},
        {"key": "t_wbe", "header": "WBE \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "t_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "TIMIT"},
        {"key": "t_a100", "header": "acc@100 \\\\ {[\\%]}", "group": "TIMIT"},
        {"key": "s_wbe", "header": "WBE \\\\ {[ms]}", "group": "Buckeye"},
        {"key": "s_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "Buckeye"},
        {"key": "s_a100", "header": "acc@100 \\\\ {[\\%]}", "group": "Buckeye"},
    ]

    def _model_rows(groups, lead_hline):
        rows = []
        for gi, (model, methods) in enumerate(groups):
            if gi > 0 or lead_hline:
                rows.append({"label": None, "cells": {}})  # \hline between models
            for mi, (mlabel, sa, ti) in enumerate(methods):
                if mi > 0:
                    rows.append({"cline": True})  # \cline between a model's own methods
                rows.append(
                    {
                        "label": model if mi == 0 else "",
                        "cells": {
                            "method": mlabel,
                            "t_wbe": _wbe(ti) if ti else None,
                            "t_a50": _acc(ti, "acc_50ms") if ti else None,
                            "t_a100": _acc(ti, "acc_100ms") if ti else None,
                            "s_wbe": _wbe(sa) if sa else None,
                            "s_a50": _acc(sa, "acc_50ms") if sa else None,
                            "s_a100": _acc(sa, "acc_100ms") if sa else None,
                        },
                    }
                )
        return rows

    rows = _model_rows(MODELS, False) + _model_rows(REFERENCE, True)

    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=(
            "Per-model alignment quality: gradient-based alignment vs. the model's own native / "
            "attention aligner, on TIMIT-test and Buckeye (segment-A 5h subset). "
            "The attention aligners (cross-/self-attn weights) use each model's native subword "
            "tokenization -- their best setting, as the coarse attention grid cannot resolve "
            "char-level targets; grad-align and the CTC / transducer aligners use the same per-token "
            "units within each model. MFA (GMM-HMM) is a dedicated-aligner reference."
        ),
        label="tab:per-model-methods",
        label_header="Model",
        col_align="|l|l|r|r|r|r|r|r|",
    )
    tk.register_output("tables/per-model-methods.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T6: hypothesis-mode -- each model aligns its OWN recognition. Identity-gated F1 + matched-WBE.
# ----------------------------------------------------------------------------------------
def _hyp_table():
    S = "buckeye-segA-5h"

    def _m(name, key, mul, fmt="{:.1f}"):
        return {"var": _RESULTS.get(f"hyp-align/{name}-metrics.txt"), "key": key, "mul": mul, "fmt": fmt}

    def _row(label, name):
        return {
            "label": label,
            "cells": {
                "mwbe": _m(name, "matched_wbe", 1000.0),
                "f50": _m(name, "f1_50ms", 1.0, "{:.3f}"),
                "f100": _m(name, "f1_100ms", 1.0, "{:.3f}"),
                "f200": _m(name, "f1_200ms", 1.0, "{:.3f}"),
                "rm": _m(name, "match_rate_ref", 100.0),
            },
        }

    grad = [
        ("Whisper-base", f"whisper-base-charlev-{S}-grad"),
        ("Voxtral", f"voxtral-charlevlogmel-{S}-grad"),
        ("Phi-4-MM", f"phi4mm-charlev-spc-{S}-grad"),
        ("Canary-Qwen", f"canary-qwen-charlev-spc-logmel-st15-{S}-grad"),
        ("Parakeet RNN-T", f"parakeet-rnnt-1.1b-logmel-{S}-grad"),
        ("Parakeet TDT", f"parakeet-tdt-0.6b-v2-logmel-{S}-grad"),
    ]
    baselines = [
        ("Whisper cross-attn DTW", f"whisper-crossattn-{S}"),
        ("CrisperWhisper (official)", f"crisperwhisper-official-{S}"),
    ]
    cols = [
        {"key": "mwbe", "header": "matched-WBE [ms]"},
        {"key": "f50", "header": "F1@50"},
        {"key": "f100", "header": "F1@100"},
        {"key": "f200", "header": "F1@200"},
        {"key": "rm", "header": "ref-match [\\%]"},
    ]
    rows = [_row(*m) for m in grad]
    rows.append({"label": None, "cells": {}})
    rows += [_row(*b) for b in baselines]

    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=(
            "Hypothesis-mode alignment on Buckeye-segA: each model aligns its OWN recognition. "
            "Identity-gated F1 at collar and Levenshtein-matched WBE against the reference."
        ),
        label="tab:hyp",
        label_header="Model",
        col_align="|l|r|r|r|r|r|",
    )
    tk.register_output("tables/hyp.tex", job.out_tex)
