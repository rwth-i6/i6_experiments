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
    _compare_table()  # ground-truth forced-mode only (separate hyp table below)
    _compare_table(with_hyp=True)  # variant: hyp-mode columns merged onto the grad rows
    _hyp_table()
    _owsm_layer_table()  # OWSM-CTC grad-align per inter-CTC emit block (side table)
    _phi4_prompt_table()  # Phi-4-MM grad-align vs the spliced instruction (side table)
    _cost_table()  # forward vs batched grad-backward per model (T5 cost)
    _streaming_offset_table()  # streaming start/end signed offset (grad vs native viterbi)
    _time_stretch_table()  # length robustness (grad vs MMS-FA, vocoder vs resample)
    _word_length_table()  # word-duration accuracy (signed word-width error) per model
    _fairness_table()  # T2 fairness 2x2 (grad vs cross-attn x char/subword)
    _ablation_table()  # T3a grad-score reduction (cross-model)
    _attribution_table()  # T3b attribution method (cross-model, fast models)
    _alignopts_table()  # T3b align-opts (grad vs cross-attn, same DP)


# ----------------------------------------------------------------------------------------
# Streaming signed-offset: per streaming model, grad vs native viterbi, SIGNED start/end boundary
#     offset (pred - ref, ms; + = late). Native chunked viterbi starts late / ends early (emission
#     delay); grad ~0. Data = start_signed_mean / end_signed_mean (align_metrics out_metrics).
# ----------------------------------------------------------------------------------------
def _streaming_offset_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def _off(base, key):
        v = _RESULTS.get(f"{base}-wbe.txt")
        j = getattr(v, "creator", None) if v is not None else None
        m = getattr(j, "out_metrics", None) if j is not None else None
        if m is None and j is not None:
            # CTC forced-align baseline (ParakeetCtcForcedAlignJob) names its metrics dict out_word_metrics.
            m = getattr(j, "out_word_metrics", None)
        return {"var": m, "key": key, "mul": 1000.0, "fmt": "{:+.0f}"}

    def g(stem_sa, stem_ti):
        return (f"align/{stem_sa}", f"align/{stem_ti}")

    MODELS = [
        (
            "Emformer RNN-T",
            [
                (
                    "grad",
                    *g(
                        f"emformer-rnnt-prefix-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"emformer-rnnt-prefix-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "native viterbi",
                    f"baseline-emformer-rnnt-native-viterbi-{S}",
                    f"baseline-emformer-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "FastConformer-CTC",
            [
                (
                    "grad",
                    *g(
                        f"fastconformer-stream-ctc-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-ctc-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                ("native viterbi", f"baseline-fastconformer-stream-ctc-{S}", f"baseline-fastconformer-stream-ctc-{T}"),
            ],
        ),
        (
            "FastConformer-RNN-T",
            [
                (
                    "grad",
                    *g(
                        f"fastconformer-stream-rnnt-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-rnnt-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "native viterbi",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{S}",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
    ]

    cols = [
        {"key": "method", "header": "Align \\\\ method"},
        {"key": "t_start", "header": "start \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "t_end", "header": "end \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "s_start", "header": "start \\\\ {[ms]}", "group": "Buckeye"},
        {"key": "s_end", "header": "end \\\\ {[ms]}", "group": "Buckeye"},
    ]

    rows = []
    for gi, (model, methods) in enumerate(MODELS):
        if gi > 0:
            rows.append({"label": None, "cells": {}})
        for mi, (mlabel, sa, ti) in enumerate(methods):
            if mi > 0:
                rows.append({"cline": True})
            rows.append(
                {
                    "label": model if mi == 0 else "",
                    "cells": {
                        "method": mlabel,
                        "t_start": _off(ti, "start_signed_mean"),
                        "t_end": _off(ti, "end_signed_mean"),
                        "s_start": _off(sa, "start_signed_mean"),
                        "s_end": _off(sa, "end_signed_mean"),
                    },
                }
            )

    caption = (
        "Signed word-boundary offset (predicted minus reference, ms; positive = late) for the streaming "
        "models, grad-align vs the model's native chunked-viterbi alignment, on TIMIT-test and Buckeye-segA. "
        "The native streaming alignment is systematically late on both boundaries (limited right-context "
        "emission delay; the start is delayed more than the end), whereas grad-align is far closer to "
        "unbiased (tens vs hundreds of ms). This signed view complements the absolute WBE: the "
        "native streaming viterbi's large WBE is largely a systematic emission-delay bias."
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:streaming-offset",
        label_header="Model",
        col_align="|l|l|r|r|r|r|",
    )
    tk.register_output("tables/streaming-offset.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# Length robustness: grad-align (wav2vec2) vs MMS-FA forced-align under audio time-stretch,
#     vocoder (librosa.effects.time_stretch, pitch-preserving) vs resample (pitch-shifted),
#     at 1.0/1.5/2.0/3.0x on TIMIT-val. grad-align tolerates mild stretch but degrades sharply
#     at 2-3x (resample worst); forced-align re-runs viterbi so it stays robust.
# ----------------------------------------------------------------------------------------
def _time_stretch_table():
    GRAD0 = "align/wav2vec2ctc-fproj_out-timit-val-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"

    def _grad(ts, method):
        if ts == "1.0":
            return GRAD0
        sfx = f"-ts{ts}" + ("-resample" if method == "resample" else "")
        return f"align/wav2vec2ctc-fproj_out-timit-val-L2_grad-pertoken{sfx}-asotTrue-bs-5-en0.5-sil1.0"

    def _mms(ts, method):
        if ts == "1.0":
            return "baseline-mms_fa-timit-val"
        return "baseline-mms_fa-timit-val" + f"-ts{ts}" + ("-resample" if method == "resample" else "")

    MODELS = [("grad-align", _grad), ("MMS-FA", _mms)]
    TS = ["1.0", "1.5", "2.0", "3.0"]
    KEYS = ["ts10", "ts15", "ts20", "ts30"]

    cols = [{"key": "method", "header": "Stretch method"}]
    cols += [{"key": k, "header": ts, "group": "WBE [ms]"} for k, ts in zip(KEYS, TS)]

    rows = []
    for gi, (model, basefn) in enumerate(MODELS):
        if gi > 0:
            rows.append({"label": None, "cells": {}})
        for mi, method in enumerate(("vocoder", "resample")):
            if mi > 0:
                rows.append({"cline": True})
            cells = {"method": method}
            for k, ts in zip(KEYS, TS):
                cells[k] = _wbe(basefn(ts, method))
            rows.append({"label": model if mi == 0 else "", "cells": cells})

    caption = (
        "Length-perturbation robustness on TIMIT-val: "
        "word-boundary error (WBE, ms) of grad-align (wav2vec2-CTC) "
        "vs the MMS-FA forced-align baseline, "
        "under audio time-stretch at factors 1.0 to 3.0 (columns), "
        "with two stretch methods: "
        "vocoder (phase vocoder, pitch-preserving) and resample (pitch-shifted). "
        "Grad-align tolerates mild stretch but degrades sharply at 2-3x "
        "(resample, which also shifts pitch, is worst), "
        "whereas native forced-align re-runs Viterbi on the stretched audio and stays robust."
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:time-stretch",
        label_header="Model",
        col_align="|l|l|r|r|r|r|",
    )
    tk.register_output("tables/time-stretch.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# Word-length (duration) accuracy, full breakdown (Buckeye-segA): WBE (mean abs boundary error),
#     signed start/end offset (pred - ref; + = late), and word-WIDTH signed error (= end - start;
#     0 = correct duration). The CTC/forced-align posteriors bias start LATE + end EARLY -> the word
#     shrinks from both ends, so |width| ~ WBE; grad shifts start+end the SAME direction (a positional
#     lead, not a width change), so width is far below WBE -> grad is more accurate on word duration.
#     TIMIT-test shows the same pattern, milder (grad's lead is largest on spontaneous Buckeye).
# ----------------------------------------------------------------------------------------
def _word_length_table():
    S = "buckeye-segA-5h"

    def _m(base, key, fmt):
        v = _RESULTS.get(f"{base}-wbe.txt")
        j = getattr(v, "creator", None) if v is not None else None
        mm = getattr(j, "out_metrics", None) if j is not None else None
        if mm is None and j is not None:
            mm = getattr(j, "out_word_metrics", None)
        return {"var": mm, "key": key, "mul": 1000.0, "fmt": fmt}

    MODELS = [
        (
            "Wav2Vec2-CTC",
            [
                ("grad", f"align/wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"),
                ("posteriors", f"baseline-mms_fa-{S}"),
            ],
        ),
        (
            "Phoneme-CTC",
            [
                ("grad", f"align/phoneme-vitouphy-prefixfwd-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0"),
                ("posteriors", f"baseline-phoneme-fa-{S}-word"),
            ],
        ),
        (
            "Nvidia CTC",
            [
                ("grad", f"align/parakeet-ctc-1.1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"),
                ("posteriors", f"baseline-parakeet-ctc-1.1b-{S}"),
            ],
        ),
        (
            "OWSM-CTC",
            [
                ("grad", f"align/owsm-ctc-v4-1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"),
                ("posteriors", f"baseline-owsm-ctc-v4-1b-{S}"),
            ],
        ),
        (
            "Whisper-base",
            [
                ("grad", f"align/whisper-base-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0"),
                ("cross-attn", f"align/baseline-whisper-base-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil1.0"),
            ],
        ),
    ]

    cols = [
        {"key": "method", "header": "Align method"},
        {"key": "wbe", "header": "WBE", "group": "[ms]"},
        {"key": "start", "header": "start off.", "group": "[ms]"},
        {"key": "end", "header": "end off.", "group": "[ms]"},
        {"key": "width", "header": "width err.", "group": "[ms]"},
    ]
    rows = []
    for gi, (model, methods) in enumerate(MODELS):
        if gi > 0:
            rows.append({"label": None, "cells": {}})
        for mi, (mlabel, base) in enumerate(methods):
            if mi > 0:
                rows.append({"cline": True})
            rows.append(
                {
                    "label": model if mi == 0 else "",
                    "cells": {
                        "method": mlabel,
                        "wbe": _m(base, "wbe", "{:.0f}"),
                        "start": _m(base, "start_signed_mean", "{:+.0f}"),
                        "end": _m(base, "end_signed_mean", "{:+.0f}"),
                        "width": _m(base, "width_signed_err", "{:+.0f}"),
                    },
                }
            )

    caption = (
        "Word-length (duration) accuracy on Buckeye-segA, per model, "
        "grad-align vs the model's alternative aligner. "
        "WBE = mean absolute word-boundary error; "
        "start/end off. = signed mean boundary offset (predicted minus reference, positive = late); "
        "width err. = signed word-width error (= end minus start offset; 0 = correct duration). "
        "The CTC / forced-align posteriors bias the start late and the end early, "
        "so words shrink from both ends and the width error is comparable to the WBE; "
        "grad-align shifts both boundaries in the same direction (a positional lead, not a width change), "
        "so its width error is far below its WBE. "
        "Grad is thus consistently more accurate on word duration. "
        "(TIMIT-test shows the same pattern, milder.)"
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:word-length",
        label_header="Model",
        col_align="|l|l|r|r|r|r|",
    )
    tk.register_output("tables/word-length.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T2 fairness 2x2: whisper grad vs cross-attn DTW, char vs subword. Method-consistent -- both use the
#     PLAIN cross-attn forced align (no head auto-selection), so the only variable per column is the
#     tokenization. Grad wins every cell; char helps grad; char collapses cross-attn on spontaneous Buckeye.
# ----------------------------------------------------------------------------------------
def _fairness_table():
    def gA(stem):
        return f"align/{stem}"

    DATASETS = [("TIMIT", "timit-test"), ("Buckeye", "buckeye-segA-5h")]
    cols = [
        {"key": "method", "header": "Align \\\\ method"},
        {"key": "c_wbe", "header": "WBE \\\\ {[ms]}", "group": "char"},
        {"key": "c_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "char"},
        {"key": "s_wbe", "header": "WBE \\\\ {[ms]}", "group": "subword"},
        {"key": "s_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "subword"},
    ]
    rows = []
    for di, (dlabel, ds) in enumerate(DATASETS):
        if di > 0:
            rows.append({"label": None, "cells": {}})
        gc = gA(f"whisper-base-logmel-{ds}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0")
        gs = gA(f"whisper-base-logmel-{ds}-L2_grad-pertoken-subword-asotTrue-bs-5-en0.5-sil1.0")
        xc = f"baseline-whisper-crossattn-charlev-{ds}"
        xs = f"baseline-whisper-crossattn-{ds}"
        rows.append(
            {
                "label": dlabel,
                "cells": {
                    "method": "grad",
                    "c_wbe": _wbe(gc),
                    "c_a50": _acc(gc, "acc_50ms"),
                    "s_wbe": _wbe(gs),
                    "s_a50": _acc(gs, "acc_50ms"),
                },
            }
        )
        rows.append({"cline": True})
        rows.append(
            {
                "label": "",
                "cells": {
                    "method": "cross-attn",
                    "c_wbe": _wbe(xc),
                    "c_a50": _acc(xc, "acc_50ms"),
                    "s_wbe": _wbe(xs),
                    "s_a50": _acc(xs, "acc_50ms"),
                },
            }
        )

    caption = (
        "Fairness 2x2 on whisper-base: gradient alignment vs cross-attention DTW, at character-level and "
        "subword tokenization (TIMIT-test, Buckeye-segA; WBE and acc@50). Both methods use the same plain "
        "cross-attention forced alignment (no head auto-selection), so the only variable per column pair is "
        "the tokenization. Grad-align wins every cell. Character targets help grad on both datasets; they "
        "help cross-attention on clean TIMIT but collapse it on spontaneous Buckeye (264.6 ms), where char "
        "tokens exceed the 80 ms attention grid. (The headline table uses the stronger head-auto-selected "
        "cross-attention variant for its subword baseline.)"
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:fairness",
        label_header="Data",
        col_align="|l|l|r|r|r|r|",
    )
    tk.register_output("tables/fairness.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T3a grad-score reduction ablation (cross-model, Buckeye-segA, word-topology fixed): the per-token
#     reduction L0.5 / L1 / L2 / L2_e across one representative model per family. A few ms apart -> the
#     reduction choice is not critical; the saliency signal, not the norm, carries the alignment.
#     ("dot"/sum reductions are excluded: signed, they break the time-softmax scoring.)
# ----------------------------------------------------------------------------------------
def _ablation_table():
    S = "buckeye-segA-5h"
    SFX = "asotTrue-bs-5-en0.5-sil1.0-wordtopo"
    MODELS = [
        ("Wav2Vec2-CTC", "wav2vec2ctc-fproj_out-prefixfwd"),
        ("Whisper-base", "whisper-base-logmel-charlev-spc"),
        ("Parakeet RNN-T", "parakeet-rnnt-1.1b-logmel"),
        ("Emformer (stream)", "emformer-rnnt-prefix-logmel"),
        ("Voxtral", "voxtral-charlevlogmel"),
    ]
    REDS = [("L0.5", "L0.5_grad"), ("L1", "L1_grad"), ("L2", "L2_grad"), ("L2_e", "L2_e_grad")]
    cols = [{"key": rk, "header": rl} for rl, rk in REDS]
    rows = []
    for mlabel, mpre in MODELS:
        rows.append(
            {
                "label": mlabel,
                "cells": {rk: _wbe(f"align/{mpre}-abl-{S}-{rk}-pertoken-{SFX}") for _, rk in REDS},
            }
        )
    caption = (
        "Grad-score reduction ablation across families, Buckeye-segA (word-topology fixed; WBE ms). The "
        "per-token reduction (L0.5 / L1 / L2 / L2_e) moves the word-boundary error by only a few ms on every "
        "family, so the reduction is not critical -- alignment quality is set by the model's saliency, not "
        "the norm. (dot/sum reductions are excluded: signed, they break the time-softmax scoring.)"
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:gradscore",
        label_header="Model",
        col_align="|l|r|r|r|r|",
    )
    tk.register_output("tables/ablation.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T3b attribution-method ablation (cross-model, Buckeye-segA, word-topology fixed): plain L2 input-grad
#     vs SmoothGrad / VarGrad / Integrated-Gradients / Expected-Gradients, on the two fast models. The
#     single-pass gradient is already a good attribution -> the multi-pass methods are not worth it
#     (the slow families are omitted for the same cost reason).
# ----------------------------------------------------------------------------------------
def _attribution_table():
    S = "buckeye-segA-5h"
    SFX = "asotTrue-bs-5-en0.5-sil1.0-wordtopo"
    MODELS = [
        ("Wav2Vec2-CTC", "wav2vec2ctc-fproj_out-prefixfwd"),
        ("Whisper-base", "whisper-base-logmel-charlev-spc"),
    ]
    METHODS = [
        ("grad (L2)", "L2_grad"),
        ("SmoothGrad", "L2-smoothgrad-std0.01-n8"),
        ("VarGrad", "L2-vargrad-std0.02-n16"),
        ("IntGrad", "L2-integrated-grad-ig16"),
        ("ExpGrad", "L2-expected-grad-eg16-bstd0.05"),
    ]
    cols = [{"key": mk, "header": ml} for ml, mk in METHODS]
    rows = []
    for mlabel, mpre in MODELS:
        rows.append(
            {
                "label": mlabel,
                "cells": {mk: _wbe(f"align/{mpre}-abl-{S}-{mk}-pertoken-{SFX}") for _, mk in METHODS},
            }
        )
    caption = (
        "Attribution-method ablation on the two fast models, Buckeye-segA (word-topology fixed; WBE ms). "
        "The plain single-pass L2 input-gradient vs SmoothGrad, VarGrad, Integrated-Gradients (16 steps) and "
        "Expected-Gradients. The multi-pass methods do not improve over the single-pass gradient "
        "(Integrated-Gradients converges back to it), so they are not worth their extra passes; the slower "
        "model families are omitted for the same cost reason."
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:attribution",
        label_header="Model",
        col_align="|l|r|r|r|r|r|",
    )
    tk.register_output("tables/attribution.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T3b align-OPTS (DP decoding) on Buckeye-segA: the SAME DP applied to whisper grad (char) and whisper
#     cross-attn (auto heads) -- different signal, same decoder. word-topology / silence / DTW corners.
#     Mostly consistent across signals; silence modeling helps grad but not attention (the attention
#     matrix already has ~no mass in silence). The openai whisper-DTW is the apply-log-off corner.
# ----------------------------------------------------------------------------------------
def _alignopts_table():
    GP = "align/whisper-base-logmel-buckeye-segA-5h-L2_grad-pertoken-charlev-spc-asotTrue-bs-5"
    AP = "align/baseline-whisper-base-crossattn-auto-buckeye-segA-5h-asotTrue-bs-5"
    # (row label, grad suffix, cross-attn suffix)
    ROWS = [
        ("default (CTC-topo)", "-en0.5-sil1.0", "-en0.5-sil1.0"),
        ("word-topology", "-en0.5-sil1.0-wordtopo", "-en0.5-sil1.0-wordtopo"),
        ("silence off", "-en0.5", ""),
        ("DTW (no blank)", "-en0.5-sil1.0-dtw", "-en0.5-sil1.0-dtw"),
        ("whisper-DTW (openai)", "-en0.5-sil1.0-wdtw", "-en0.5-sil1.0-wdtw"),
    ]
    cols = [
        {"key": "g_wbe", "header": "WBE \\\\ {[ms]}", "group": "grad"},
        {"key": "g_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "grad"},
        {"key": "a_wbe", "header": "WBE \\\\ {[ms]}", "group": "cross-attn"},
        {"key": "a_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "cross-attn"},
    ]
    rows = []
    for lbl, gs, as_ in ROWS:
        gb, ab = GP + gs, AP + as_
        rows.append(
            {
                "label": lbl,
                "cells": {
                    "g_wbe": _wbe(gb),
                    "g_a50": _acc(gb, "acc_50ms"),
                    "a_wbe": _wbe(ab),
                    "a_a50": _acc(ab, "acc_50ms"),
                },
            }
        )
    caption = (
        "Alignment DP options on Buckeye-segA, applied to the same decoder for both signals: whisper grad "
        "(char-level) and whisper cross-attention (auto-selected heads). The decoding is shared "
        "infrastructure -- the grad-vs-attention difference lives in the signal, not the decoder -- and the "
        "options behave mostly consistently. The exception is silence modeling, which helps the grad signal "
        "but not the attention signal (the attention matrix already carries ~no mass in silence). The openai "
        "whisper-DTW row is the apply-log-off / no-silence corner, an exact special case of our DP, and the "
        "weakest setting for both."
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:alignopts",
        label_header="DP option",
        col_align="|l|r|r|r|r|",
    )
    tk.register_output("tables/alignopts.tex", job.out_tex)


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
def _compare_table(with_hyp=False):
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
                        f"wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"wav2vec2ctc-fproj_out-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
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
                        f"phoneme-vitouphy-prefixfwd-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0",
                        f"phoneme-vitouphy-prefixfwd-{T}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0",
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
            "Nvidia CTC",
            [
                (
                    "grad",
                    *g(
                        f"parakeet-ctc-1.1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"parakeet-ctc-1.1b-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                ("posteriors", f"baseline-parakeet-ctc-1.1b-{S}", f"baseline-parakeet-ctc-1.1b-{T}"),
            ],
        ),
        (
            # OWSM-CTC: general graphemic CTC, our least-favourable model for grad-align (still runs,
            # just imprecise). Shown with its BEST emit block (6); per-block detail in tab:owsm-per-layer.
            "OWSM-CTC",
            [
                (
                    "grad",
                    *g(
                        f"owsm-ctc-v4-1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"owsm-ctc-v4-1b-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                # posteriors = CTC forced-align of the model's emission; best/natural at the final block.
                ("posteriors", f"baseline-owsm-ctc-v4-1b-{S}", f"baseline-owsm-ctc-v4-1b-{T}"),
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
            # Emformer RNN-T: the streaming transducer (vs offline Parakeet). grad resists the
            # streaming emission delay that the native forced alignment carries.
            "Emformer (streaming)",
            [
                (
                    "grad",
                    *g(
                        f"emformer-rnnt-prefix-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"emformer-rnnt-prefix-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-emformer-rnnt-native-viterbi-{S}",
                    f"baseline-emformer-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "FastConformer-CTC (stream)",
            [
                (
                    "grad",
                    *g(
                        f"fastconformer-stream-ctc-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-ctc-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-fastconformer-stream-ctc-{S}",
                    f"baseline-fastconformer-stream-ctc-{T}",
                ),
            ],
        ),
        (
            "FastConformer-RNN-T (stream)",
            [
                (
                    "grad",
                    *g(
                        f"fastconformer-stream-rnnt-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-rnnt-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{S}",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{T}",
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

    # Hyp-mode (each model aligns its OWN recognition, Buckeye only) -> grad rows only, where it exists.
    HYP = {
        "Whisper-base": f"whisper-base-charlev-{S}-grad",
        "Voxtral": f"voxtral-charlevlogmel-{S}-grad",
        "Phi-4-MM": f"phi4mm-charlev-spc-{S}-grad",
        "Canary-Qwen": f"canary-qwen-charlev-spc-logmel-st15-{S}-grad",
        "Parakeet RNN-T": f"parakeet-rnnt-1.1b-logmel-{S}-grad",
        "Parakeet TDT": f"parakeet-tdt-0.6b-v2-logmel-{S}-grad",
        "Whisper-large-v3": f"whisper-large-v3-charlev-{S}-grad",
    }
    # Models with no word-level own-recognition -> hyp-mode is structurally n/a (not "unrun"):
    # MMS_FA (Wav2Vec2-CTC) + Phoneme-CTC emit no word boundaries; MFA is a forced-aligner, not a recognizer.
    HYP_NA = {"Wav2Vec2-CTC", "Phoneme-CTC", "MFA (GMM-HMM)"}

    def _hm(name, key, mul, fmt):
        return {"var": _RESULTS.get(f"hyp-align/{name}-metrics.txt"), "key": key, "mul": mul, "fmt": fmt}

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
    if with_hyp:
        # hyp-mode (own-recognition) columns: matched-WBE + F1@50, on grad rows only.
        cols += [
            {"key": "h_mwbe", "header": "m-WBE \\\\ {[ms]}", "group": "Buckeye (hyp)"},
            {"key": "h_f50", "header": "F1@50", "group": "Buckeye (hyp)"},
        ]

    def _model_rows(groups, lead_hline):
        rows = []
        for gi, (model, methods) in enumerate(groups):
            if gi > 0 or lead_hline:
                rows.append({"label": None, "cells": {}})  # \hline between models
            for mi, (mlabel, sa, ti) in enumerate(methods):
                if mi > 0:
                    rows.append({"cline": True})  # \cline between a model's own methods
                cells = {
                    "method": mlabel,
                    "t_wbe": _wbe(ti) if ti else None,
                    "t_a50": _acc(ti, "acc_50ms") if ti else None,
                    "t_a100": _acc(ti, "acc_100ms") if ti else None,
                    "s_wbe": _wbe(sa) if sa else None,
                    "s_a50": _acc(sa, "acc_50ms") if sa else None,
                    "s_a100": _acc(sa, "acc_100ms") if sa else None,
                }
                if with_hyp:
                    hn = HYP.get(model) if mi == 0 else None  # hyp only on the grad row
                    if hn:
                        cells["h_mwbe"] = _hm(hn, "matched_wbe", 1000.0, "{:.1f}")
                        cells["h_f50"] = _hm(hn, "f1_50ms", 1.0, "{:.3f}")
                    elif mi == 0 and model in HYP_NA:
                        # structurally no word-level own-recognition -> mark, don't leave empty
                        cells["h_mwbe"] = "n/a"
                        cells["h_f50"] = "n/a"
                    else:
                        cells["h_mwbe"] = None
                        cells["h_f50"] = None
                rows.append({"label": model if mi == 0 else "", "cells": cells})
        return rows

    rows = _model_rows(MODELS, False) + _model_rows(REFERENCE, True)

    caption = (
        "Per-model alignment quality: gradient-based alignment vs. the model's own native / "
        "attention aligner, on TIMIT-test and Buckeye (segment-A 5h subset). "
        "The attention aligners (cross-/self-attn weights) use each model's native subword "
        "tokenization -- their best setting per align method, as the coarse attention grid cannot resolve "
        "char-level targets; grad-align and the CTC / transducer aligners use the same per-token "
        "units within each model. MFA (GMM-HMM) is a dedicated-aligner reference."
    )
    if with_hyp:
        caption += (
            " The rightmost group adds hypothesis-mode metrics (each model aligns its OWN "
            "recognition; matched-WBE and identity-gated F1@50ms), shown on the grad row. "
            "n/a = no word-level own-recognition (MMS\\_FA / Phoneme-CTC emit no word boundaries; "
            "MFA is a forced-aligner)."
        )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:per-model-methods" + ("-hyp" if with_hyp else ""),
        label_header="Model",
        col_align="|l|l|r|r|r|r|r|r|" + ("r|r|" if with_hyp else ""),
    )
    tk.register_output("tables/per-model-" + ("merged" if with_hyp else "methods") + ".tex", job.out_tex)


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
        ("Whisper-large-v3", f"whisper-large-v3-charlev-{S}-grad"),
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
        adjustbox=True,
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


# ----------------------------------------------------------------------------------------
# T7: OWSM-CTC grad-align emitted from each inter-CTC self-conditioning block (side table).
# ----------------------------------------------------------------------------------------
def _owsm_layer_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def _tag(layer):
        # blocks 6/12/15/21 carry an `lyr{N}-` tag; block 27 is the default final-emission base.
        return "" if layer == 27 else f"lyr{layer}-"

    def gbase(layer, ds):  # grad-align output base
        return f"align/owsm-ctc-v4-1b-{_tag(layer)}{ds}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"

    def fbase(layer, ds):  # forced-align (posteriors) baseline base
        return f"baseline-owsm-ctc-v4-1b-{_tag(layer)}{ds}"

    cols = [
        {"key": "layer", "header": "Emit \\\\ block"},
        {"key": "t_grad", "header": "grad \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "t_fa", "header": "posteriors \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "s_grad", "header": "grad \\\\ {[ms]}", "group": "Buckeye"},
        {"key": "s_fa", "header": "posteriors \\\\ {[ms]}", "group": "Buckeye"},
    ]
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
    caption = (
        "OWSM-CTC gradient alignment emitted from each inter-CTC self-conditioning block "
        "(6/12/15/21) vs the final block (27), on TIMIT-test and Buckeye-segA. The CTC emission "
        "forced-aligns well at every block ($\\sim$142\\,ms), but grad-align is weak at all of them "
        "-- worse than the 153\\,ms uniform-split trivial baseline, the earliest block 6 the least weak. "
        "The method still runs; this is simply our least-favourable model (see discussion)."
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:owsm-per-layer",
        label_header="OWSM-CTC",
        col_align="|l|l|r|r|r|r|",
    )
    tk.register_output("tables/owsm-per-layer.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T8: Speech-LLM prompt-splice sensitivity (Phi-4-MM): grad-align WBE vs the instruction.
# ----------------------------------------------------------------------------------------
def _phi4_prompt_table():
    def base(tag, level):
        return f"align/phi4mm-prompt-{tag}-{level}-timit-test-L2_e_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"

    cols = [
        {"key": "sub", "header": "subword [ms]"},
        {"key": "char", "header": "char [ms]"},
    ]
    prompts = [
        ("default", "into text (neutral)"),
        ("verbatim", "verbatim, exactly as spoken"),
        ("word", "exactly, word for word"),
        ("char", "at the character level"),
    ]
    rows = []
    for tag, label in prompts:
        rows.append(
            {
                "label": label,
                "cells": {
                    "sub": "n/a" if tag == "char" else _wbe(base(tag, "subword")),
                    "char": _wbe(base(tag, "char")),
                },
            }
        )
    caption = (
        "Prompt sensitivity of gradient alignment on a speech LLM (Phi-4-MM), TIMIT-test, forced "
        "ground-truth transcription -- only the text instruction spliced around the audio is varied. "
        "Word-boundary error for subword and character-level targets. The last row instructs "
        "character-level output, paired only with character-level targets."
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:phi4-prompt",
        label_header="Instruction",
        col_align="|l|r|r|",
    )
    tk.register_output("tables/phi4-prompt.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T5: cost -- forward vs batched grad-backward, per model (same GPU).
# ----------------------------------------------------------------------------------------
def _cost_table():
    m = _RESULTS.get("cost-benchmark-metrics.txt")

    def cell(name, metric):
        return {"var": m, "key": f"{name}|{metric}", "fmt": "{:.0f}", "missing": "(*)"}

    cols = [
        {"key": "fwd", "header": "forward [ms/s]"},
        {"key": "bwd", "header": "+grad backward [ms/s]"},
        {"key": "tot", "header": "grad total [ms/s]"},
    ]
    models = [
        "Wav2Vec2-CTC",
        "Nvidia CTC",
        "OWSM-CTC",
        "Whisper-base",
        "Parakeet RNN-T",
        "Parakeet TDT",
        "Voxtral",
        "Phi-4-MM",
        "Canary-Qwen",
    ]
    rows = [
        {
            "label": name,
            "cells": {
                "fwd": cell(name, "fwd_ms_per_s"),
                "bwd": cell(name, "bwd_ms_per_s"),
                "tot": cell(name, "total_ms_per_s"),
            },
        }
        for name in models
    ]
    caption = (
        "Cost of gradient alignment, per model, on a single GPU (TIMIT-test, 40 seqs): "
        "wall-clock for the forward pass vs the batched per-token grad backward, in ms per second "
        "of audio. The native methods (forced-align / attention-DTW) consume the same forward and "
        "skip the backward, so the forward column is their cost; gradient alignment adds the backward "
        "(fast batched variant). The align/DP step (shared, ~identical across methods) is excluded. "
        "(*) = batched backward unavailable for this model (vmap-incompatible attention kernel)."
    )
    job = WriteLatexTableJob(
        adjustbox=True,
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:cost",
        label_header="Model",
        col_align="|l|r|r|r|",
    )
    tk.register_output("tables/cost.tex", job.out_tex)
