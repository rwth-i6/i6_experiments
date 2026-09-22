"""
DLM on top of Albert Zeyer's FZJ text-injection ASR winner.

Step 1 (this file today): rebuild the winner's graph with Albert's own builder, so every job gets
the hash it has in his setup, and import his finished jobs instead of recomputing them.
Nothing here is meant to be run by the manager yet.

The winner and all its recogs come from ONE builder call, copied verbatim from the ``_sa == 50``
iteration of the specaug loop in ``exp2026_05_28_tts_encoder_fzj.py::py`` -- including
``with_ctc_lm_recog=True``, which wires CTC+LM, AED+CTC+LM label-sync and both DLM-sum recogs
(with the RZ DLM ``evKXenj1Z2j3`` from his ``import/dlm/``).

Import, from the setup root (loads the config, links finished jobs, submits nothing)::

    wrap.sh python ./sis c --script recipe/i6_experiments/users/dorian_koch/e2026_09_16__fzj_dlm/fzj_dlm.py \\
        -c 'from i6_experiments.users.dorian_koch.e2026_09_16__fzj_dlm.fzj_dlm import import_albert_jobs, report' \\
        -c 'import_albert_jobs("dryrun")' -c 'report()'

(``"symlink"`` instead of ``"dryrun"`` to actually link.)
Project notes: ``projects/2026-09-16-fzj-dlm.md`` in the local remote-setup repo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sisyphus import tk

import returnn.frontend as rf

__all__ = ["py", "import_albert_jobs", "report", "report_outputs"]
__setup_root_prefix__ = "exp2026_09_16_fzj_dlm"

# Albert's readable work trees. Many job dirs in the first are symlinks into
# /e/home/jusers/zeyer1/... (unreadable for us); the second holds the readable copies of those.
ALBERT_WORK_DIRS: List[str] = [
    "/e/project1/spell/zeyer1/setups/2026-05-28-tts-encoder/work",
    "/e/project1/spell/zeyer1/setups/2026-05-26-base/work",
]

WINNER_NAME = "pseudo-enc-logmel-mfatable-realdur2-lerp-dur07-packed-single-gumbel-muon-nep38-specaug50-stepcomp"
WINNER_TRAIN_JOB = "i6_core/returnn/training/ReturnnTrainingJob.8iFbool3x3TU"

# DLM data from the winner (paper-best DLM recipe, see dlm_on_winner.py):
#   "off"   -- not built
#   "smoke" -- built; only the first 4-GPU hypothesis bundle is registered (verify GPU use + outputs first)
#   "hyps"  -- all hypothesis bundles registered
#   "train" -- all bundles + the paper-best DLM trained on them (4-GPU DDP), user decision 2026-09-16 22:20
DLM_DATA_STAGE = "train"

# German arm names say what each trains on, on top of the vocab-surgered English winner (ep38):
#   zeroshot             -- no German training; winner decoded on MLS-de as-is (context only)
#   deAudio              -- 9 h German paired audio only (the baseline)
#   deText+enAudio       -- German text injection + English LS-960 audio (the claim)
#   deText+deAudio       -- German text injection + 9 h German audio x4, no English audio
#   deText+enAudio-enSpm -- first deText+enAudio run, INVALID: injection text used the English SPM (§104)
#   deText-only          -- German text injection, no audio at all (dropped, RETURNN empty-stream bugs)
# German cross-lingual arms (german_xling.py). "off" | "surgery" | "zeroshot" | "deText+enAudio-enSpm" | "deAudio".
# ⚠ COMMA-SEPARATED, and treated as a SET -- "deText+enAudio-enSpm,deAudio" builds both. This is not cosmetic: the
# stages are mutually exclusive branches, so flipping a single-valued flag from "deText+enAudio-enSpm" to "deAudio"
# REMOVES arm B from the graph, and the manager then stops tracking and resubmitting it. Arm B is a
# multi-hour training that resumes across walltime boundaries, so dropping it mid-flight would
# silently strand it. Verified 2026-09-19 by graph diff: under "deAudio" alone, arm B's training id
# is absent.
# "zeroshot" = the winner, UNMODIFIED, decoded on MLS-de test. It needs no training and no graph-build
# patch: the recog helpers take `task` as a parameter, so the German task is simply passed in.
# ⚠ Arm A is CONTEXT, never the baseline: the original 10,240-piece SPM is uppercase English and
# cannot write German orthography at all (no Ä Ö Ü), so its WER will be catastrophic by construction.
# Quoting an A->B gain as the contribution would be inflated by "we added three characters".
# deText-only (zero paired audio) crashed twice with `RuntimeError: max(): ... input.numel() == 0`.
# Root cause found 2026-09-20 and FIXED in RETURNN: `_packed_backend._torch_relayout_frames:5198`
# bound-checks `pos_raw.max() <= n_out`, and `.max()` raises on an empty tensor -- reached because
# `aed_pseudo_enc_frontend_single_stream_train_step:6037` runs feature extraction over the audio
# stream unconditionally, and under `ls_audio_subset=0` that stream is present but EMPTY.
# The bound is vacuously true for an empty layout, so the fix short-circuits on the (static) shape.
# ⚠ That patch is LOCAL to tools/returnn -- see backlog; it must survive a pull or deText-only breaks again.
# deText-only DROPPED (user call, 2026-09-20): it needs three RETURNN fixes in three files and the third
# (device propagation through the empty source) is of unknown depth. deText+deAudio answers the same question
# -- does the pseudo-encoder carry German once the English-audio language cue is gone -- with zero
# framework work, and is the better system besides. deText+enAudio is arm B re-run with the German SPM (§104).
GERMAN_STAGE = "deText+enAudio-enSpm,deText+enAudio,deText+deAudio,deText+deAudio-long,enarpa,deAudio"
# Acoustic-prior budget for the German arms. ⚠ "9h" is the usable one: the 1 h duration table is
# 21.8% floor-collapsed and its spectra are truncation-biased for affricates/stops (backlog 18, 21).
GERMAN_BUDGET = "9h"
# Plan verification step 1: cap arm B at 1 h walltime first. `__time_rqmt` is not hashed, so this is
# the SAME job as the full run -- it stops early and resumes when the cap is lifted.
GERMAN_SMOKE = False
# ⚠ SEPARATE from GERMAN_SMOKE, deliberately. Arm B is mid-run with an 11 h allocation; flipping the
# shared flag would cap its NEXT resubmission at 1 h. Arm C has never run, and arm B's 1 h smoke
# found FIVE distinct defects that graph-building cannot see (§69), so arm C gets the same treatment.
GERMAN_SMOKE_ARMC = False  # smoke PASSED 2026-09-19 19:09: ckpt loads, ogg zip reads, dev eval real, losses fall
# Arm C sweep: give the BASELINE its best shot before claiming arm B beats it (backlog 82/90/92a).
# Empty either list to disable. 4 arms x ~50 min training, NO recog (selected on the dev curve).
# `None` in the enc-mult list = encoder at full LR (the pilot's setting).
ARM_C_SWEEP_PEAK_LRS = (5e-4, 2e-4)
ARM_C_SWEEP_ENC_MULTS = (None, 0.1)
ARM_C_SWEEP_NEP = 10  # the useful window is epochs 2-5 (backlog 85); 10 gives margin
# ⚠ NO `keep_epochs` for the sweep, for two independent reasons:
#  1. `learning_rates` records dev scores for **every** epoch regardless of which checkpoints are
#     retained, and that curve is the whole selection signal here (§92a) -- so retention is
#     irrelevant to a sweep arm;
#  2. `cleanup_old_models` lives in **post_config**, so passing it via `extra_config_updates` raises
#     `AssertionError: cleanup_old_models in post_config would overwrite existing entry in config`.
# And at nep=10 the RETURNN default keep set is {4,5,8,10}, which already brackets the useful
# window (epochs 2-5) -- so even the FINAL run does not need the override the §78 trap seemed to
# demand. The trap was specific to nep=40, where the default set starts at 5 and the optimum was 3.
# Continue-train the winner with GlowTTS TTS audio added to its paired-audio branch (winner_plus_tts.py).
WINNER_PLUS_TTS = True
# Which finetune checkpoints to evaluate, e.g. 1 -> epoch.001.pt. Empty = off. The checkpoint is
# referenced as a RAW PATH (no creator), so this does not depend on the still-running training job and
# becomes runnable the moment the file exists.
# 🔴 **The FINAL epoch must be in here.** It was `= 1` alone, so the winner+TTS finetune -- a ~12 h,
# 10-epoch training -- would have run to completion with an eval wired only for its FIRST epoch, i.e.
# no readable answer to the question it exists to ask (does adding TTS audio help?). Epoch 1 already
# showed one sub-epoch of TTS audio HURTS (backlog 55); whether 10 epochs recovers is the actual
# result, and nothing was going to measure it. Keeping 1 as well makes it a dose-response pair
# rather than a single point.
EVAL_FINETUNE_EPOCHS = (1, 10)

# Hypothesis-pass WER of the +TTS finetune on REAL LS-960 audio and on GlowTTS audio -- the pair the
# finetune exists to move (winner 2.22% / 14.53%; old RZ CTC 5.02% / 10.25%, winner_plus_tts.py:7-9).
# The LS dev/test recogs do NOT answer this: they are a different population (dev/test, unaugmented)
# from the 2.22% (random 20k of LS-960 *train*, with the GetCtcHypsCfgV6 augmentation).
# Registers ONLY bundle 01, which carries 2 LS-960 audio passes AND 6 LM-text/GlowTTS shards --
# i.e. both numbers from one 4-GPU job instead of the full 12-bundle DLM data pass.
FT_HYPS_WER = True
FT_HYPS_EPOCH = 10
# Bundle whose GlowTTS text is DISJOINT from the finetune's training audio (parts 1-100).
# Bundle 03 = parts 143-222, read off its seq-list jobs' `info` files. See backlog 112.
FT_HYPS_CLEAN_BUNDLE = 3

# Continue the winner WITH its optimizer state, instead of importing its weights (user call).
# Motivation (backlog): the +TTS finetune got WORSE than its own starting point for ~5 epochs before
# recovering -- dev CE 0.1819 -> 0.1864 at epoch 5, same shape on devtrain, so not overfitting -- and
# the worst epoch coincides exactly with the LR peak (breakpoints [4.5, 9.0, 10] on [1e-5, 5e-4, ...]).
# A resumed run removes the cold-optimizer explanation AND, because the text partition is
# (epoch-1) % 75, advances the LM text to partitions 38-47 instead of re-reading 0-9 a third time.
# The flat LR is the discriminating half: if GlowTTS-audio WER still improves without a dip, the ramp
# was waste; if it does not improve, the dip was the price of learning the TTS data.
WINNER_PLUS_TTS_RESUMED = True
WINNER_PLUS_TTS_RESUMED_NEP = 48  # winner's 38 + 10 more, so the model scan finds epoch.038
WINNER_PLUS_TTS_RESUMED_LR = 1e-5

# 🔴 The ABLATION that makes the resumed +TTS arm interpretable (user call, 2026-09-20): continue the
# winner on its OWN data -- no TTS zips -- with the identical resume, identical flat LR and identical
# epoch range. Without it "+TTS is better" is confounded with "trained 10 sub-epochs longer", and that
# confound is NOT small here: the winner's OCLR had decayed to ~1e-6 by epoch 38, so resuming at a
# flat 1e-5 is a 10x LR increase that would move the model on its own. This arm holds everything
# fixed except the presence of Rossenbach's TTS audio in the `asr` branch.
# ⚠ Its text branch still advances to LM partitions 38-47 exactly as the +TTS arm does, so the two
# see the SAME injected text -- the only difference is the audio.
WINNER_CONT_NO_TTS = True
# The cold-start ablation (user, 2026-09-21): the ORIGINAL +TTS finetune's exact call (fresh optimizer via
# import_model_train_epoch1, nEp10, peak LR 5e-4 schedule) with `with_tts=False`. Completes the 2x2:
# {TTS, no TTS} x {cold optimizer, resumed optimizer}.
WINNER_COLD_NO_TTS = True
# The winner on the Loquacious eval subsets (out-of-domain), with and without our LS DLMs.
LOQ_EVAL_WINNER = True
# Albert's best Loquacious ASR (AED 5.75 dev / 6.47 test) with Albert's Loquacious DLM (RZ
# base-puttingItTogether(low)-nEp200, qOO1vlKjNpTW ep200; relayed to import/dlm/, sha256 c277a759... checked
# equal on RZ and FZJ 2026-09-21). Same architecture as our DLMs at model_dim 1024, Loquacious SPM.
LOQ_EVAL_ALBERT_ASR = True
LOQ_BEST_ASR_NAME = "base-large-srcExp0-nFullEp5_6-muon-lr2_5e3-bs24m-specaug60-stepcomp-len40s"
_dlm_hyp_jobs: List[Any] = []
_dlm_task_ref: List[Any] = []  # the DLM data task, for console inspection


def pinned_path(path: str) -> tk.Path:
    """
    A creator-less :class:`tk.Path` whose hash is FROZEN to its current location.

    Every raw checkpoint reference below is an absolute path inside ``gs.BASE_DIR`` with no creator
    job, which sisyphus warns about (`job_path.py:70`) for a real reason: such a path hashes by its
    **location string** (`:129-136` hashes ``(creator, path)``, and creator is None), so moving or
    renaming the setup silently re-hashes every job that consumes it -- orphaning finished work and
    re-running multi-day trainings. Pinning with ``hash_overwrite`` freezes the identity against
    that.

    ⚠ The overwrite is deliberately the path's OWN current string, which makes this change provably
    hash-neutral: ``_sis_hash`` builds ``(None, path)`` either way, byte for byte. Pinning to a
    prettier label (``"our_dlm_ep50"``) would be nicer to read and would re-hash all 2,048 jobs.
    Verified by graph diff: 0 ids moved.
    """
    return tk.Path(path, hash_overwrite=path)


# DLMs relayed from our RZ setup (rsync RZ -> FZJ, sha256-verified), by model_dim.
OUR_DLM_IMPORT_DIR = "/e/project1/spell/koch13/setups/2026-09-16-fzj-dlm/import/dlm"
DLM_N1280 = OUR_DLM_IMPORT_DIR + (
    "/base-scalingLaws-enc24-dec8-n1280-nEp200.ReturnnTrainingJob.LfAv45zfWLtF/output/models/epoch.200.pt"
)


# 🔴 OUR OWN trained DLM (backlog 87). `train_paper_best_dlm_4gpu` deliberately wires no recog --
# "evaluate with the batched DLM-sum instead" (dlm_on_winner.py:252) -- and `fzj_dlm.py` DISCARDS its
# return value, so nothing in the graph depended on this 50-epoch 4-GPU training. It would have
# finished green and produced no number. Referenced as a RAW PATH (no creator), like
# FINETUNE_TRAIN_JOB, so these recogs do not depend on the still-running job and become runnable the
# moment the checkpoint exists.
# ⚠ `num_epochs = 50` in its own returnn.config (the alias says nEp200 -- that is 200 counted the
# other way; RETURNN counts 50 with 4 GPUs), and `model_dim = 1280`. Both read off the config, not
# inferred from the alias.
OUR_TRAINED_DLM_EPOCH = 50
OUR_TRAINED_DLM = (
    "/e/home/jusers/koch13/jupiter/setups/2026-09-16-fzj-dlm"
    "/work/i6_core/returnn/training/ReturnnTrainingJob.pMb0YjIfsID0"
    f"/output/models/epoch.{OUR_TRAINED_DLM_EPOCH:03d}.pt"
)
# Albert's Loquacious DLM (see LOQ_EVAL_ALBERT_ASR). Loquacious SPM -- only for Loquacious-vocab ASR models.
LOQ_DLM = OUR_DLM_IMPORT_DIR + (
    "/base-puttingItTogether-low-nEp200.ReturnnTrainingJob.qOO1vlKjNpTW/output/models/epoch.200.pt"
)
# Set False to drop the eval of our own DLM (e.g. if its recogs crowd the queue).
EVAL_OUR_TRAINED_DLM = True


def _get_dlm(checkpoint: str, *, model_dim: int):
    """
    A DLM as an explicit-checkpoint model, same structure as Albert's ``_get_imported_dlm``
    (24-layer encoder / 8-layer decoder, RMSNorm, gated FF, rotary attention, input EOS),
    which read its dicts off the RZ training job's returnn.config. Only ``model_dim`` varies
    between our DLMs; checked against each job's returnn.config.
    """
    from i6_core.returnn.training import PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces.model import ModelDefWithCfg
    from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import ModelWithCheckpoint
    from denoising_lm_2024.error_correction_model import aed_model_def

    trafo_kwargs = {
        "model_dim": model_dim,
        "pos_enc": None,
        "norm": {"class": "rf.RMSNorm"},
        "ff": {"class": "returnn.frontend.decoder.transformer.FeedForwardGated"},
        "dropout": 0.0,
        "att_dropout": 0.0,
    }
    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(
            aed_model_def,
            {
                "_encoder_model_dict": {
                    "class": "returnn.frontend.encoder.transformer.TransformerEncoder",
                    "num_layers": 24,
                    "layer_opts": {"self_att": {"class": "rf.RotaryPosSelfAttention", "with_bias": False}},
                    **trafo_kwargs,
                },
                "_decoder_model_dict": {
                    "class": "returnn.frontend.decoder.transformer.TransformerDecoder",
                    "num_layers": 8,
                    "layer_opts": {"self_att": {"class": "rf.RotaryPosCausalSelfAttention", "with_bias": False}},
                    **trafo_kwargs,
                },
                "input_add_eos": True,
            },
        ),
        checkpoint=PtCheckpoint(pinned_path(checkpoint)),
    )


def py():
    from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

    prefix = get_setup_prefix_for_module(__name__)

    # Record the task / model / config the builder hands its CTC+LM recog (imported at call time from the
    # module), to reuse them verbatim for a plain CTC recog below instead of re-deriving them.
    import unittest.mock
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext import ctc_lm_batched as _ctc_lm

    ctc_lm_kwargs: Dict[str, Any] = {}
    _orig_ctc_lm = _ctc_lm.ctc_recog_recomb_labelwise_prior_auto_scale_batched

    def _record_ctc_lm(**kwargs):
        ctc_lm_kwargs.update(kwargs)
        return _orig_ctc_lm(**kwargs)

    with unittest.mock.patch.object(_ctc_lm, "ctc_recog_recomb_labelwise_prior_auto_scale_batched", _record_ctc_lm):
        winner_model = _train_winner(prefix)
    assert ctc_lm_kwargs, "builder did not call ctc_recog_recomb_labelwise_prior_auto_scale_batched"

    # Plain CTC (no LM, no prior) for the result table, 2026-09-16.
    _ctc_only_recog_batched(
        prefix=f"{prefix}/aed/{WINNER_NAME}/ctc-only-batched",
        task=ctc_lm_kwargs["task"],
        ctc_model=ctc_lm_kwargs["ctc_model"],
        aux_ctc_layer=ctc_lm_kwargs["aux_ctc_layer"],
        num_shards=ctc_lm_kwargs["num_shards"],
        extra_config=ctc_lm_kwargs.get("extra_config"),
    )

    _german_stages = {s.strip() for s in GERMAN_STAGE.split(",") if s.strip()}
    assert _german_stages <= {
        "off",
        "surgery",
        "zeroshot",
        "deText+enAudio-enSpm",
        "deText+enAudio",
        "deText-only",
        "deText+deAudio",
        "deText+deAudio-long",
        "enarpa",
        "deAudio",
    }, f"unknown German stage: {_german_stages}"

    if _german_stages & {"surgery", "zeroshot", "deText+enAudio-enSpm", "deAudio"}:
        # Widen the winner's output layer to the German vocab (10,240 -> 10,243). Cheap (CPU) and a
        # prerequisite for arms B/C, so it is built as soon as any German stage is on -- running it
        # early de-risks the arm rather than discovering a broken checkpoint mid-training.
        from .german_xling import get_surgered_winner_checkpoint
        from .winner_plus_tts import winner_checkpoint

        _de_ckpt = get_surgered_winner_checkpoint(winner_checkpoint(winner_model), budget=GERMAN_BUDGET)
        tk.register_output(f"{prefix}/german/winner-vocab-extended.pt", _de_ckpt)

    if "deText+enAudio-enSpm" in _german_stages:
        # Arm B -- THE CLAIM: English LS-960 audio + German text injection, from the surgered
        # checkpoint. See german_xling.train_german_arm_b for what differs from winner_plus_tts.
        from .german_xling import train_german_arm_b

        train_german_arm_b(
            prefix=f"{prefix}/german", winner_model=winner_model, budget=GERMAN_BUDGET, smoke=GERMAN_SMOKE
        )

    if "deText+enAudio" in _german_stages:
        # 🔴 Arm B, RE-RUN CORRECTLY (backlog 104). Arm B as first run is INVALID: its
        # `glow_tts_text_spm_opts` pointed at the ENGLISH 10,240 SPM while its four other tokenizers
        # used the German 10,243, so the injection text went through a tokenizer that cannot encode
        # an umlaut -- 25,975 `<unk>` (2.710% of target positions, 53.7% of lines) and the umlaut ids
        # were NEVER produced. It emitted 0 umlauts in 126,239 test words, identical to untrained
        # arm A. This is the arm the paper's claim rests on, so it has to be re-run, not patched up.
        #
        # ⚠ The tokenizer bug is NOT the whole 95.75 WER -- umlaut words are only 7.5% of test words,
        # so it explains at most ~7.5 points (§104d retracts my first over-claim here). The rest is
        # the code-switching that deText-only/deText+enAudio-enSpm+C address. This arm isolates the tokenizer fix alone.
        from .german_xling import train_german_arm_b

        train_german_arm_b(
            prefix=f"{prefix}/german",
            winner_model=winner_model,
            budget=GERMAN_BUDGET,
            smoke=GERMAN_SMOKE,
            fix_text_spm=True,
            german_dev=True,
        )

    if "deText+deAudio" in _german_stages:
        # 🔴 Arm B+C (user call, 2026-09-20) -- the replacement for deText-only, and the plan's optional
        # 4th arm: German paired audio AND German text injection, from the surgered checkpoint.
        #
        # It answers deText-only's question without touching RETURNN. deText-only wanted to delete the English
        # audio branch because language was perfectly predictable from feature type (real audio =>
        # English, pseudo audio => German), which is the cue arm B learned and code-switches on. Arm
        # B+C removes that cue just as completely -- no English audio remains -- while keeping the
        # tensors non-degenerate, so none of the three empty-stream defects can fire.
        #
        # It is also the better system and the natural "does injection add anything on top of the
        # audio we already have" ablation against arm C.
        #
        # ⚠ Carries the §104 SPM fix, like deText+enAudio -- an arm that cannot emit an umlaut answers nothing.
        from .german_xling import train_german_arm_b

        train_german_arm_b(
            prefix=f"{prefix}/german",
            winner_model=winner_model,
            budget=GERMAN_BUDGET,
            smoke=GERMAN_SMOKE,
            german_audio=True,
            fix_text_spm=True,
            german_dev=True,
        )

    if "deText+deAudio-long" in _german_stages:
        # deText+deAudio trained 4x longer (user, 2026-09-22): nEp10 saw only ~13% of the injection text
        # (10 x 1/75) and its dev_de curve was flat from ep6 without overfitting. Two audio doses:
        # repeat 1 = 40 German-audio passes (same total as the nEp10 run), repeat 2 = 80 passes.
        # ⚠ Two things change with nep: text coverage AND specaugment, whose step-keyed ramp
        # (1850, 5550, 9250) never completed in the 3,528-update nEp10 run but does at ~14k updates.
        from .german_xling import train_german_arm_b

        for _rep in (1, 2):
            train_german_arm_b(
                prefix=f"{prefix}/german",
                winner_model=winner_model,
                budget=GERMAN_BUDGET,
                smoke=GERMAN_SMOKE,
                german_audio=True,
                german_audio_repeat=_rep,
                fix_text_spm=True,
                german_dev=True,
                nep=40,
            )

    if "enarpa" in _german_stages:
        # German phone prior from the english_us_arpa aligner (german_xling `_DE_TABLES`, user 2026-09-22):
        # one procedure for both budgets, no audio beyond the pipeline's. Adds runs; the "selftrained"
        # arms stay for comparison. 1 h arms use the same recipes as 9 h (deText+deAudio repeat 4 =
        # deAudio's 40 passes, the PatchAsrBranchToGerman rule). deAudio 9 h is not re-run: it never
        # reads the pseudo table (the table only initialises the unused pseudo_enc rows).
        from .german_xling import train_german_arm_b, train_german_arm_c

        for _b in ("1h", "9h"):
            train_german_arm_b(
                prefix=f"{prefix}/german",
                winner_model=winner_model,
                budget=_b,
                smoke=GERMAN_SMOKE,
                german_audio=True,
                fix_text_spm=True,
                german_dev=True,
                prior="enarpa",
            )
        train_german_arm_c(prefix=f"{prefix}/german", winner_model=winner_model, budget="1h", prior="enarpa")

    if "deText-only" in _german_stages:
        # Arm B-zero (user call, 2026-09-20): **no paired audio at all** -- the pseudo-encoder alone.
        #
        # Why: arm B scored 95.75 WER while demonstrably knowing German words, code-switching into
        # English on real German audio (`ALLES WAS ICH INSIDE AN IS DASS WHEN UNTER`). With the English
        # ASR branch present, language is perfectly predictable from FEATURE TYPE -- real audio always
        # meant English, pseudo-audio always meant German -- and real German audio is a combination the
        # model never saw. Dropping the audio branch removes that cue entirely, so German is the only
        # target language in training. The plan called the English branch "the real-audio anchor" and
        # rejected `ls_audio_subset == 0` without ever running it; this runs it.
        #
        # Also carries the §104 tokenizer fix, without which the text branch would again feed German
        # through the ENGLISH SPM and never emit an umlaut. Two changes at once, deliberately: arm B as
        # it stands is broken, so reproducing its bug in a new arm would buy nothing.
        from .german_xling import train_german_arm_b

        train_german_arm_b(
            prefix=f"{prefix}/german",
            winner_model=winner_model,
            budget=GERMAN_BUDGET,
            smoke=GERMAN_SMOKE,
            no_audio=True,
            fix_text_spm=True,
            german_dev=True,
        )

    if "deAudio" in _german_stages:
        # Arm C -- the BASELINE: MLS-de paired audio at the same budget, no text injection.
        # This is the plan's one controlled comparison (B vs C); see train_german_arm_c for the
        # list of everything held fixed against arm B.
        from .german_xling import train_german_arm_c

        train_german_arm_c(
            prefix=f"{prefix}/german", winner_model=winner_model, budget=GERMAN_BUDGET, smoke=GERMAN_SMOKE_ARMC
        )

        # Arm C sweep (backlog 82/90/92a). The baseline must get its BEST shot, or "arm B beats arm C"
        # is an artefact of arm C's config rather than a result -- and §78 showed the pilot's config is
        # knowingly suboptimal (memorises by epoch 3; peak LR inherited from arm B, whose data stream
        # is ~100x larger).
        # ⚠ `no_recog=True`: these arms are selected on the FREE `learning_rates` dev curve, which §92a
        # proved ranks epochs exactly as WER does. Only the winner gets a real recog afterwards --
        # otherwise the sweep spends ~9 h of GPU decoding to choose between 50-minute trainings.
        # `enc_lr_mult` down-weights the ENCODER only: lowering the global LR would also slow the
        # output layer, which is the one part that genuinely must learn (3 brand-new symbols).
        for _c_lr in ARM_C_SWEEP_PEAK_LRS:
            for _c_enc in ARM_C_SWEEP_ENC_MULTS:
                train_german_arm_c(
                    prefix=f"{prefix}/german-sweep",
                    winner_model=winner_model,
                    budget=GERMAN_BUDGET,
                    peak_lr=_c_lr,
                    enc_lr_mult=_c_enc,
                    nep=ARM_C_SWEEP_NEP,
                    no_recog=True,
                )

    if "zeroshot" in _german_stages:
        # Arm A: winner zero-shot on MLS-de test. Same recog helper and the same winner model object
        # as the English table above -- only `task` differs, which is what makes it a clean control.
        from .german_xling import get_mls_de_task

        _de_task = get_mls_de_task(extended_vocab=False)
        _de_res = _ctc_only_recog_batched(
            prefix=f"{prefix}/german/zeroshot-winner-mls-de",
            task=_de_task,
            ctc_model=ctc_lm_kwargs["ctc_model"],
            aux_ctc_layer=ctc_lm_kwargs["aux_ctc_layer"],
            num_shards=ctc_lm_kwargs["num_shards"],
            extra_config=ctc_lm_kwargs.get("extra_config"),
        )
        from i6_experiments.users.dorian_koch.speech_llm.result_notify import notify_result

        notify_result(
            "german-zeroshot-winner",
            {"mls_de_test_ctc": _de_res.output},
            note=(
                "Arm A: English winner, UNMODIFIED vocab, zero-shot on MLS-de test."
                " CONTEXT ONLY -- the 10,240-piece English SPM cannot write German orthography"
                " (no umlauts), so a catastrophic WER here is expected and is not a baseline."
            ),
        )

    if DLM_DATA_STAGE != "off":
        from .dlm_on_winner import get_dlm_task_on_winner

        _dlm_task, jobs = get_dlm_task_on_winner(
            hyps_model=ctc_lm_kwargs["ctc_model"],
            extra_config=ctc_lm_kwargs.get("extra_config"),
            alias_prefix=f"{prefix}/dlm-data",
        )
        _dlm_hyp_jobs[:] = jobs
        _dlm_task_ref[:] = [_dlm_task]
        # smoke: bundle 01 = 2 LS-960 passes (ogg audio) + 6 LM-text shards (GlowTTS path), i.e. both code paths
        register = {1: jobs[1]} if DLM_DATA_STAGE == "smoke" else dict(enumerate(jobs))
        for b, job in register.items():
            for key, outs in job.out_files.items():
                for fn, path in outs.items():
                    tk.register_output(f"{prefix}/dlm-data/hyps-batched-{b:02d}/{key}/{fn}", path)
        if DLM_DATA_STAGE == "train":
            from .dlm_on_winner import train_paper_best_dlm_4gpu

            train_paper_best_dlm_4gpu(_dlm_task)

    # Continue training the winner with TTS audio added (user request, 2026-09-17). Independent of the DLM
    # line above: it only adds jobs, and it touches none of tts_data's module state, so the hypothesis
    # bundles now running keep their hashes.
    if WINNER_PLUS_TTS:
        from .winner_plus_tts import train_winner_plus_tts

        train_winner_plus_tts(prefix=f"{prefix}/winner-plus-tts", winner_model=winner_model)

    # The two resumed arms' ModelWithCheckpoints, for the follow-up evals below (hyps pass, our DLM).
    # Capturing a return value cannot re-hash anything.
    _resumed_exps = {}

    if WINNER_PLUS_TTS_RESUMED:
        # Same arm, same data, same model def -- only the training JOB CLASS differs, so the winner's
        # optimizer moments are loaded instead of being re-estimated from zero. See
        # resumed_training.py: RETURNN has no "import the optimizer" option because its answer is
        # resumption, gated on the .opt.pt sibling being present (engine/base.py:101-107).
        from .winner_plus_tts import train_winner_plus_tts

        _resumed_exps["plusTts"] = train_winner_plus_tts(
            prefix=f"{prefix}/winner-plus-tts",
            winner_model=winner_model,
            resume_from_winner=True,
            nep=WINNER_PLUS_TTS_RESUMED_NEP,
            flat_lr=WINNER_PLUS_TTS_RESUMED_LR,
        )

    if WINNER_CONT_NO_TTS:
        # The ablation: byte-identical to the resumed +TTS arm above except `with_tts=False`, which
        # skips `_PatchAsrBranchWithTts` and leaves the `asr` branch as the winner's own LS-960
        # OggZip. Same resume donor, same nep, same flat LR, so the pair differs in one factor.
        from .winner_plus_tts import train_winner_plus_tts

        _resumed_exps["contNoTts"] = train_winner_plus_tts(
            prefix=f"{prefix}/winner-plus-tts",
            winner_model=winner_model,
            resume_from_winner=True,
            nep=WINNER_PLUS_TTS_RESUMED_NEP,
            flat_lr=WINNER_PLUS_TTS_RESUMED_LR,
            with_tts=False,
        )

    # The cold-start no-TTS ablation, see WINNER_COLD_NO_TTS. Same call as the WINNER_PLUS_TTS block above.
    # Collected with the resumed arms for the follow-up evals below (plain CTC, hyps pass); our-DLM recogs
    # are added in the dlm-ours block.
    _final_exps = dict(_resumed_exps)
    if WINNER_COLD_NO_TTS:
        from .winner_plus_tts import train_winner_plus_tts

        _final_exps["coldNoTts"] = train_winner_plus_tts(
            prefix=f"{prefix}/winner-plus-tts", winner_model=winner_model, with_tts=False
        )

    # Plain CTC for the two resumed arms at their final epoch -- the one row of the LS paper table that
    # train_winner_plus_tts does not produce (same call as the winner's / the cold start's ctc-only row).
    for _tag, _exp in _final_exps.items():
        import dataclasses as _dcx

        _ctc_only_recog_batched(
            prefix=f"{prefix}/winner-plus-tts/ctc-only-{_tag}-final",
            task=ctc_lm_kwargs["task"],
            ctc_model=_dcx.replace(ctc_lm_kwargs["ctc_model"], checkpoint=_exp.get_last_fixed_epoch().checkpoint),
            aux_ctc_layer=ctc_lm_kwargs["aux_ctc_layer"],
            num_shards=ctc_lm_kwargs["num_shards"],
            extra_config=ctc_lm_kwargs.get("extra_config"),
        )

    if FT_HYPS_WER:
        # The measurement the +TTS finetune was actually built to move: hypothesis-pass WER on REAL
        # LS-960 audio and on GlowTTS audio. Reference pair (winner_plus_tts.py:7-9, random 20k seed 0,
        # same augmentation): winner 2.22% / 14.53%, old RZ CTC 5.02% / 10.25%.
        #
        # 🔴 Do NOT substitute the LS dev/test recog numbers for the "LS audio" cell. Those are
        # dev/test, unaugmented; this is a random 20k of LS-960 *train* decoded under
        # GetCtcHypsCfgV6's dropout/specaug/mixup. Different population -- conflating them is how the
        # 2.01% misreading in backlog R1 happened.
        #
        # Cost control: `get_dlm_task_on_winner` builds the whole 12-bundle DLM-data pass, but
        # Sisyphus only RUNS what a registered output needs, so registering bundle 01 alone leaves the
        # other 11 unbuilt. Bundle 01 is the right one: `DLM_DATA_STAGE == "smoke"` already uses it
        # because it carries both code paths -- 2 LS-960 ogg-audio passes and 6 LM-text/GlowTTS shards
        # -- so one 4-GPU job yields both halves of the table. Each item is a FULL pass (LS 281,241
        # seqs; text 538,910), not a shard, so the seed-0 20k sample is a proper random sample of the
        # population rather than a slice of one length-sorted shard (backlog R1 again).
        import dataclasses as _dc
        from i6_core.returnn.training import PtCheckpoint as _PtCkpt
        from .dlm_on_winner import get_dlm_task_on_winner as _get_dlm_task
        from .winner_plus_tts import FINETUNE_TRAIN_JOB as _FT_JOB

        _ft_hyps_model = _dc.replace(
            ctc_lm_kwargs["ctc_model"],
            checkpoint=_PtCkpt(pinned_path(f"{_FT_JOB}/output/models/epoch.{FT_HYPS_EPOCH:03d}.pt")),
        )
        _, _ft_hyp_jobs = _get_dlm_task(
            hyps_model=_ft_hyps_model,
            # Same recog model config as the winner's own hypothesis pass -- `pseudo_speech_enc` and
            # friends. Omitting it builds the GlowTTS variant against a pseudo-encoder checkpoint
            # (see the ep-eval note below, BatchedReturnnForwardJob.g1vY1m2utxfM).
            extra_config=ctc_lm_kwargs.get("extra_config"),
            alias_prefix=f"{prefix}/dlm-data-ft",
        )
        # 🔴 The UNCONTAMINATED re-measurement (backlog 112). Bundle 01's six GlowTTS shards are
        # parts 3-62, and the finetune trained on the audio of parts 1-100 -- so its 7.42% GlowTTS
        # number was measured on text it had already heard spoken, while the winner (which saw no
        # TTS audio at all) was not. The confound runs in the direction that flatters our arm.
        # Bundle 03 is parts 143-222: disjoint from 1-100, so this pair is clean. The WINNER's
        # bundle 03 already exists on disk (`dlm-data/hyps-batched-03`), so only the finetune side
        # costs a job. ⚠ Bundle 02 is parts 63-142 and straddles the boundary -- do not use it.
        for _key, _outs in _ft_hyp_jobs[FT_HYPS_CLEAN_BUNDLE].out_files.items():
            for _fn, _path in _outs.items():
                tk.register_output(f"{prefix}/dlm-data-ft/hyps-batched-{FT_HYPS_CLEAN_BUNDLE:02d}/{_key}/{_fn}", _path)

        for _key, _outs in _ft_hyp_jobs[1].out_files.items():
            for _fn, _path in _outs.items():
                tk.register_output(f"{prefix}/dlm-data-ft/hyps-batched-01/{_key}/{_fn}", _path)

        # The same clean bundle-03 pass for the two RESUMED arms at their final epoch (the checkpoint
        # their LS recogs use), so "+TTS resumed" vs "keep training, no TTS" is compared on the metric
        # the TTS arm exists to move -- their LS dev/test tables cannot answer that. Winner reference:
        # `dlm-data/hyps-batched-03`, same seeded sample via analysis/tts_hyps_wer.py.
        for _tag, _exp in _final_exps.items():
            _m = _dc.replace(ctc_lm_kwargs["ctc_model"], checkpoint=_exp.get_last_fixed_epoch().checkpoint)
            _, _jobs = _get_dlm_task(
                hyps_model=_m,
                extra_config=ctc_lm_kwargs.get("extra_config"),
                alias_prefix=f"{prefix}/dlm-data-ft-{_tag}",
            )
            for _key, _outs in _jobs[FT_HYPS_CLEAN_BUNDLE].out_files.items():
                for _fn, _path in _outs.items():
                    tk.register_output(
                        f"{prefix}/dlm-data-ft-{_tag}/hyps-batched-{FT_HYPS_CLEAN_BUNDLE:02d}/{_key}/{_fn}", _path
                    )

    for _ep in EVAL_FINETUNE_EPOCHS:
        # Same plain-CTC recog as the winner's `ctc-only-batched` row, on the finetune's epoch-N
        # checkpoint, so the two numbers are directly comparable (winner: 1.78/3.95/1.92/4.34).
        import dataclasses
        from i6_core.returnn.training import PtCheckpoint
        from i6_experiments.users.dorian_koch.speech_llm.result_notify import notify_result
        from .winner_plus_tts import FINETUNE_TRAIN_JOB

        _ckpt = pinned_path(f"{FINETUNE_TRAIN_JOB}/output/models/epoch.{_ep:03d}.pt")
        _ft_model = dataclasses.replace(ctc_lm_kwargs["ctc_model"], checkpoint=PtCheckpoint(_ckpt))
        _ctc_res = _ctc_only_recog_batched(
            prefix=f"{prefix}/winner-plus-tts/ctc-only-ep{_ep:03d}",
            task=ctc_lm_kwargs["task"],
            ctc_model=_ft_model,
            aux_ctc_layer=ctc_lm_kwargs["aux_ctc_layer"],
            num_shards=ctc_lm_kwargs["num_shards"],
            extra_config=ctc_lm_kwargs.get("extra_config"),
        )
        # The winner's BEST configuration (1.36/2.93/1.56/3.24): CTC+AED+LM label-sync first-pass
        # search, no prior. Same call as exp2026_05_28_tts_encoder_fzj.py:3064, only the checkpoint
        # differs, so the numbers are directly comparable.
        from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_lm_batched import (
            ctc_aed_lm_label_sync_recog_auto_scale_batched,
        )

        # 🔴 `pseudo_speech_enc` is REQUIRED here, and its absence is what broke this eval on
        # 2026-09-18 (BatchedReturnnForwardJob.g1vY1m2utxfM, worker exit 1):
        #   Unexpected key(s): 'pseudo_enc.embedding.weight', ...
        #   Missing key(s):    'tts.glow_tts_model.*'   (563 keys)
        # `aed_glowtts_model_def` branches on `config.bool("pseudo_speech_enc", False)`: without it the
        # recog builds the **GlowTTS** variant while the checkpoint is a **pseudo-encoder** model.
        # Plain CTC was unaffected because it never constructs the AED/TTS half.
        # ⚠ This one key is sufficient: every *shape*-affecting key the model_def reads
        # (`pseudo_enc_units`, `pseudo_enc_phone_states`, `pseudo_enc_channel_concat`,
        # `pseudo_enc_start_layer`) is left at its default by the training config too. The remaining
        # `pseudo_enc_*` keys there (durations, smoothing, specaug, the frozen table) only affect
        # training-time behaviour or an initialiser the checkpoint overwrites.
        _ls_extra_config = {**(ctc_lm_kwargs.get("extra_config") or {}), "pseudo_speech_enc": True}
        _ls_res = ctc_aed_lm_label_sync_recog_auto_scale_batched(
            prefix=f"{prefix}/winner-plus-tts/ctc+aed+lm-labelsync-ep{_ep:03d}",
            task=ctc_lm_kwargs["task"],
            aed_ctc_model=_ft_model,
            lm=ctc_lm_kwargs["lm"],
            aux_ctc_layer=ctc_lm_kwargs["aux_ctc_layer"],
            num_shards=ctc_lm_kwargs["num_shards"],
            extra_config=_ls_extra_config,
        )
        # "Never miss a result" sink: a mini_task that fires exactly when these recogs finish,
        # writing output/RESULTS/<tag> + a line in RESULTS.jsonl, so a landed WER cannot sit unread.
        # This setup had no sink at all; the job lives in our own tree, so it is a free import.
        # 🔴 ONE notification PER eval, deliberately not one bundling both.
        # The first version passed {"ctc_only": ..., "ctc_aed_lm_labelsync": ...} together, so the
        # sink depended on BOTH outputs -- and when the label-sync eval failed, the sink could never
        # run, suppressing the plain-CTC number that HAD landed. That is exactly the failure this sink
        # exists to prevent, reintroduced by bundling. Keep them independent.
        _baselines = "Winner baselines (dev-clean/dev-other/test-clean/test-other):"
        notify_result(
            f"winner-plus-tts-ep{_ep:03d}-ctc-only",
            {"ctc_only": _ctc_res.output},
            note=f"winner+TTS finetune epoch {_ep}, plain CTC. {_baselines} 1.78/3.95/1.92/4.34.",
        )
        notify_result(
            f"winner-plus-tts-ep{_ep:03d}-labelsync",
            {"ctc_aed_lm_labelsync": _ls_res.output},
            note=(f"winner+TTS finetune epoch {_ep}, CTC+AED+LM label-sync. {_baselines} 1.36/2.93/1.56/3.24."),
        )

    # End-to-end check (2026-09-16): the same winner recogs with the 729M DLM (LfAv45zfWLtF).
    # The builder looks up _get_imported_dlm at call time, so swapping it makes the DLM-sum recogs
    # the only new jobs; everything else resolves to the identical (imported) jobs.
    from i6_experiments.users.zeyer.experiments import exp2026_05_28_tts_encoder_fzj as _fzj

    with unittest.mock.patch.object(_fzj, "_get_imported_dlm", lambda: _get_dlm(DLM_N1280, model_dim=1280)):
        _train_winner(prefix + "/dlm-n1280")

    # The same DLM-sum recogs against the DLM WE TRAINED here (backlog 87), so that training finally
    # has a consumer. Identical mechanism to the block above -- only the checkpoint differs -- which
    # also makes the two directly comparable: imported n1280 (LfAv45zfWLtF, nEp200) vs ours
    # (pMb0YjIfsID0) on the winner's own hypotheses.
    if EVAL_OUR_TRAINED_DLM:
        with unittest.mock.patch.object(_fzj, "_get_imported_dlm", lambda: _get_dlm(OUR_TRAINED_DLM, model_dim=1280)):
            _train_winner(prefix + f"/dlm-ours-ep{OUR_TRAINED_DLM_EPOCH:03d}")

        # The same swap for the two resumed arms, so the DLM-sum rows of "+TTS resumed" and "keep
        # training, no TTS" use OUR DLM like the winner's row above. Only the DLM-sum recogs are new;
        # the trainings and non-DLM recogs resolve to the identical jobs of the calls above.
        from .winner_plus_tts import train_winner_plus_tts

        with unittest.mock.patch.object(_fzj, "_get_imported_dlm", lambda: _get_dlm(OUR_TRAINED_DLM, model_dim=1280)):
            if WINNER_COLD_NO_TTS:
                train_winner_plus_tts(
                    prefix=f"{prefix}/dlm-ours-ep{OUR_TRAINED_DLM_EPOCH:03d}/winner-plus-tts",
                    winner_model=winner_model,
                    with_tts=False,
                )
            if WINNER_PLUS_TTS:
                # The cold-start +TTS finetune (EXsiZj08AB1C), so the paper table has our DLM in every
                # column. Only its DLM-sum recogs are new.
                train_winner_plus_tts(
                    prefix=f"{prefix}/dlm-ours-ep{OUR_TRAINED_DLM_EPOCH:03d}/winner-plus-tts", winner_model=winner_model
                )
            for _tag in _resumed_exps:
                train_winner_plus_tts(
                    prefix=f"{prefix}/dlm-ours-ep{OUR_TRAINED_DLM_EPOCH:03d}/winner-plus-tts",
                    winner_model=winner_model,
                    resume_from_winner=True,
                    nep=WINNER_PLUS_TTS_RESUMED_NEP,
                    flat_lr=WINNER_PLUS_TTS_RESUMED_LR,
                    with_tts=_tag == "plusTts",
                )

    if LOQ_EVAL_WINNER:
        _loq_eval_winner(prefix=f"{prefix}/loq-eval", ctc_lm_kwargs=ctc_lm_kwargs)

    if LOQ_EVAL_ALBERT_ASR:
        _loq_eval_albert_asr(prefix=f"{prefix}/loq-eval-albert")


def _loq_eval_albert_asr(*, prefix: str):
    """Albert's best Loquacious ASR (`LOQ_BEST_ASR_NAME`, ReturnnTrainingJob.BiRxyzqSYEig ep142) with the
    Loquacious DLM: AED+CTC (his headline), CTC+Loq-LM, CTC+AED+Loq-LM label-sync, and CTC(+AED)+DLM-sum.

    The task / model / aux CTC layer are taken verbatim from his builder
    (``exp2026_05_28_tts_encoder_fzj._train_loquacious_baselines``) by recording the kwargs of its AED+CTC
    recog call, so nothing is re-derived. ⚠ ``tk.register_output`` is stubbed during that call: the builder
    defines ~20 Loquacious trainings, and registering their outputs would put them all into our graph as
    unfinished jobs. Only what the recogs below depend on enters the graph; those are Albert's finished
    jobs, linked via ``import_albert_jobs("symlink")``.
    """
    import unittest.mock
    from sisyphus import tk as _tk
    from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
    from i6_experiments.users.zeyer.experiments import exp2026_05_28_tts_encoder_fzj as _fzj
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext import aed_ctc_batched as _aed_ctc
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_lm_batched import (
        ctc_recog_recomb_labelwise_prior_auto_scale_batched,
        ctc_aed_lm_label_sync_recog_auto_scale_batched,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.dlm_sum_batched import (
        ctc_dlm_sum_recog_auto_scale_batched,
        aed_ctc_dlm_sum_recog_auto_scale_batched,
    )

    recorded: Dict[str, Dict[str, Any]] = {}
    real_aed_ctc = _aed_ctc.aed_ctc_timesync_recog_recomb_auto_scale_batched

    def _record(**kwargs):
        recorded[kwargs["prefix"]] = kwargs

    with (
        unittest.mock.patch.object(_aed_ctc, "aed_ctc_timesync_recog_recomb_auto_scale_batched", _record),
        unittest.mock.patch.object(_tk, "register_output", lambda *_a, **_k: None),
    ):
        _fzj._train_loquacious_baselines(prefix=get_setup_prefix_for_module(_fzj.__name__))
    (kw,) = [v for k, v in recorded.items() if k.endswith(f"/loq/aed/{LOQ_BEST_ASR_NAME}/aed+ctc-batched")]
    task, asr_model, aux_ctc_layer = kw["task"], kw["aed_ctc_model"], kw["aux_ctc_layer"]
    loq_prior = _fzj._get_loq_transcription_labelwise_prior("spm10k", task)
    loq_dlm = _get_dlm(LOQ_DLM, model_dim=1024)
    p = f"{prefix}/{LOQ_BEST_ASR_NAME}"
    _common = dict(task=task, aux_ctc_layer=aux_ctc_layer, num_shards=8)

    real_aed_ctc(prefix=f"{p}/aed+ctc-batched", aed_ctc_model=asr_model, **_common)
    ctc_recog_recomb_labelwise_prior_auto_scale_batched(
        prefix=f"{p}/ctc+lm-batched", ctc_model=asr_model, lm=_fzj._get_loq_lm(), labelwise_prior=loq_prior, **_common
    )
    ctc_aed_lm_label_sync_recog_auto_scale_batched(
        prefix=f"{p}/ctc+aed+lm-labelsync-batched", aed_ctc_model=asr_model, lm=_fzj._get_loq_lm(), **_common
    )
    for fn, name in (
        (ctc_dlm_sum_recog_auto_scale_batched, "ctc+dlm-sum-batched"),
        (aed_ctc_dlm_sum_recog_auto_scale_batched, "ctc+aed+dlm-sum-batched"),
    ):
        fn(prefix=f"{p}/loq-dlm/{name}", asr_model=asr_model, dlm=loq_dlm, labelwise_prior=loq_prior, **_common)


def _loq_eval_winner(*, prefix: str, ctc_lm_kwargs: Dict[str, Any]):
    """The winner (an LS model) on the four Loquacious eval subsets: plain CTC, CTC+LS-LM, and DLM-sum with
    the imported and with our DLM. Out-of-domain, exactly the protocol of Albert's
    ``denoising_lm_2024/sis_recipe/ood_exps.py`` (same ``get_loquacious_eval_task_raw``, LS spm10k vocab,
    LS LM, LS labelwise prior; scales tuned per subset on its own dev split), so the numbers sit next to
    his RZ table for the old CTC models (e.g. voxpopuli CTC+LM 15.75/15.2 vs LS-DLM-sum 15.9/15.62).

    ⚠ Our LS DLMs are vocab-compatible ONLY with LS-spm10k ASR models. Albert's Loquacious ASR models use
    the Loquacious SPM (``TrainSentencePieceJob.SZcvHsG1gYNM``) -- they need a Loquacious DLM instead.
    """
    from i6_experiments.users.zeyer.datasets.loquacious import EvalSubSplits
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_lm_batched import (
        ctc_recog_recomb_labelwise_prior_auto_scale_batched,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.dlm_sum_batched import (
        ctc_dlm_sum_recog_auto_scale_batched,
        aed_ctc_dlm_sum_recog_auto_scale_batched,
    )
    from denoising_lm_2024.sis_recipe.ood_exps import get_loquacious_eval_task_raw

    _common = dict(
        aux_ctc_layer=ctc_lm_kwargs["aux_ctc_layer"],
        num_shards=ctc_lm_kwargs["num_shards"],
        extra_config=ctc_lm_kwargs.get("extra_config"),
    )
    _dlms = {
        "dlm-n1280": lambda: _get_dlm(DLM_N1280, model_dim=1280),
        f"dlm-ours-ep{OUR_TRAINED_DLM_EPOCH:03d}": lambda: _get_dlm(OUR_TRAINED_DLM, model_dim=1280),
    }
    for subset in EvalSubSplits:
        task = get_loquacious_eval_task_raw(eval_set_name=subset)
        p = f"{prefix}/{subset}"
        _ctc_only_recog_batched(
            prefix=f"{p}/ctc-only-batched", task=task, ctc_model=ctc_lm_kwargs["ctc_model"], **_common
        )
        ctc_recog_recomb_labelwise_prior_auto_scale_batched(
            prefix=f"{p}/ctc+lm-batched",
            task=task,
            ctc_model=ctc_lm_kwargs["ctc_model"],
            lm=ctc_lm_kwargs["lm"],
            labelwise_prior=ctc_lm_kwargs["labelwise_prior"],
            **_common,
        )
        for dlm_name, get_dlm in _dlms.items():
            for fn, name in (
                (ctc_dlm_sum_recog_auto_scale_batched, "ctc+dlm-sum-batched"),
                (aed_ctc_dlm_sum_recog_auto_scale_batched, "ctc+aed+dlm-sum-batched"),
            ):
                fn(
                    prefix=f"{p}/{dlm_name}/{name}",
                    task=task,
                    asr_model=ctc_lm_kwargs["ctc_model"],
                    dlm=get_dlm(),
                    labelwise_prior=ctc_lm_kwargs["labelwise_prior"],
                    **_common,
                )


def _ctc_only_recog_batched(
    *,
    prefix: str,
    task,
    ctc_model,
    aux_ctc_layer: Optional[int],
    num_shards: int,
    extra_config: Optional[Dict[str, Any]] = None,
    beam_size: int = 64,
    ctc_soft_collapse_threshold: Optional[float] = 0.8,
):
    """
    Plain CTC first pass: time-synchronous recombination beam search, no LM, no prior.

    The first pass of ``ctc_lm_batched.ctc_recog_recomb_labelwise_prior_auto_scale_batched`` with the LM
    left out: same search def (``recog_ext.ctc.model_recog_with_recomb``, which skips LM and prior when
    ``model.lm`` is None), same base config, beam, soft collapse and beam-scaled batch size.
    """
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc_batched import (
        _combined_recog_batched,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc import model_recog_with_recomb

    config: Dict[str, Any] = {
        "behavior_version": 24,
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "recog_recomb": "max",
        "recog_version": 10,
        "aux_loss_layers": [aux_ctc_layer] if aux_ctc_layer is not None else [],
    }
    if extra_config:
        config = dict_update_deep(config, extra_config)
    if ctc_soft_collapse_threshold is not None:
        config.update(
            {"ctc_soft_collapse_threshold": ctc_soft_collapse_threshold, "ctc_soft_collapse_reduce_type": "max_renorm"}
        )
    config["beam_size"] = beam_size
    config["batch_size"] = int(40_000 * ctc_model.definition.batch_size_factor * min(32 / beam_size, 1))
    score = _combined_recog_batched(
        prefix=prefix,
        task=task,
        model=ctc_model,
        config=config,
        num_shards=num_shards,
        recog_def=model_recog_with_recomb,
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", score.output)
    return score


def _train_winner(prefix: str):
    """The ``_sa == 50`` call of the specaug loop in ``exp2026_05_28_tts_encoder_fzj.py::py``, verbatim.

    Returns the ``ModelWithCheckpoints`` so a follow-up experiment can init from a real checkpoint of it
    (see ``winner_plus_tts``) rather than naming a path.
    """
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon
    from i6_experiments.users.zeyer.datasets.hf_librispeech_mfa_alignments import (
        get_mfa_phone_mean_logmel_table,
        get_mfa_phone_duration_table,
    )
    from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import _train_tts_encoder

    _sa = 50
    return _train_tts_encoder(
        WINNER_NAME,
        prefix=prefix,
        with_ctc_lm_recog=True,
        text_train_epoch_split=75,
        batch_size_audio_frames=70_000,
        batch_size_phon=6_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
        pseudo_enc_duration_table=get_mfa_phone_duration_table().out_duration_table,
        pseudo_enc_duration_sigma=0.45,
        pseudo_enc_duration_scale=0.7,
        pseudo_enc_max_len_factor=10,
        train_seq_ordering="random",
        pseudo_enc_lerp=True,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        glow_tts_add_silence_between_words=0.15,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        behavior_version=29,  # packed tensors need >= 29
        pseudo_enc_frontend_concat=True,
        extra_config_updates={
            "optimizer.class": rf.build_dict(Muon)["class"],
            "packed_tensors": True,
            "torch_distributed": {"reduce_type": "grad_explicit"},
            "batch_size": None,
            "packed_batch_size": {"data": 11_200_000, "classes": 5_000, "phonemes": 6_000},
            "batching": "random",
            "torch_cuda_graph": {
                "batch_size_bound": 500,
                "dim_capacity": {"data": 312_000, "classes": 80, "phonemes": 300},
                "warmup_steps": 0,
                "compile": True,
            },
            "optimizer.weight_decay": 0.027,  # 0.01 / 0.370
            "specaugment_num_spatial_mask_factor": _sa,
            "specaugment_steps": (1850, 5550, 9250),  # (5000, 15000, 25000) * 0.370
        },
        extra_config_deletes=["optimizer.epsilon"],
    )


def import_albert_jobs(mode: str = "dryrun"):
    """Link Albert's finished jobs into our work/ (mode: dryrun | symlink | copy | hardlink)."""
    tk.import_work_directory(ALBERT_WORK_DIRS, mode=mode)


def report(max_listed: int = 60):
    """Print, for every job in the loaded graph: whether it exists in our work/ and is finished."""
    import os

    jobs = tk.sis_graph.jobs()
    counts: Dict[str, int] = {}
    missing = []
    for job in jobs:
        setup, finished = job._sis_setup(), job._sis_setup() and job._sis_finished()
        linked = os.path.islink(job._sis_path())
        key = (
            "finished" + (" (symlink)" if linked else "") if finished else ("setup, unfinished" if setup else "absent")
        )
        counts[key] = counts.get(key, 0) + 1
        if not finished:
            missing.append(job)
    print(f"graph: {len(jobs)} jobs")
    for key, n in sorted(counts.items()):
        print(f"  {n:6d}  {key}")

    winner = [j for j in jobs if j._sis_id() == WINNER_TRAIN_JOB]
    print(f"winner training job {WINNER_TRAIN_JOB}: " + ("IN GRAPH" if winner else "NOT IN GRAPH (hash mismatch?)"))
    if winner:
        w = winner[0]
        models = os.path.join(w._sis_path(), "output", "models")
        present = sorted(os.listdir(models)) if os.path.isdir(models) else []
        print(f"  finished={w._sis_setup() and w._sis_finished()}  checkpoints={present}")

    print(f"not finished here ({len(missing)}), first {max_listed}:")
    for job in sorted(missing, key=lambda j: j._sis_id())[:max_listed]:
        aliases = sorted(job.get_aliases() or [])
        print(f"  {job._sis_id()}  {aliases[0] if aliases else ''}")


def report_outputs(name_substr: str = "", max_bytes: int = 2000):
    """Print every registered output: available or not, and the content of small ones (WER summaries).
    Works from `sis c` without a manager, i.e. without creating output/ links."""
    import os
    from sisyphus.graph import OutputPath

    for target in sorted(tk.sis_graph.targets, key=lambda t: t.name):
        if name_substr not in target.name or not isinstance(target, OutputPath):
            continue
        path = target._sis_path.get_path()
        ok = target._sis_path.available()
        line = f"{'OK ' if ok else 'MISSING'}  {target.name}"
        if ok and os.path.isfile(path) and os.path.getsize(path) <= max_bytes:
            with open(path) as f:
                line += "  " + " ".join(f.read().split())
        print(line)
