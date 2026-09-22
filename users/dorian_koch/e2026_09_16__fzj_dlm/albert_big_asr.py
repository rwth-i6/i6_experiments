"""Our DLM (and the matched baselines) on Albert's larger injection ASR, EncL24-DecL8.

His 2026-09-19 run ``...-specaug50-stepcomp-encL24-decL8`` (24 encoder / 8 decoder layers, text
injection, 246k updates) is the first ASR that beats the winner without an LM -- AED+CTC
1.51/3.26/1.60/3.53 vs the winner's 1.49/3.40/1.65/3.66 -- while WITH the LM the two are level:
CTC+AED+LM 1.36/2.90/1.52/3.24 vs 1.36/2.93/1.56/3.24 (his notes, 2026-09-22 07:45 entry).
Our DLM reaches 3.23 on the winner, i.e. it already buys what his extra size buys. The open question
this module answers: does the DLM gain ADD to a stronger ASR, or do the two overlap?

⚠ This is a TRANSFER test, not a like-for-like swap: our DLM (``pMb0YjIfsID0``) was trained on the
WINNER's own hypotheses, so a null result means "does not transfer for free", never "the DLM is weak".

Feasible without any vocab work because his run uses the SAME SentencePiece model as ours
(``TrainSentencePieceJob.ofYcs4cMRS8T``, ``vocab_dim 10240``, read off his ``returnn.config``).
That is exactly what blocks the Loquacious ASR (``fzj_dlm.LOQ_EVAL_ALBERT_ASR``), and it does not
apply here.

Two deliberate construction choices:

* The recog model is the WINNER's own recog model with ONLY the two layer counts changed, so every
  other ``model_config`` entry is provably the one our pipeline already builds for the winner
  (the alternative -- calling his ``_train_tts_encoder`` with ``enc_num_layers=24`` -- would put a
  65 GPU-h TRAINING into our graph).
* His checkpoint is referenced as a RAW PATH inside HIS setup, pinned via ``pinned_path``, with no
  creator job -- same convention as ``OUR_TRAINED_DLM`` / ``FINETUNE_TRAIN_JOB``. His training is
  not ours to run, re-run, or depend on.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

# Albert's injection EncL24-DecL8 training (his notes name the job at launch: 3OhnXYDXQNO6), final
# epoch 38 of the nep38 schedule. Readable for us in his project-space setup copy; his $HOME setup
# is permission-denied, so do NOT switch this to an output/ alias path.
ALBERT_BIG_ASR_CKPT = (
    "/e/project1/spell/zeyer1/setups/2026-05-28-tts-encoder/work/i6_core/returnn/training"
    "/ReturnnTrainingJob.3OhnXYDXQNO6/output/models/epoch.038.pt"
)
ALBERT_BIG_ASR_ENC_LAYERS = 24
ALBERT_BIG_ASR_DEC_LAYERS = 8
# The winner's counts, asserted before patching so this never silently edits a different model.
WINNER_ENC_LAYERS = 16
WINNER_DEC_LAYERS = 6


def _big_asr_model(winner_ctc_model, ckpt_path):
    """The winner's recog model with his layer counts and his checkpoint."""
    from i6_core.returnn.training import PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces.model import ModelDefWithCfg
    from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import ModelWithCheckpoint

    definition = winner_ctc_model.definition
    assert isinstance(definition, ModelDefWithCfg), f"winner model def is {type(definition)}, expected ModelDefWithCfg"
    cfg = copy.deepcopy(definition.config)
    enc, dec = cfg["enc_build_dict"], cfg["dec_build_dict"]
    assert (enc["num_layers"], dec["num_layers"]) == (WINNER_ENC_LAYERS, WINNER_DEC_LAYERS), (
        f"expected the winner's {WINNER_ENC_LAYERS}/{WINNER_DEC_LAYERS} model,"
        f" got {enc['num_layers']}/{dec['num_layers']}"
    )
    # His model differs from the winner in exactly these two numbers (his `_train_tts_encoder`
    # `enc_num_layers` / `dec_num_layers` feed the same two dicts); everything else -- DbMel front
    # end, pseudo-encoder attachment, out_dim 1024, 8 heads -- is unchanged, verified against his
    # returnn.config (`"num_layers": 24` at :322, `"num_layers": 8, "model_dim": 1024` at :336).
    enc["num_layers"] = ALBERT_BIG_ASR_ENC_LAYERS
    dec["num_layers"] = ALBERT_BIG_ASR_DEC_LAYERS
    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(definition.model_def, cfg),
        checkpoint=PtCheckpoint(ckpt_path),
    )


def eval_albert_big_asr(
    *,
    prefix: str,
    ctc_lm_kwargs: Dict[str, Any],
    dlm,
    dlm_tag: str,
    ckpt_path,
):
    """The four decoders our paper table quotes, on his EncL24-DecL8 checkpoint.

    ``ctc_lm_kwargs`` are the winner's own recorded recog kwargs (task, lm, labelwise_prior,
    num_shards, extra_config), so every non-model ingredient is identical to the winner's row and
    the two columns are directly comparable.
    """
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc_batched import (
        aed_ctc_timesync_recog_recomb_auto_scale_batched,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_lm_batched import (
        ctc_aed_lm_label_sync_recog_auto_scale_batched,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.dlm_sum_batched import (
        ctc_dlm_sum_recog_auto_scale_batched,
        aed_ctc_dlm_sum_recog_auto_scale_batched,
    )
    from i6_experiments.users.dorian_koch.speech_llm.result_notify import notify_result

    model = _big_asr_model(ctc_lm_kwargs["ctc_model"], ckpt_path)
    task = ctc_lm_kwargs["task"]
    num_shards = ctc_lm_kwargs["num_shards"]
    prior = ctc_lm_kwargs.get("labelwise_prior")
    assert prior is not None, "winner recog passed no labelwise_prior; the DLM-sum recogs need it"
    # 🔴 `aux_loss_layers` must name HIS last encoder layer, or the recog builds the CTC head at
    # layer 16 while `aux_ctc_layer=24` reads layer 24 (his own recogs pass `[enc_num_layers]`, and
    # his training used [4, 10, 16, 24]). The `pseudo_speech_enc` keys come along in the winner's
    # extra_config -- his run is a pseudo-encoder model too, so they are required here as well
    # (dropping them is the 2026-09-18 "Missing key(s): tts.glow_tts_model.*" failure).
    extra_config = {
        **(ctc_lm_kwargs.get("extra_config") or {}),
        "aux_loss_layers": [ALBERT_BIG_ASR_ENC_LAYERS],
    }
    aux = ALBERT_BIG_ASR_ENC_LAYERS

    # (1) his headline LM-free number, which also proves our pipeline reproduces his 1.51/3.26/1.60/3.53
    _aed_ctc = aed_ctc_timesync_recog_recomb_auto_scale_batched(
        prefix=f"{prefix}/aed+ctc-batched",
        task=task,
        aed_ctc_model=model,
        aux_ctc_layer=aux,
        num_shards=num_shards,
        extra_config=extra_config,
    )
    # (2) his best LM decoder: the head-to-head against the winner's 1.36/2.93/1.56/3.24
    _labelsync = ctc_aed_lm_label_sync_recog_auto_scale_batched(
        prefix=f"{prefix}/ctc+aed+lm-labelsync-batched",
        task=task,
        aed_ctc_model=model,
        lm=ctc_lm_kwargs["lm"],
        aux_ctc_layer=aux,
        num_shards=num_shards,
        extra_config=extra_config,
    )
    # (3)+(4) the actual question: our DLM on his hypotheses
    _ctc_dlm = ctc_dlm_sum_recog_auto_scale_batched(
        prefix=f"{prefix}/{dlm_tag}/ctc+dlm-sum-batched",
        task=task,
        asr_model=model,
        dlm=dlm,
        labelwise_prior=prior,
        aux_ctc_layer=aux,
        num_shards=num_shards,
        extra_config=extra_config,
    )
    _aed_dlm = aed_ctc_dlm_sum_recog_auto_scale_batched(
        prefix=f"{prefix}/{dlm_tag}/ctc+aed+dlm-sum-batched",
        task=task,
        asr_model=model,
        dlm=dlm,
        labelwise_prior=prior,
        aux_ctc_layer=aux,
        num_shards=num_shards,
        extra_config=extra_config,
    )

    # One sink per eval, never bundled: a failed eval must not suppress a landed one (fzj_dlm.py:705).
    _ref = "winner: aed+ctc 1.49/3.40/1.65/3.66, ctc+aed+lm 1.36/2.93/1.56/3.24, ctc+aed+dlm(ours) 1.35/3.01/1.53/3.23"
    notify_result(
        "albert-big-asr-aed+ctc",
        {"aed_ctc": _aed_ctc.output},
        note=f"Albert EncL24-DecL8 ep38, AED+CTC. He reports 1.51/3.26/1.60/3.53. {_ref}",
    )
    notify_result(
        "albert-big-asr-labelsync",
        {"ctc_aed_lm_labelsync": _labelsync.output},
        note=f"Albert EncL24-DecL8 ep38, CTC+AED+LM. He reports 1.36/2.90/1.52/3.24. {_ref}",
    )
    notify_result(
        "albert-big-asr-ctc+dlm-ours",
        {"ctc_dlm_sum": _ctc_dlm.output},
        note=f"Albert EncL24-DecL8 ep38 + OUR DLM, CTC+DLM-sum (transfer test). {_ref}",
    )
    notify_result(
        "albert-big-asr-ctc+aed+dlm-ours",
        {"ctc_aed_dlm_sum": _aed_dlm.output},
        note=f"Albert EncL24-DecL8 ep38 + OUR DLM, CTC+AED+DLM-sum (transfer test). {_ref}",
    )
