"""Guard: the PersonaPlex TRAINING system-prompt prefix is what INFERENCE feeds, and the codes path works.

WHY. PersonaPlex is conditioned on a hybrid system prompt (voice + role text) that inference steps
through ``LMGen.step_system_prompts`` before the user speaks. Until 2026-09-21 training built a
different prefix -- audio code 0 on all 16 rows and no voice or silence segments -- so the model was
finetuned under a prompt it never sees at inference. Nothing in a loss curve shows that.

  1. PREFIX PARITY. Drive the REAL ``step_system_prompts`` (voice -> silence -> text -> silence) on a
     recording stub that captures every (text, agent, user) token it would feed, and require
     ``system_prompt_prefix`` to produce the identical [17, P] frames. Then rebuild the OLD code-0
     prefix and require it to FAIL the same comparison, so the guard is not vacuous.
  2. LOADER, codes corpus. A one-row arrow corpus in the podcast codes schema plus ``context`` and
     ``voice_codes`` columns goes through the real ``build_data_loader``: the prefix must equal (1),
     the loss mask must be 0 exactly over it, and the dialogue part must equal what the BASE-MOSHI
     codes path (``MoshiTokenizer.build_codes_stored``) builds from the same row with the same
     text-stream knobs -- one text-row convention, not two.

Runs in moshi_family_venv (sphn/sentencepiece), CPU, ~1 min (loads the mimi + spm from the HF cache):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_persona_prompt_prefix.py'
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.personaplex.models.lm import LMGen, SILENCE_TOKENS, SINE_TOKENS  # noqa: E402
from moshi_family.personaplex.train_data import (  # noqa: E402
    PersonaPlexDataConfig,
    PersonaPlexTokenizer,
    build_data_loader,
    system_prompt_prefix,
)

N_Q, SIL = 16, 6
rng = np.random.default_rng(0)
voice = rng.integers(0, 2048, size=(8, 50))
text_ids = [11, 222, 3333, 44, 5]


# ---- 1. prefix parity against the real step_system_prompts -------------------------------------
class Rec:
    """Just enough of an LMGen for step_system_prompts: the real methods, a recording step()."""


for name in (
    "step_system_prompts",
    "_step_voice_prompt",
    "_step_voice_prompt_core",
    "_step_voice_prompt_frame",
    "_step_audio_silence",
    "_step_audio_silence_core",
    "_step_text_prompt",
    "_step_text_prompt_core",
    "_encode_zero_frame",
    "_encode_sine_frame",
):
    setattr(Rec, name, getattr(LMGen, name))

rec = Rec()
rec.lm_model = types.SimpleNamespace(device="cpu")
rec.zero_text_code = 3
rec.audio_silence_frame_cnt = SIL
rec.text_prompt_tokens = text_ids
rec.voice_prompt = "x.wav"
rec.voice_prompt_audio = object()  # non-None -> the audio branch
rec.voice_prompt_embeddings = None
rec.save_voice_prompt_embeddings = False
rec._encode_voice_prompt_frames = lambda mimi: (
    torch.as_tensor(voice[:, i : i + 1]).view(1, 8, 1) for i in range(voice.shape[1])
)
frames = []


def step(moshi_tokens, text_token, input_tokens, return_embeddings=False):
    t = int(text_token) if not torch.is_tensor(text_token) else int(text_token.reshape(-1)[0])
    frames.append(np.concatenate([[t], moshi_tokens.reshape(-1).numpy(), input_tokens.reshape(-1).numpy()]))


rec.step = step
import contextlib  # noqa: E402

with contextlib.redirect_stdout(open(os.devnull, "w")):
    rec.step_system_prompts(mimi=None)
inference = np.stack(frames, axis=1)  # [17, P]
training = system_prompt_prefix(text_ids, voice, n_q=N_Q, silence_frames=SIL).numpy()
assert inference.shape == training.shape, (inference.shape, training.shape)
assert np.array_equal(inference, training), "training prefix != what step_system_prompts feeds"
assert inference.shape[1] == voice.shape[1] + SIL + len(text_ids) + SIL

old = np.full((1 + N_Q, len(text_ids)), 3)
old[0] = text_ids
old[1:] = 0  # the pre-2026-09-21 prefix: code 0 on every audio row, no voice, no silence
assert old.shape != inference.shape or not np.array_equal(old, inference), "guard is vacuous"
print(
    f"[1] prefix == step_system_prompts, {inference.shape[1]} frames (voice {voice.shape[1]} + {SIL} + text "
    f"{len(text_ids)} + {SIL}); the old code-0 prefix is rejected"
)

# ---- 2. the codes loader --------------------------------------------------------------------
from datasets import Dataset  # noqa: E402

from moshi_family.moshi_train_data import MoshiDataConfig, MoshiTokenizer  # noqa: E402

K, F, FR = 8, 250, 12.5
ca = rng.integers(0, 2048, size=(K, F)).astype(np.int16)
cu = rng.integers(0, 2048, size=(K, F)).astype(np.int16)
al = [
    {"text": w, "start": 0.5 + i * 0.8, "end": 0.9 + i * 0.8, "speaker": "assistant"}
    for i, w in enumerate("the quick brown fox jumps over the lazy dog".split())
]
prompt = "You enjoy having a good conversation. Talk about foxes."
tmp = tempfile.mkdtemp(prefix="check_ppx_prefix_")
Dataset.from_list(
    [
        {
            "id": "ep#0@a#w0",
            "duration": F / FR,
            "n_codebooks": K,
            "n_frames": F,
            "frame_rate": FR,
            "codes_assistant": ca.reshape(-1).tolist(),
            "codes_user": cu.reshape(-1).tolist(),
            "alignments": al,
            "context": prompt,
            "voice_codes": voice.astype(np.int16).reshape(-1).tolist(),
        }
    ]
).save_to_disk(tmp)

knobs = dict(emit_epad=True, onset_floor=True)
tok = PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1", device="cpu", cfg=PersonaPlexDataConfig(duration_sec=60, silence_frames=SIL, **knobs)
)
codes, mask = next(
    build_data_loader(tmp, tok, batch_size=1, seed=0, system_prompt_key="context", voice_codes_key="voice_codes")
)
codes, mask = codes[0], mask[0]
want_prefix = system_prompt_prefix(tok.system_prompt_text_tokens(prompt), voice, n_q=N_Q, silence_frames=SIL)
P = want_prefix.shape[1]
assert torch.equal(codes[:, :P], want_prefix), "loader prefix differs from system_prompt_prefix"
assert not mask[:P].any() and mask[P:].all(), "loss mask must be 0 exactly over the prompt"
mtok = MoshiTokenizer("kyutai/moshiko-pytorch-bf16", device="cpu", cfg=MoshiDataConfig(duration_sec=60, **knobs))
ref, _ = mtok.build_codes_stored(ca, cu, al)
dlg = codes[:, P:]
assert dlg.shape == ref.shape and torch.equal(dlg, ref), "dialogue part differs from the base-Moshi codes path"
assert (dlg[0] != 3).any(), "text row is all PAD"
print(f"[2] codes loader: prefix {P} frames masked, dialogue {dlg.shape[1]} frames == base-Moshi build_codes_stored")
print("OK")
