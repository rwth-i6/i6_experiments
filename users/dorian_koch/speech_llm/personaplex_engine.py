"""PersonaPlex offline inference engine -- the science layer (lives in *our* repo).

PersonaPlex (`nvidia/personaplex-7b-v1`) is an end-to-end causal full-duplex model in the
Moshi family, but its reply is driven by a **persona/system text prompt + a voice prompt**
applied through the fork's *extended* `LMGen` (`load_voice_prompt[_embeddings]` ->
`text_prompt_tokens` -> `step_system_prompts(mimi)`) before the user audio is streamed. The
plain base-Moshi `CheckpointInfo`/`InferenceState` path does NOT do this, so we must mirror
PersonaPlex's own offline inference (`moshi.offline.run_inference`).

This engine is a faithful, **load-once + loop** refactor of that `run_inference` (the fork's
CLI loads the 7B per file -> useless for 1000 clips). It reuses the fork's helpers verbatim
(`warmup`, `decode_tokens_to_pcm`, `wrap_with_system_tags`, `load_audio`/`_iterate_audio`/
`encode_from_sphn`) and exposes the *ablation surface* -- persona prompt, voice, sampling,
and a **RAG hook** (`context_provider`) -- as our parameters.

Both benchmarks drive it via `personaplex_offline_inference.py`:
* knowledge: pad each input with `lead_in_s` + `capture_s` silence (PersonaPlex trims output
  to input length, so trailing silence is what lets the full spoken answer be captured);
* FDB: `lead_in_s=capture_s=0` -> process the clip as-is (output aligns with input timing).

Pinned against the fork's `moshi/moshi/offline.py` (verified 2026-06-16). If the fork changes
those helper names, the imports below fail loudly -- re-sync then.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Callable

import numpy as np
import sphn
import torch

# PersonaPlex's README QA persona (matches the fork's `--text-prompt` default).
DEFAULT_SYSTEM_PROMPT = (
    "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
)
DEFAULT_VOICE = "NATM1.pt"  # a voice basename inside the model's voices.tgz

# RAG / context hook: `context_provider(clip_stem) -> str | None` returns text folded into
# the persona prompt for that clip. Reserved for Moshi-RAG; the retriever plugs in here.
ContextProvider = Callable[[str], "str | None"]


def resolve_voice_dir(hf_repo: str, voice_prompt_dir: str | None = None) -> str:
    """Return a dir of voice prompts; download+extract voices.tgz from HF if not given.

    Replicates `moshi.offline._get_voice_prompt_dir` (kept local to avoid importing a
    private name). The HF fetch is gated -- jobs carry HF_TOKEN; see personaplex.md.
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir
    from huggingface_hub import hf_hub_download

    voices_tgz = Path(hf_hub_download(hf_repo, "voices.tgz"))
    voices_dir = voices_tgz.parent / "voices"
    if not voices_dir.exists():
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")
    return str(voices_dir)


class PersonaPlexModel:
    """Loaded-once PersonaPlex inference state (two Mimi, tokenizer, LM, LMGen).

    Construction mirrors `moshi.offline.run_inference` steps 1-5 (load mimi x2, tokenizer,
    LM; build the fork's `LMGen` with its persona-capable kwargs; stream + warmup).
    """

    def __init__(
        self,
        hf_repo: str,
        *,
        device: str = "cuda",
        cpu_offload: bool = False,
        temp_audio: float = 0.8,
        temp_text: float = 0.7,
        topk_audio: int = 250,
        topk_text: int = 25,
        greedy: bool = False,
        trained_weights: str | None = None,
    ):
        import sentencepiece
        from huggingface_hub import hf_hub_download

        from moshi.models import LMGen, loaders
        from moshi.offline import warmup

        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device)
        self.other_mimi = loaders.get_mimi(mimi_weight, device)  # agent-side decoder (fork uses two)

        tok = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tok)

        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
        if trained_weights:
            # Overlay a finetuned-weights checkpoint onto the base LM. Our SpeechFinetune /
            # PERSONAPLEX_ADAPTER writes consolidated/trained_heads.safetensors -- the
            # requires_grad params only (depformer + per-codebook audio heads + text head +
            # out_norm; backbone frozen). It is therefore a PARTIAL state_dict, NOT a LoRA, so
            # load strict=False (the frozen backbone keys are legitimately 'missing' from it)
            # and assert every checkpoint key maps onto a real model param -- a stray
            # 'unexpected' key means the checkpoint does not fit this base, so fail loud
            # instead of silently no-op'ing the finetune.
            import safetensors.torch as st

            sd = st.load_file(trained_weights, device=device)
            assert sd, f"trained_weights {trained_weights} is empty"
            missing, unexpected = lm.load_state_dict(sd, strict=False)
            assert not unexpected, (
                f"trained_weights has {len(unexpected)} key(s) absent from the base LM "
                f"(checkpoint does not fit this model): {list(unexpected)[:5]}"
            )
            print(
                f"[personaplex] overlaid {len(sd)} finetuned tensor(s) onto the base model "
                f"({len(missing)} base params kept frozen)",
                flush=True,
            )
        lm.eval()

        self.device = device
        self.sample_rate = self.mimi.sample_rate
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
            use_sampling=not greedy,
            temp=temp_audio,
            temp_text=temp_text,
            top_k=topk_audio,
            top_k_text=topk_text,
        )
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        warmup(self.mimi, self.other_mimi, self.lm_gen, device, self.frame_size)

    def set_voice(self, voice_path: str) -> None:
        """Load the agent voice prompt (`.pt` embeddings or a wav), once per run."""
        if voice_path.endswith(".pt"):
            self.lm_gen.load_voice_prompt_embeddings(voice_path)
        else:
            self.lm_gen.load_voice_prompt(voice_path)


def _pad_audio(audio, sr: int, lead_s: float, cap_s: float):
    nl, nr = int(round(lead_s * sr)), int(round(cap_s * sr))
    if nl == 0 and nr == 0:
        return audio
    if isinstance(audio, torch.Tensor):
        return torch.nn.functional.pad(audio, (nl, nr))  # pads the last dim
    # ndim-agnostic: pad only the last (time) axis with silence.
    return np.pad(audio, [(0, 0)] * (audio.ndim - 1) + [(nl, nr)])


def run_pairs(
    model: PersonaPlexModel,
    pairs: list[tuple[str, str]],
    *,
    hf_repo: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    voice: str = DEFAULT_VOICE,
    voice_prompt_dir: str | None = None,
    lead_in_s: float = 0.0,
    capture_s: float = 0.0,
    context_provider: ContextProvider | None = None,
) -> None:
    """Run PersonaPlex over `(in_wav, out_wav)` pairs, writing each reply wav.

    Mirrors `run_inference` steps 6-11 per clip (set persona+voice, reset streaming,
    `step_system_prompts`, stream the user frames, decode, trim/pad to input length), with
    the model loaded once. `system_prompt` is re-tokenized per clip so a `context_provider`
    (RAG) can vary it.
    """
    from moshi.models.lm import _iterate_audio as lm_iterate_audio
    from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
    from moshi.models.lm import load_audio as lm_load_audio
    from moshi.offline import decode_tokens_to_pcm, wrap_with_system_tags

    lm_gen, mimi, other_mimi, tok = model.lm_gen, model.mimi, model.other_mimi, model.text_tokenizer
    model.set_voice(os.path.join(resolve_voice_dir(hf_repo, voice_prompt_dir), voice))
    sr = model.sample_rate

    for done, (in_wav, out_wav) in enumerate(pairs, 1):
        prompt = system_prompt
        if context_provider is not None:
            extra = context_provider(Path(in_wav).stem)
            if extra:
                prompt = f"{system_prompt}\n\nRelevant context:\n{extra}"
        lm_gen.text_prompt_tokens = tok.encode(wrap_with_system_tags(prompt)) if prompt else None

        # Prompt phase: voice + system text primed before any user audio.
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()

        user_audio = _pad_audio(lm_load_audio(in_wav, sr), sr, lead_in_s, capture_s)
        total = user_audio.shape[-1]
        frames = []
        with torch.no_grad():
            for user_encoded in lm_encode_from_sphn(
                mimi,
                lm_iterate_audio(user_audio, sample_interval_size=lm_gen._frame_size, pad=True),
                max_batch=1,
            ):
                for c in range(user_encoded.shape[-1]):
                    tokens = lm_gen.step(user_encoded[:, :, c : c + 1])
                    if tokens is None:
                        continue
                    frames.append(decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens))

        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        if frames:
            pcm = np.concatenate(frames, axis=-1)
            if pcm.shape[-1] > total:
                pcm = pcm[:total]
            elif pcm.shape[-1] < total:
                pcm = np.concatenate([pcm, np.zeros(total - pcm.shape[-1], dtype=pcm.dtype)], axis=-1)
        else:
            # Model stayed silent (e.g. FDB interruption clips): emit input-length silence,
            # not a 1-sample wav. NeMo's normalize_batch needs >1 sample (std over length 1 = nan).
            pcm = np.zeros(total, dtype=np.float32)
        # Hard floor so downstream ASR never receives a sub-hop_length clip.
        min_len = max(sr // 10, 1)
        if pcm.shape[-1] < min_len:
            pcm = np.concatenate([pcm, np.zeros(min_len - pcm.shape[-1], dtype=pcm.dtype)], axis=-1)
        sphn.write_wav(str(out_wav), pcm, sr)
        print(f"[personaplex] {done}/{len(pairs)} clips done", flush=True)
