"""PersonaPlex training data path + loss (Workstream E of the PersonaPlex replication).

Runs inside the **personaplex venv** (torch 2.4.1, moshi-personaplex fork, sphn, soundfile,
datasets, safetensors -- but NO torchaudio/torchcodec/scipy). So audio is read as raw WAV bytes
from the arrow table and decoded with ``soundfile`` (HF ``Audio`` decode would need torchcodec),
and resampling uses ``sphn`` (no torchaudio).

The PersonaPlex fork (import name ``moshi``) is NOT the moshi-finetune trainer, so
``moshi_arrow_dataset`` (which imports ``finetune.data.interleaver``) cannot be reused. This is the
self-contained replacement: read the annotated arrow dataset (``MoshiAnnotate`` output: columns
``audio_assistant``/``audio_user`` raw wav bytes, ``alignments``, ``duration``) and produce the
``codes [B, K, T]`` tensors ``moshi.models.lm.LMModel.forward_train`` consumes, plus a per-frame loss
mask that zeros the **system-prompt** region (paper: "we mask out loss backpropagation to the system
prompt"). We re-encode raw audio with the PersonaPlex mimi here, so the annotate step's own codec
need not match (it doesn't reuse annotate codes).

Codes layout (pinned to the fork's ``_lm_kwargs`` + ``get_moshi_lm``):
    row 0          : text token per frame (card 32000, padding id 3)
    rows 1..n_q    : audio codebooks from ``mimi.encode`` of the stereo (assistant, user) wav (n_q=16)
  => num_codebooks = 1 + n_q = 17, matching ``len(LMModel.delays)``.

VERIFY(personaplex) -- two seams the first GPU smoke must confirm (asserted below so a mismatch
fails loud, never silently mis-trains):
  (1) the word->frame TEXT interleaving here reimplements the fork's (absent) training tokenizer;
  (2) audio codebook count from ``mimi.encode`` of a 2-channel input == the 16 rows forward_train
      expects (shape-asserted).
Hybrid system prompt: role text via ``wrap_with_system_tags`` -> leading text frames, loss masked.
VOICE-prompt training conditioning is a further seam (TODO) -- see personaplex.md.
"""

from __future__ import annotations

import site as _site
import sys as _sys

# The recipe tree has a top-level ``moshi`` (kyutai source) that SHADOWS the PersonaPlex fork in the
# venv site-packages whenever ``recipe`` is on PYTHONPATH (the documented thin-venv gotcha -- see
# personaplex.md). Make the venv site-packages win for ``import moshi`` so we load the PersonaPlex
# fork (patched strict=False loader), not recipe/moshi (kyutai, strict load -> depformer size
# mismatch). Only ``moshi`` collides; i6_experiments/speech_llm/sisyphus are recipe-only.
for _p in _site.getsitepackages():
    if _p in _sys.path:
        _sys.path.remove(_p)
        _sys.path.insert(0, _p)

import io
import logging
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import soundfile as sf
import sphn
import torch

logger = logging.getLogger("personaplex.data")

TEXT_PADDING_ID = 3  # _lm_kwargs["existing_text_padding_id"]
TEXT_ROW = 0
AUDIO_OFFSET = 1


@dataclass
class PersonaPlexDataConfig:
    """Knobs for the training data path (paper: seq 2048 tok @ 12.5 Hz = 163.84 s)."""

    duration_sec: float = 163.84
    default_system_prompt: str = (
        "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
    )


def _norm_aligns(aligns) -> list[tuple]:
    """Coerce the alignments column to (word:str, (start_s, end_s), speaker) tuples (dual-mode)."""
    out = []
    for a in aligns or []:
        if isinstance(a, dict):
            word, s, e, spk = a.get("text"), a.get("start"), a.get("end"), a.get("speaker", "")
        else:
            word, span, spk = a[0], a[1], (a[2] if len(a) > 2 else "")
            s, e = span[0], span[1]
        out.append((str(word), (float(s), float(e)), spk))
    return out


def _decode_wav_bytes(b: bytes) -> tuple[np.ndarray, int]:
    """Decode raw WAV bytes -> (mono float32 [T], sample_rate). soundfile, no torchcodec."""
    data, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if data.ndim == 2:  # collapse any stray channels to mono
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sr)


class PersonaPlexTokenizer:
    """Loads the PersonaPlex fork's mimi + sentencepiece text tokenizer; builds training codes."""

    def __init__(self, hf_repo: str, *, device: str = "cuda", cfg: PersonaPlexDataConfig | None = None):
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        self.cfg = cfg or PersonaPlexDataConfig()
        self.device = device
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device)
        self.n_q = 16  # audio codebooks the LM consumes (fork stereo Moshi layout)
        self.frame_rate = self.mimi.frame_rate
        self.sample_rate = self.mimi.sample_rate
        tok = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self.text_tok = sentencepiece.SentencePieceProcessor(tok)
        self.num_frames = int(round(self.cfg.duration_sec * self.frame_rate))

    def system_prompt_text_tokens(self, prompt: str) -> list[int]:
        from moshi.offline import wrap_with_system_tags

        return list(self.text_tok.encode(wrap_with_system_tags(prompt))) if prompt else []

    def _interleave_text(self, aligns: list[tuple], num_frames: int) -> torch.Tensor:
        """Place each word's text tokens at its onset frame; pad (id 3) elsewhere. [1, num_frames].

        VERIFY(personaplex): reimplements the fork's (absent) training text interleaver.
        """
        row = torch.full((num_frames,), TEXT_PADDING_ID, dtype=torch.long)
        for word, (start_s, _e), _spk in aligns:
            f0 = int(round(start_s * self.frame_rate))
            if f0 < 0 or f0 >= num_frames:
                continue
            for j, tid in enumerate(self.text_tok.encode(word)):
                if f0 + j < num_frames:
                    row[f0 + j] = tid
        return row[None, :]

    @torch.no_grad()
    def build_codes(
        self,
        assistant: np.ndarray,
        user: np.ndarray,
        sr_native: int,
        alignments,
        *,
        system_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (codes [K, T], loss_mask [T]) from decoded mono channels + alignments."""
        n = max(assistant.shape[0], user.shape[0])
        assistant = np.pad(assistant, (0, n - assistant.shape[0]))
        user = np.pad(user, (0, n - user.shape[0]))
        if sr_native != self.sample_rate:
            assistant = sphn.resample(assistant, sr_native, self.sample_rate)
            user = sphn.resample(user, sr_native, self.sample_rate)
        end = int(self.cfg.duration_sec * self.sample_rate)
        wav = torch.from_numpy(np.stack([assistant[:end], user[:end]], axis=0)).float().to(self.device)

        audio_tokens = self.mimi.encode(wav[:, None])  # -> view to [1, n_q, Ta]
        audio_tokens = audio_tokens.view(1, -1, audio_tokens.shape[-1])
        assert audio_tokens.shape[1] == self.n_q, (
            f"VERIFY(personaplex): mimi.encode gave {audio_tokens.shape[1]} codebooks, expected n_q={self.n_q}"
        )
        Ta = min(audio_tokens.shape[-1], self.num_frames)
        audio_tokens = audio_tokens[..., :Ta]

        text_row = self._interleave_text(_norm_aligns(alignments), Ta).to(self.device)
        dialogue_codes = torch.cat([text_row, audio_tokens[0]], dim=0)  # [K, Ta]

        prompt = system_prompt if system_prompt is not None else self.cfg.default_system_prompt
        sp_ids = self.system_prompt_text_tokens(prompt)
        K = dialogue_codes.shape[0]
        if sp_ids:
            sp = torch.full((K, len(sp_ids)), TEXT_PADDING_ID, dtype=torch.long, device=self.device)
            sp[TEXT_ROW] = torch.tensor(sp_ids, dtype=torch.long, device=self.device)
            sp[AUDIO_OFFSET:] = 0  # silence code over the prompt
            codes = torch.cat([sp, dialogue_codes], dim=1)
            mask = torch.cat([torch.zeros(len(sp_ids), dtype=torch.bool), torch.ones(Ta, dtype=torch.bool)])
        else:
            codes, mask = dialogue_codes, torch.ones(Ta, dtype=torch.bool)
        return codes.cpu(), mask


def _collate(samples: list[tuple[torch.Tensor, torch.Tensor]], pad_text: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad (codes [K,T], mask [T]) to batch max T; audio rows pad 0, text row pads `pad_text`."""
    K = samples[0][0].shape[0]
    T = max(c.shape[1] for c, _ in samples)
    B = len(samples)
    codes = torch.zeros(B, K, T, dtype=torch.long)
    codes[:, TEXT_ROW] = pad_text
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, (c, m) in enumerate(samples):
        t = c.shape[1]
        codes[i, :, :t] = c
        mask[i, :t] = m
    return codes, mask


def _read_table(dataset_path: str):
    """Load the arrow dataset and return its pyarrow table (raw bytes, no Audio decode)."""
    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)
    return ds.data.table


def build_data_loader(
    dataset_path: str,
    tokenizer: PersonaPlexTokenizer,
    *,
    batch_size: int,
    seed: int = 0,
    system_prompt_key: str | None = None,
    infinite: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(codes [B,K,T], loss_mask [B,T])`` from the annotated arrow dataset (DDP-sharded).

    Reads raw wav bytes + alignments straight from the pyarrow table (avoids the HF ``Audio``
    decode that needs torchcodec). ``system_prompt_key``: optional column with a per-row role
    prompt (e.g. service ``context``); falls back to the default QA persona.
    """
    table = _read_table(dataset_path)
    n = table.num_rows
    cols = set(table.column_names)
    has_split = "audio_assistant" in cols
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rng = np.random.default_rng(seed + rank)

    def decode_row(idx: int):
        if has_split:
            ab = table.column("audio_assistant")[idx].as_py()["bytes"]
            ub = table.column("audio_user")[idx].as_py()["bytes"]
            assistant, sr = _decode_wav_bytes(ab)
            user, _ = _decode_wav_bytes(ub)
        else:  # legacy single stereo "audio" column
            raw = table.column("audio")[idx].as_py()["bytes"]
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            assistant, user = data[:, 0], data[:, 1]
        aligns = table.column("alignments")[idx].as_py()
        sp = table.column(system_prompt_key)[idx].as_py() if (system_prompt_key and system_prompt_key in cols) else None
        return assistant, user, sr, aligns, sp

    while True:
        order = rng.permutation(n)
        buf: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx in order[rank::world]:
            try:
                assistant, user, sr, aligns, sp = decode_row(int(idx))
                buf.append(tokenizer.build_codes(assistant, user, sr, aligns, system_prompt=sp))
            except Exception as e:  # one bad row must not kill a long run; log + skip
                logger.warning("skip row %d: %r", int(idx), e)
                continue
            if len(buf) == batch_size:
                yield _collate(buf, TEXT_PADDING_ID)
                buf = []
        if not infinite:
            if buf:
                yield _collate(buf, TEXT_PADDING_ID)
            return


def personaplex_loss(
    out,
    codes: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    dep_q: int,
    audio_nonsemantic_weight: float = 0.02,
    text_pad_weight: float = 0.3,
) -> torch.Tensor:
    """Paper-weighted CE over text + audio with the system-prompt masked out.

    out: LMOutput(logits [B,K,T,card], mask, text_logits [B,1,T,text_card], text_mask). Targets are
    the input ``codes`` (forward_train realigns logits to the input). Down-weights: non-semantic
    audio codebooks (all but the first/semantic) x0.02, padded text tokens x0.3 (paper); loss_mask
    zeros the system-prompt frames.
    """
    import torch.nn.functional as F

    B, _K, T = codes.shape
    fm = loss_mask.to(codes.device)

    # forward_train fills invalid/delayed positions with NaN logits; nan_to_num so CE is finite and
    # the masks (which already encode validity, incl. zero/initial tokens) zero those out. Clamp
    # targets into range so masked-out positions never index out of bounds (NaN*0 would poison sum).
    text_logits = torch.nan_to_num(out.text_logits[:, 0])  # [B, T, text_card]
    text_card = text_logits.shape[-1]
    text_tgt = codes[:, TEXT_ROW]
    tvalid = out.text_mask[:, 0] & fm
    tt = text_tgt.reshape(-1).clamp(0, text_card - 1)
    tl = F.cross_entropy(text_logits.reshape(-1, text_card), tt, reduction="none")
    tw = torch.where(text_tgt.reshape(-1) == TEXT_PADDING_ID, text_pad_weight, 1.0)
    tl = (tl * tw * tvalid.reshape(-1).float()).sum() / tvalid.sum().clamp(min=1)

    audio_logits = torch.nan_to_num(out.logits)  # [B, K, T, card]
    card = audio_logits.shape[-1]
    audio_tgt = codes[:, AUDIO_OFFSET : AUDIO_OFFSET + dep_q]
    avalid = out.mask[:, :dep_q] & fm[:, None, :]
    al = F.cross_entropy(
        audio_logits[:, :dep_q].reshape(-1, card),
        audio_tgt.reshape(-1).clamp(0, card - 1),
        reduction="none",
    ).view(B, dep_q, T)
    cb_w = torch.full((dep_q,), audio_nonsemantic_weight, device=audio_logits.device)
    cb_w[0] = 1.0  # semantic audio codebook (weight 1); rest non-semantic (x0.02)
    al = (al * cb_w[None, :, None] * avalid.float()).sum() / avalid.float().sum().clamp(min=1)
    return tl + al
