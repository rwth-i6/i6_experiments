"""Batched Chatterbox Turbo synthesis for ``chatterbox_inference.py``: many utterances (any mix of voices) at once.

Upstream ``ChatterboxTurboTTS.generate`` runs one utterance at a time: T3 (GPT-2 medium) emits one speech token per
forward at batch 1, then S3Gen and the Perth watermark. Here:

* ``ContinuousT3``: continuous batching. ``slots`` utterances decode together over a static KV cache (bf16); each
  finished one is read out and its slot refilled with the next prompt, so no slot idles waiting for the longest one.
  Per slot: its own write position, position id, seen-token mask (repetition penalty) and random stream. The step
  is captured as a CUDA graph per 128-position attention window. Sampling is upstream's chain (temperature, top-k,
  top-p, repetition penalty) with the exponential-race form of categorical sampling (argmax p / E, E ~ Exp(1) drawn
  from a generator seeded per utterance), so an utterance's draws do not depend on what else is in the batch.
* ``s3gen_batch``: S3Gen for a group of utterances at once (right-padded tokens with lengths, per-utterance seeded
  mean-flow noise, mel zeroed past each end, per-utterance trim and fade-in), finished utterances pooled and sorted
  by length so a group pads little.
* ``speed_s3gen``: the flow estimator's attention in bf16 on PyTorch's memory-efficient kernel (on H100 PyTorch picks
  cuDNN attention for bf16 with a mask, which is slower than fp32 here), no host sync in its mask builder; HiFiGAN
  compiled.
* No watermark: Perth only marks audio as synthetic, and these corpora are not published.

Not bit-reproducible run to run (GPU non-determinism in the batched kernels); measured against upstream ``generate``
on 116 utterances: Parakeet WER difference +0.003 [95% -0.008, +0.014], duration ratio median 1.00-1.01; 960 SFT
dialogues at 172 dialogue-h per GPU-h vs ~6-7 for the per-utterance loop (same H100, word timings running).
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

#: speech-token ids >= this are not audio (start / stop / special); upstream drops them before S3Gen
S3_VOCAB = 6561


def text_tokens(model, text: str) -> torch.Tensor:
    """Upstream's text path: ``punc_norm`` then the tokenizer, one text at a time."""
    from chatterbox.tts_turbo import punc_norm

    return model.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True).input_ids[0]


def _processors(temperature, top_k, top_p, repetition_penalty):
    """The logits processors of ``T3.inference_turbo``, in its order (used for each utterance's first token)."""
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    p = LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
        p.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        p.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        p.append(TopPLogitsWarper(top_p))
    if repetition_penalty != 1.0:
        p.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    return p


class ContinuousT3:
    """Continuous-batching T3 decode (see the module docstring). ``run`` returns each utterance's speech tokens as
    ``inference_turbo`` does (stop token dropped)."""

    def __init__(self, t3, slots: int, max_prompt: int, *, temperature=0.8, top_k=1000, top_p=0.95,
                 repetition_penalty=1.2, max_gen_len=1000, bucket: int = 128, kv_dtype=torch.bfloat16):  # fmt: skip
        tf = t3.tfmr
        self.t3, self.tf, self.b = t3, tf, slots
        dev = t3.device
        cfg = tf.config
        self.h, self.d = cfg.n_head, cfg.n_embd // cfg.n_head
        self.max_gen, self.bucket = max_gen_len, bucket
        self.lmax = -(-(max_prompt + max_gen_len + 2) // bucket) * bucket
        self.K = torch.zeros(len(tf.h), slots, self.h, self.lmax, self.d, device=dev, dtype=kv_dtype)
        self.V = torch.zeros_like(self.K)
        self.bias = torch.full((slots, self.lmax), float("-inf"), device=dev, dtype=kv_dtype)
        self.ar = torch.arange(slots, device=dev)
        self.zeros = torch.zeros(slots, device=dev, dtype=kv_dtype)
        self.c = torch.zeros(slots, dtype=torch.long, device=dev)  # next cache position
        self.pos = torch.zeros(slots, dtype=torch.long, device=dev)  # last position id
        self.si = torch.zeros(slots, dtype=torch.long, device=dev)  # next output / noise index
        self.tok = torch.zeros(slots, 1, dtype=torch.long, device=dev)
        vocab = t3.speech_head.out_features
        self.seen = torch.zeros(slots, vocab, dtype=torch.bool, device=dev)
        self.noise = torch.ones(slots, max_gen_len + 1, vocab, device=dev)
        self.stop = t3.hp.stop_speech_token
        self.out = torch.full((slots, max_gen_len + 1), self.stop, dtype=torch.long, device=dev)
        self.done = torch.ones(slots, dtype=torch.bool, device=dev)  # empty slots count as done
        self.temp, self.k, self.p, self.pen = temperature, top_k, top_p, repetition_penalty
        self.procs = _processors(temperature, top_k, top_p, repetition_penalty)
        self.graphs = {}

    def _step(self, w):
        t3, tf, b, ar = self.t3, self.tf, self.b, self.ar
        live = ~self.done
        self.bias.index_put_((ar, self.c), self.zeros)  # a device tensor: no host copy inside the graph
        self.pos += live.long()
        x = t3.speech_emb(self.tok) + tf.wpe(self.pos)[:, None]
        kd = self.K.dtype
        for i, blk in enumerate(tf.h):
            q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(self.h * self.d, dim=2)
            q, k, v = (t.view(b, 1, self.h, self.d).transpose(1, 2) for t in (q, k, v))
            self.K[i][ar, :, self.c] = k[:, :, 0].to(kd)
            self.V[i][ar, :, self.c] = v[:, :, 0].to(kd)
            a = F.scaled_dot_product_attention(q.to(kd), self.K[i][:, :, :w], self.V[i][:, :, :w],
                                               attn_mask=self.bias[:, None, None, :w])  # fmt: skip
            x = x + blk.attn.c_proj(a.to(x.dtype).transpose(1, 2).reshape(b, 1, self.h * self.d))
            x = x + blk.mlp(blk.ln_2(x))
        s = t3.speech_head(tf.ln_f(x))[:, -1, :].float()
        if self.temp > 0 and self.temp != 1.0:
            s = s / self.temp
        if self.k > 0:
            s = s.masked_fill(s < torch.topk(s, min(self.k, s.shape[-1]))[0][..., -1, None], float("-inf"))
        if self.p < 1.0:
            srt, idx = torch.sort(s, descending=False)
            rm = srt.softmax(-1).cumsum(-1) <= (1 - self.p)
            rm[..., -1:] = False
            s = s.masked_fill(rm.scatter(1, idx, rm), float("-inf"))
        if self.pen != 1.0:
            s = torch.where(self.seen, torch.where(s < 0, s * self.pen, s / self.pen), s)
        nxt = (F.softmax(s, dim=-1) / self.noise[ar, self.si]).argmax(-1, keepdim=True)
        nxt = torch.where(live[:, None], nxt, torch.full_like(nxt, self.stop))
        self.out[ar, self.si] = nxt[:, 0]
        self.seen.scatter_(1, nxt, True)
        self.tok.copy_(nxt)
        self.c += live.long()
        self.si.add_(live.long()).clamp_(max=self.max_gen)  # in place: the graph replays on these buffers
        self.done |= (nxt[:, 0] == self.stop) | (self.si >= self.max_gen)

    def _graph(self, w):
        if w not in self.graphs:
            state = (self.bias, self.pos, self.tok, self.seen, self.done, self.out, self.c, self.si)
            snap = [t.clone() for t in state]
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                self._step(w)  # warm-up outside the capture
            torch.cuda.current_stream().wait_stream(st)
            for t, s in zip(state, snap):
                t.copy_(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._step(w)
            self.graphs[w] = g
        return self.graphs[w]

    @torch.inference_mode()
    def _refill(self, slots: list[int], conds: list, texts: list[torch.Tensor], seeds: list[int]):
        """Prefill the new utterances (one left-padded batch) and move each into its slot: cache, mask, counters,
        noise drawn from its seed, its first token."""
        t3, dev = self.t3, self.t3.device
        start = torch.full((1, 1), t3.hp.start_speech_token, dtype=torch.long, device=dev)
        rows = [t3.prepare_input_embeds(t3_cond=c, text_tokens=t.view(1, -1).to(dev), speech_tokens=start)[0][0]
                for c, t in zip(conds, texts)]  # fmt: skip
        m, n = len(rows), max(r.shape[0] for r in rows)
        emb = rows[0].new_zeros(m, n, rows[0].shape[-1])
        mask = torch.zeros(m, n, dtype=torch.long, device=dev)
        for i, r in enumerate(rows):
            emb[i, n - r.shape[0] :] = r
            mask[i, n - r.shape[0] :] = 1
        kw = (
            {"attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp(min=0)}
            if bool((mask == 0).any())
            else {}
        )
        out = self.tf(inputs_embeds=emb, use_cache=True, **kw)
        cache = out.past_key_values
        layers = cache.layers if hasattr(cache, "layers") else None
        kv = [(layers[i].keys, layers[i].values) if layers is not None else cache[i] for i in range(len(self.tf.h))]
        logits = t3.speech_head(out[0][:, -1:])[:, -1, :]
        p0 = F.softmax(self.procs(start.expand(m, 1), logits), dim=-1)
        for j, (s, r) in enumerate(zip(slots, rows)):
            nr = r.shape[0]
            for i, (k, v) in enumerate(kv):
                self.K[i, s, :, :nr] = k[j, :, n - nr :].to(self.K.dtype)
                self.V[i, s, :, :nr] = v[j, :, n - nr :].to(self.V.dtype)
            self.bias[s] = float("-inf")
            self.bias[s, :nr] = 0.0
            self.c[s], self.pos[s] = nr, nr - 1
            g = torch.Generator(device=dev).manual_seed(int(seeds[j]))
            self.noise[s].exponential_(generator=g)
            first = int((p0[j] / self.noise[s, 0]).argmax())
            self.seen[s] = False
            self.seen[s, t3.hp.start_speech_token] = True
            self.seen[s, first] = True
            self.tok[s, 0] = first
            self.out[s] = self.stop
            self.out[s, 0] = first
            self.si[s] = 1
            self.done[s] = first == self.stop

    @torch.inference_mode()
    def run(self, conds: list, texts: list[torch.Tensor], seeds: list[int], on_done=None, harvest_every: int = 8):
        """Decode every utterance, longest text first; ``on_done(i, tokens)`` as each finishes (else returned)."""
        q = sorted(range(len(texts)), key=lambda i: -texts[i].numel())
        who = [None] * self.b
        res = [None] * len(texts)
        filled = [0] * self.b  # host mirror of each slot's used cache length

        def refill(free):
            take = [(s, q.pop(0)) for s in free[: len(q)]]
            if take:
                self._refill([s for s, _ in take], [conds[i] for _, i in take], [texts[i] for _, i in take],
                             [seeds[i] for _, i in take])  # fmt: skip
                for s, i in take:
                    who[s] = i
                    filled[s] = int(self.c[s])

        refill(list(range(self.b)))
        while any(w is not None for w in who):
            w = min(self.lmax, -(-(max(filled[s] for s in range(self.b) if who[s] is not None) + harvest_every)
                                 // self.bucket) * self.bucket)  # fmt: skip
            g = self._graph(w)
            for _ in range(harvest_every):
                g.replay()
            done = self.done.tolist()
            free = []
            for s in range(self.b):
                if who[s] is None:
                    free.append(s)
                    continue
                filled[s] += harvest_every
                if done[s]:
                    row = self.out[s]
                    hit = (row == self.stop).nonzero()
                    toks = row[: int(hit[0])].clone() if len(hit) else row.clone()
                    i, who[s] = who[s], None
                    if on_done is not None:
                        on_done(i, toks)
                    else:
                        res[i] = toks
                    free.append(s)
            if q and free:
                refill(free)
        return res


def _chunk_mask_nosync(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size, static_chunk_size,
                       num_decoding_left_chunks, enable_full_context=True):  # fmt: skip
    """S3Gen's ``add_optional_chunk_mask`` for the flow decoder's only call pattern (full context: the padding mask
    itself), without its all-false-row check, a host sync at each of its 14 calls per estimator step. A row is all
    false only for a zero-length utterance, which never reaches it."""
    assert not use_dynamic_chunk and static_chunk_size <= 0 and masks.dtype == torch.bool
    return masks


class _S3AttnProc:
    """diffusers ``AttnProcessor2_0`` for the flow estimator's self-attention (no cross-attention, norms or residual
    options in use there): q / k / v and the key-padding bias in bf16 on the memory-efficient SDPA kernel, the bias
    broadcast over heads, the output cast back. fp32 attention was ~35% of S3Gen's GPU time; on H100 PyTorch routes
    bf16 attention with a mask to cuDNN, measured slower here than fp32."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **kw):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        assert encoder_hidden_states is None and attn.norm_cross is None and attn.group_norm is None
        b, t, _ = hidden_states.shape
        h = attn.heads
        q, k, v = (
            f(hidden_states).view(b, t, h, -1).transpose(1, 2).bfloat16() for f in (attn.to_q, attn.to_k, attn.to_v)
        )
        m = attention_mask[:, None].bfloat16() if attention_mask is not None else None  # (B, 1, 1, T)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=attn.scale)
        o = attn.to_out[1](attn.to_out[0](o.transpose(1, 2).reshape(b, t, -1).to(hidden_states.dtype)))
        if attn.residual_connection:
            o = o + hidden_states
        return o / attn.rescale_output_factor


def speed_s3gen(model):
    """Patch S3Gen for batched inference: no host sync in the flow decoder's mask builder, the estimator's attention
    in bf16 on the memory-efficient kernel (``_S3AttnProc``), HiFiGAN's decode compiled (dynamic shapes; the first
    calls compile, ~20 s once per process)."""
    import chatterbox.models.s3gen.decoder as dec_mod
    from diffusers.models.attention_processor import Attention

    dec_mod.add_optional_chunk_mask = _chunk_mask_nosync
    n = 0
    for m in model.s3gen.flow.decoder.estimator.modules():
        if isinstance(m, Attention):
            m.set_processor(_S3AttnProc())
            n += 1
    assert n > 0, "no attention modules in the flow estimator: chatterbox changed, re-check speed_s3gen"
    model.s3gen.mel2wav.decode = torch.compile(model.s3gen.mel2wav.decode, dynamic=True)


@torch.inference_mode()
def s3gen_batch(
    model,
    tokens: list[torch.Tensor],
    conds: list,
    seeds: list[int],
    *,
    sil_tokens: int = 3,
    n_timesteps: int = 2,
    meanflow: bool = True,
    flow_autocast: torch.dtype | None = None,
) -> list:
    """S3Gen (``tokens_to_wav`` without watermark) for many utterances, one call per group of equal voice-prompt
    length: tokens right-padded with their lengths (the flow encoder and decoder take masks), each utterance's
    mean-flow noise from its own seed, HiFiGAN on the padded mels (zero mel past an utterance's end, as the
    convolutions' own zero padding sees upstream), each waveform trimmed to 2 x tokens x hop and faded in as
    upstream. Defaults = Turbo (3 appended silence tokens, 2 mean-flow steps); the regular / multilingual S3Gen is
    ``sil_tokens=0, n_timesteps=10, meanflow=False`` (flow matching with CFG; the generated part's noise is seeded per
    utterance, the voice-prompt part's comes from the global generator as upstream's). -> float32 numpy waveforms at
    ``model.sr``."""
    from chatterbox.tts_turbo import S3GEN_SIL

    s3 = model.s3gen
    dev, dt = s3.device, s3.dtype
    toks = []
    for t in tokens:
        t = t[t < S3_VOCAB].to(dev)
        toks.append(torch.cat([t, torch.tensor([S3GEN_SIL] * sil_tokens, dtype=torch.long, device=dev)]))
    groups: dict = {}
    for i, c in enumerate(conds):
        g = c.gen
        groups.setdefault((int(g["prompt_token"].shape[-1]), int(g["prompt_feat"].shape[1])), []).append(i)
    out = [None] * len(tokens)
    for idx in groups.values():
        lens = torch.tensor([toks[i].numel() for i in idx], device=dev)
        tmax = int(lens.max())
        tok = torch.zeros(len(idx), tmax, dtype=torch.long, device=dev)
        noise = torch.zeros(len(idx), 80, 2 * tmax, dtype=dt, device=dev)
        for j, i in enumerate(idx):
            tok[j, : toks[i].numel()] = toks[i]
            gen = torch.Generator(device=dev).manual_seed(int(seeds[i]))
            noise[j, :, : 2 * toks[i].numel()] = torch.randn(1, 80, 2 * toks[i].numel(), dtype=dt, device=dev,
                                                           generator=gen)  # fmt: skip
        ref = [conds[i].gen for i in idx]
        cat = lambda k: torch.cat([r[k].to(dev) for r in ref], dim=0)  # noqa: E731
        amp = torch.autocast("cuda", dtype=flow_autocast) if flow_autocast is not None else nullcontext()
        with amp:
            feat, _ = s3.flow.inference(
                token=tok, token_len=lens, prompt_token=cat("prompt_token"), prompt_token_len=cat("prompt_token_len"),
                prompt_feat=cat("prompt_feat"), prompt_feat_len=None, embedding=cat("embedding"), finalize=True,
                n_timesteps=n_timesteps, noised_mels=noise, meanflow=meanflow,
            )  # fmt: skip
        feat = feat.to(dtype=dt)
        for j in range(len(idx)):  # zero mel past each utterance's end
            feat[j, :, 2 * int(lens[j]) :] = 0
        wav, _ = s3.hift_inference(feat, None)
        wav = wav.float()
        hop = wav.shape[-1] // feat.shape[-1]
        for j, i in enumerate(idx):
            w = wav[j, : 2 * int(lens[j]) * hop].clone()
            w[: len(s3.trim_fade)] *= s3.trim_fade
            out[i] = w.cpu().numpy()
    return out


def synthesize(model, items: list[dict], *, slots: int = 96, s3_batch: int = 32, s3_pool: int = 96,
               log_every: float = 60.0) -> list:  # fmt: skip
    """Waveforms (float32 numpy at ``model.sr``) for ``items`` = [{"text", "cond" (a ``model.conds``), "seed"}]:
    ``ContinuousT3`` with ``slots`` slots; finished utterances pooled ``s3_pool`` at a time, sorted by length and run
    through ``s3gen_batch`` in groups of ``s3_batch``."""
    if not items:
        return []
    toks = [text_tokens(model, it["text"]) for it in items]
    dec = ContinuousT3(model.t3, min(slots, len(items)),
                       max_prompt=model.t3.hp.speech_cond_prompt_len + max(t.numel() for t in toks) + 4)  # fmt: skip
    out = [None] * len(items)
    pend = []
    t0, last = time.time(), [time.time()]

    def flush():
        pend.sort(key=lambda p: p[1].numel())
        for a in range(0, len(pend), s3_batch):
            grp = pend[a : a + s3_batch]
            ws = s3gen_batch(model, [t for _, t in grp], [items[i]["cond"] for i, _ in grp],
                             [items[i]["seed"] for i, _ in grp])  # fmt: skip
            for (i, _), w in zip(grp, ws):
                out[i] = w
        pend.clear()
        if time.time() - last[0] > log_every:
            last[0] = time.time()
            n = sum(o is not None for o in out)
            print(f"[chatterbox_fast] {n}/{len(items)} utterances, {time.time() - t0:.0f} s", flush=True)

    def done(i, t):
        pend.append((i, t))
        if len(pend) >= s3_pool:
            flush()

    dec.run([it["cond"].t3 for it in items], toks, [it["seed"] for it in items], on_done=done)
    if pend:
        flush()
    assert all(o is not None for o in out)
    return out
