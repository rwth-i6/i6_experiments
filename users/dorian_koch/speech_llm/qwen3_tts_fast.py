"""Batched Qwen3-TTS (12 Hz, 1.7B Base) voice-clone synthesis: many utterances at once, upstream's model and sampling.

Upstream ``Qwen3TTSModel.generate_voice_clone`` (``qwen_tts`` 0.1.1) runs a fixed batch through HF ``generate``: per
codec frame (12.5 / s) one talker step (28 layers) whose forward first runs a second, nested HF ``generate`` of the
code predictor (5 layers, 15 sequential steps filling codebooks 1-15), every step through HF's Python sampling loop.
The batch runs until its slowest row ends (``max_new_tokens`` 8192 by default), then the speech tokenizer decodes it.

Reproduced from upstream unchanged: the prompt embeddings (built by upstream's own ``Qwen3TTSForConditionalGeneration
.generate`` up to the talker call, so the ICL clone prompt, language tag, speaker embedding and the streamed trailing
text are upstream's), the module weights and their bf16 forward (the same submodules called in the same order; the
codebook embeddings summed in one ``sum`` as upstream), the talker's sampling chain (repetition penalty over the
generated first-codebook ids, end token masked for the first two frames, the 1024 control ids suppressed,
temperature, top-k, top-p) and the code predictor's (temperature, top-k, top-p), the defaults from the checkpoint's
``generate_config.json`` via upstream's ``_merge_generate_kwargs``, and the decode (``speech_tokenizer.decode`` of
reference codes + generated codes, the reference part cut off proportionally).

Changed (runtime only):

* ``ContinuousTalker``: continuous batching over ``slots`` rows (a finished row is replaced by the next utterance at
  the next harvest), a static bf16 KV cache, and the whole frame step (code predictor's 15 steps + talker step +
  both samplers) captured as one CUDA graph per 128-position attention window.
* Sampling noise: the categorical draw is the exponential race (``argmax p / E``, the same distribution as
  ``torch.multinomial``) with ``E`` from a counter hash of (utterance seed, frame, codebook, token id), so the noise
  an utterance sees does not depend on the batch or slot (its tokens still do through bf16 batch-shape rounding: once
  one token flips, the continuation differs). Upstream draws from the global torch RNG, so no
  utterance is bit-reproducible against it in sampled mode. ``do_sample=False`` is greedy as upstream.
* Frame cap per utterance (``max_frames``, default ``default_max_frames``); upstream ``max_new_tokens=N`` keeps at
  most N - 1 frames, so ``max_frames=F`` matches ``max_new_tokens=F + 1``.
* The speech tokenizer decodes finished utterances in length-sorted groups on a side stream / thread while the talker
  keeps generating.

Measured (one H100, 2026-11-19, p11 pilot dialogue turns; scripts and logs in
``/hpcwork/tt201262/sbatch/sft_pilot/q3fast/``):

* Speed (model load excluded; prompts, graph capture and decode included): the 441 turns x 4 (17,926 s audio) at 113x
  realtime with 64 slots (64 GB peak), 78x with 32 slots (33 GB); the 441 turns alone 98x at 64 slots (52 GB with
  ``dec_batch=8``, 70 GB with 16), 99x at 96 slots (93 GB: no gain). Upstream
  ``generate_voice_clone`` on 32 turns: 5.2x at batch 8, 17x at batch 32 (one batch; ~70% of its time is the nested
  code-predictor ``generate``).
* Greedy (60 turns, 15 each en/de/ru/zh): no utterance has all 16 codebooks identical to upstream batch 1, and
  neither does upstream batch 8 against upstream batch 1 (first first-codebook divergence: median frame 9 for both).
  Teacher-forced along upstream's greedy codes (8 turns): logits within 0.25 (median) / 1.25 (max) of upstream's,
  argmax disagreements only at upstream top-2 margins <= 0.5, i.e. bf16 near-ties, no systematic difference.
* Sampled, Qwen3-ASR CER vs the input text, 60 turns x 7 seeds: upstream 0.0384, fast 0.0397, difference +0.0012
  [95% -0.0006, +0.0032] (bootstrap over turns); per language en -0.0008, de +0.0015, ru +0.0045 [+0.0005, +0.0101],
  zh -0.0004. The Russian gap did not replicate on 40 fresh Russian turns x 5 seeds: 0.0194 vs 0.0194, difference
  -0.0000 [-0.0024, +0.0023]; all Russian pooled (55 turns, 305 pairs) +0.0017 [-0.0009, +0.0037]. Large per-sample
  differences fall on both sides (code-switched English words, ASR script choices).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL = (
    "/hpcwork/p0023999/common_hf_home/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/"
    "fd4b254389122332181a7c3db7f27e918eec64e3"
)
M32 = 0xFFFFFFFF


def load(device: str = "cuda", path: str = MODEL):
    """Upstream's ``Qwen3TTSModel`` in bf16 with SDPA attention (as the upstream baseline loads it)."""
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(path, device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa")


def prompt(model, ref_audio, ref_text: str | None, timbre_only: bool = False):
    """Upstream's voice-clone prompt for one clip: ICL (reference codes + speaker embedding + reference text), or with
    ``timbre_only`` the speaker embedding alone (upstream ``x_vector_only_mode``). ICL continues the reference
    recording, accent included, so text in another language than the clip's comes out with the clip's accent
    (60 foreign spans, Qwen3-ASR in the span's language: ICL 11.9% vs speaker embedding only 1.0% CER)."""
    if timbre_only:
        return model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=None, x_vector_only_mode=True)[0]
    return model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)[0]


def default_max_frames(text: str, language: str) -> int:
    """Frame cap from a slow reading of ``text`` x 2.5 + 3 s (Chinese / Japanese 3 characters / s, else 12)."""
    rate = 3 if language.lower() in ("chinese", "japanese") else 12
    return int(12.5 * (2.5 * len(text) / rate + 3))


class _Captured(Exception):
    pass


@torch.inference_mode()
def prompt_embeds(model, text: str, language: str, item) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(talker prompt embeddings [n, D], trailing text embeddings [t, D], the text pad embedding [1, D]) exactly as upstream's ``generate`` builds
    them for this utterance: its code runs up to the talker ``generate`` call, which is intercepted."""
    m = model.model
    model._validate_languages([language])
    ids = model._tokenize_texts([model._build_assistant_text(text)])
    ref_ids = [model._tokenize_texts([model._build_ref_text(item.ref_text)])[0] if item.ref_text else None]
    cap = {}

    def grab(**kw):
        cap.update(kw)
        raise _Captured

    m.talker.generate = grab
    try:
        m.generate(input_ids=ids, ref_ids=ref_ids, voice_clone_prompt=model._prompt_items_to_voice_clone_prompt([item]),
                   languages=[language], non_streaming_mode=False)  # fmt: skip
    except _Captured:
        pass
    finally:
        del m.talker.generate
    return cap["inputs_embeds"][0], cap["trailing_text_hidden"][0], cap["tts_pad_embed"].view(1, -1)


def _mix(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 integer hash on int64 tensors holding uint32 values (products wrap; the low 32 bits are exact)."""
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


class ContinuousTalker:
    """Continuous-batching talker + code predictor decode (module docstring). ``run`` returns each utterance's codec
    frames [frames, 16] (the end frame excluded)."""

    def __init__(self, model, slots: int, max_prompt: int, max_text: int, max_frames: int, gen: dict,
                 bucket: int = 128):  # fmt: skip
        tk = model.model.talker
        cfg = tk.config
        self.tk, self.tm, self.cp = tk, tk.model, tk.code_predictor
        dev, dt = tk.device, tk.dtype
        self.b = B = slots
        self.nh, self.kvh, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        ccfg = self.cp.config
        self.cnh, self.ckvh = ccfg.num_attention_heads, ccfg.num_key_value_heads
        self.ngroups = cfg.num_code_groups
        self.bucket, self.max_frames = bucket, max_frames
        self.lmax = -(-(max_prompt + max_frames + 2) // bucket) * bucket
        L = cfg.num_hidden_layers
        self.K = torch.zeros(L, B, self.kvh, self.lmax, self.d, device=dev, dtype=dt)
        self.V = torch.zeros_like(self.K)
        self.bias = torch.full((B, self.lmax), float("-inf"), device=dev, dtype=dt)
        self.Kc = torch.zeros(ccfg.num_hidden_layers, B, self.ckvh, self.ngroups, ccfg.head_dim, device=dev, dtype=dt)
        self.Vc = torch.zeros_like(self.Kc)
        self.cpos = [torch.arange(p, p + n, device=dev)[None].expand(B, n) for p, n in [(0, 2)] + [(j, 1) for j in
                                                                                               range(2, self.ngroups)]]  # fmt: skip
        self.ar = torch.arange(B, device=dev)
        self.zero = torch.zeros(B, device=dev, dtype=dt)
        D = cfg.hidden_size
        self.h = torch.zeros(B, 1, D, device=dev, dtype=dt)  # talker hidden of the last step (upstream past_hidden)
        self.code0 = torch.zeros(B, dtype=torch.long, device=dev)
        self.g = torch.zeros(B, dtype=torch.long, device=dev)  # frame being completed / fed
        self.pos = torch.zeros(B, dtype=torch.long, device=dev)  # next cache position (= talker position id)
        self.cap = torch.ones(B, dtype=torch.long, device=dev)
        self.key = torch.zeros(B, dtype=torch.long, device=dev)
        self.done = torch.ones(B, dtype=torch.bool, device=dev)
        self.V_ = cfg.vocab_size
        self.seen = torch.zeros(B, self.V_, dtype=torch.bool, device=dev)
        self.trail = torch.zeros(B, max_text, D, device=dev, dtype=dt)
        self.tlen = torch.zeros(B, dtype=torch.long, device=dev)
        self.out = torch.zeros(B, max_frames + 1, self.ngroups, dtype=torch.long, device=dev)
        self.eos = cfg.codec_eos_token_id
        sup = torch.zeros(self.V_, dtype=torch.bool, device=dev)
        sup[self.V_ - 1024 :] = True
        sup[self.eos] = False
        self.sup = sup
        self.vh = _mix((torch.arange(max(self.V_, ccfg.vocab_size), device=dev) * 0x9E3779B1 + 0x2545F491) & M32)
        self.gen = gen
        self.tts_pad = None
        self.graphs = {}
        self.use_graphs = True

    # ---- sampling (upstream's processor chains; exponential race with hashed noise) ----
    def _noise(self, key, frame: torch.Tensor, cb: int, n: int) -> torch.Tensor:
        """Exp(1) noise [rows, n] from a hash of (utterance key, frame, codebook, token id). 23-bit uniforms: (k + 0.5)
        / 2^23 is exact in float32 and < 1, so the noise is > 0 (24 bits rounded the top value to 1.0 -> noise 0 -> p / 0
        = NaN for a masked token, which argmax picked: a suppressed control id sampled, 2026-11-19)."""
        k = _mix(key ^ ((frame * 16 + cb) & M32))
        x = _mix(k[:, None] ^ self.vh[None, :n])
        return -torch.log(((x >> 9).float() + 0.5) * (1.0 / 8388608))

    def _draw(self, s, do_sample, temp, top_k, top_p, noise):
        if not do_sample:
            return s.argmax(-1)
        if temp != 1.0:
            s = s / temp
        if top_k:
            thr = torch.topk(s, min(top_k, s.shape[-1]))[0][..., -1:]
            s = s.masked_fill(s < thr, float("-inf"))
        if top_p < 1.0:
            srt, idx = torch.sort(s, descending=False)
            rm = srt.softmax(-1).cumsum(-1) <= (1 - top_p)
            rm[..., -1:] = False
            s = s.masked_fill(rm.scatter(1, idx, rm), float("-inf"))
        return (F.log_softmax(s, dim=-1) - noise().log()).argmax(-1)  # log of the race p / E; masked: -inf

    def _pick0(self, lg, seen, g, frame, key=None):
        """Talker logits [n, V] (float) -> first-codebook ids [n]. ``g``: frames already generated - 1."""
        gn = self.gen
        s = lg
        if gn["repetition_penalty"] != 1.0:
            p = gn["repetition_penalty"]
            s = torch.where(seen, torch.where(s < 0, s * p, s / p), s)
        s = s.clone()
        s[:, self.eos] = torch.where(g + 1 < 2, torch.full_like(s[:, 0], float("-inf")), s[:, self.eos])
        s = s.masked_fill(self.sup, float("-inf"))
        return self._draw(s, gn["do_sample"], gn["temperature"], gn["top_k"], gn["top_p"],
                          lambda: self._noise(self.key if key is None else key, frame, 0, s.shape[-1]))  # fmt: skip

    def _pick_sub(self, lg, cb):
        gn = self.gen
        return self._draw(lg, gn["subtalker_dosample"], gn["subtalker_temperature"], gn["subtalker_top_k"],
                          gn["subtalker_top_p"], lambda: self._noise(self.key, self.g, cb, lg.shape[-1]))  # fmt: skip

    # ---- code predictor (upstream's nested generate, unrolled) ----
    def _cp_layers(self, x, p0: int, n: int, pid):
        from qwen_tts.core.models.modeling_qwen3_tts import apply_rotary_pos_emb, repeat_kv

        cpm, B = self.cp.model, self.b
        cos, sin = cpm.rotary_emb(x, pid)
        for i, layer in enumerate(cpm.layers):
            at = layer.self_attn
            hs = layer.input_layernorm(x)
            q = at.q_norm(at.q_proj(hs).view(B, n, -1, at.head_dim)).transpose(1, 2)
            k = at.k_norm(at.k_proj(hs).view(B, n, -1, at.head_dim)).transpose(1, 2)
            v = at.v_proj(hs).view(B, n, -1, at.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            self.Kc[i, :, :, p0 : p0 + n] = k
            self.Vc[i, :, :, p0 : p0 + n] = v
            kk = repeat_kv(self.Kc[i, :, :, : p0 + n], at.num_key_value_groups)
            vv = repeat_kv(self.Vc[i, :, :, : p0 + n], at.num_key_value_groups)
            a = F.scaled_dot_product_attention(q, kk, vv, is_causal=n > 1, scale=at.scaling)
            x = x + at.o_proj(a.transpose(1, 2).reshape(B, n, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return cpm.norm(x)

    def _code_predictor(self):
        """Codebooks 1..15 of the current frame from (talker hidden, first codebook) -> (codes [B, 15], embs)."""
        cp, tk = self.cp, self.tk
        e0 = tk.get_input_embeddings()(self.code0)[:, None]
        x = cp.small_to_mtp_projection(torch.cat([self.h, e0], dim=1))
        hs = self._cp_layers(x, 0, 2, self.cpos[0])
        embs, codes = [e0], []
        for j in range(self.ngroups - 1):
            c = self._pick_sub(cp.lm_head[j](hs[:, -1]).float(), j + 1)
            codes.append(c)
            embs.append(cp.model.codec_embedding[j](c)[:, None])
            if j < self.ngroups - 2:
                x = cp.small_to_mtp_projection(embs[-1])
                hs = self._cp_layers(x, j + 2, 1, self.cpos[j + 1])
        return torch.stack(codes, 1), torch.cat(embs, dim=1)

    # ---- talker ----
    def _talker(self, x, w):
        from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

        tm, B = self.tm, self.b
        self.bias.index_put_((self.ar, self.pos), self.zero)
        cos, sin = tm.rotary_emb(x, self.pos.view(1, B, 1).expand(3, B, 1))
        nrep = self.nh // self.kvh
        for i, layer in enumerate(tm.layers):
            at = layer.self_attn
            hs = layer.input_layernorm(x)
            q = at.q_norm(at.q_proj(hs).view(B, 1, -1, self.d)).transpose(1, 2)
            k = at.k_norm(at.k_proj(hs).view(B, 1, -1, self.d)).transpose(1, 2)
            v = at.v_proj(hs).view(B, 1, -1, self.d).transpose(1, 2)
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, at.rope_scaling["mrope_section"],
                                                   at.rope_scaling["interleaved"])  # fmt: skip
            self.K[i][self.ar, :, self.pos] = k[:, :, 0]
            self.V[i][self.ar, :, self.pos] = v[:, :, 0]
            # GQA without copying the cache: the nrep query heads of one KV head as nrep query positions
            a = F.scaled_dot_product_attention(q.reshape(B, self.kvh, nrep, self.d), self.K[i][:, :, :w],
                                               self.V[i][:, :, :w], attn_mask=self.bias[:, None, None, :w],
                                               scale=at.scaling)  # fmt: skip
            x = x + at.o_proj(a.reshape(B, 1, self.nh * self.d))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return tm.norm(x)

    def _step(self, w):
        live = ~self.done
        codes, embs = self._code_predictor()
        self.out[self.ar, self.g.clamp(max=self.max_frames)] = torch.cat([self.code0[:, None], codes], 1)
        txt = self.trail[self.ar, self.g.clamp(max=self.trail.shape[1] - 1)]
        txt = torch.where((self.g < self.tlen)[:, None], txt, self.tts_pad)
        x = embs.sum(1, keepdim=True) + txt[:, None]
        h = self._talker(x, w)
        nxt = self._pick0(self.tk.codec_head(h[:, -1]).float(), self.seen, self.g, self.g + 1)
        self.seen.scatter_(1, nxt[:, None], True)
        self.h.copy_(h)
        self.code0.copy_(torch.where(live, nxt, self.code0))
        self.g += live.long()
        self.pos += live.long()
        self.done |= (nxt == self.eos) | (self.g >= self.cap)

    def _graph(self, w):
        if w not in self.graphs:
            state = (self.bias, self.h, self.code0, self.g, self.pos, self.done, self.seen, self.out)
            snap = [t.clone() for t in state]
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                self._step(w)
            torch.cuda.current_stream().wait_stream(st)
            for t, s in zip(state, snap):
                t.copy_(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._step(w)
            self.graphs[w] = g
        return self.graphs[w]

    def precapture(self):
        """Capture every attention window's step graph up front (before a side stream starts decoding)."""
        for w in range(self.bucket, self.lmax + 1, self.bucket):
            self._graph(w)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def _refill(self, slots, prompts, seeds, caps):
        """Prefill the new utterances (one left-padded batch, upstream's position ids), sample their first frame's
        first codebook, move each into its slot."""
        tk, tm = self.tk, self.tm
        dev = tk.device
        m, n = len(prompts), max(p[0].shape[0] for p in prompts)
        emb = prompts[0][0].new_zeros(m, n, prompts[0][0].shape[-1])
        mask = torch.zeros(m, n, dtype=torch.long, device=dev)
        for j, (e, *_) in enumerate(prompts):
            emb[j, n - e.shape[0] :] = e
            mask[j, n - e.shape[0] :] = 1
        pid = mask.cumsum(-1) - 1
        pid = pid.masked_fill(mask == 0, 1)
        out = tm(inputs_embeds=emb, attention_mask=mask, position_ids=pid[None].expand(3, -1, -1), use_cache=True)
        hid = out.last_hidden_state
        cache = out.past_key_values
        kv = [(cache.layers[i].keys, cache.layers[i].values) if hasattr(cache, "layers") else cache[i]
              for i in range(len(tm.layers))]  # fmt: skip
        key = _mix((torch.tensor([int(s) & M32 for s in seeds], device=dev) * 0x9E3779B1 + 0x632BE5AB) & M32)
        seen = torch.zeros(m, self.V_, dtype=torch.bool, device=dev)
        neg = torch.full((m,), -1, dtype=torch.long, device=dev)
        first = self._pick0(tk.codec_head(hid[:, -1]).float(), seen, neg, torch.zeros_like(neg), key)
        for j, s in enumerate(slots):
            e, t = prompts[j][:2]
            nr = e.shape[0]
            for i, (k, v) in enumerate(kv):
                self.K[i, s, :, :nr] = k[j, :, n - nr :]
                self.V[i, s, :, :nr] = v[j, :, n - nr :]
            self.bias[s] = float("-inf")
            self.bias[s, :nr] = 0.0
            self.h[s] = hid[j, -1:]
            self.code0[s] = first[j]
            self.g[s] = 0
            self.pos[s] = nr
            self.cap[s] = caps[j]
            self.key[s] = key[j]
            self.seen[s] = False
            self.seen[s, first[j]] = True
            self.trail[s, : t.shape[0]] = t
            self.tlen[s] = t.shape[0]
            self.done[s] = caps[j] <= 0

    @torch.inference_mode()
    def run(self, prompts, seeds, caps, on_done=None, harvest_every: int = 8, order=None):
        """Decode every utterance (``order``: queue order, default longest prompt + text first); ``on_done(i,
        frames)`` as each finishes (else the frames are returned)."""
        q = list(order) if order is not None else sorted(range(len(prompts)), key=lambda i: -caps[i])
        who, res, filled = [None] * self.b, [None] * len(prompts), [0] * self.b

        def refill(free):
            take = [(s, q.pop(0)) for s in free[: len(q)]]
            if take:
                self._refill([s for s, _ in take], [prompts[i] for _, i in take], [seeds[i] for _, i in take],
                             [caps[i] for _, i in take])  # fmt: skip
                for s, i in take:
                    who[s] = i
                    filled[s] = prompts[i][0].shape[0]

        refill(list(range(self.b)))
        while any(w is not None for w in who):
            w = min(self.lmax, -(-(max(filled[s] for s in range(self.b) if who[s] is not None) + harvest_every)
                                 // self.bucket) * self.bucket)  # fmt: skip
            if self.K.is_cuda and self.use_graphs:
                g = self._graph(w)
                for _ in range(harvest_every):
                    g.replay()
            else:  # CPU / debugging: eager steps
                for _ in range(harvest_every):
                    self._step(w)
            done = self.done.tolist()
            free = []
            for s in range(self.b):
                if who[s] is None:
                    free.append(s)
                    continue
                filled[s] += harvest_every
                if done[s]:
                    fr = self.out[s, : int(self.g[s])].clone()
                    i, who[s] = who[s], None
                    if on_done is not None:
                        on_done(i, fr)
                    else:
                        res[i] = fr
                    free.append(s)
            if q and free:
                refill(free)
        return res


def gen_kwargs(model, **over) -> dict:
    """Upstream's sampling settings: explicit values, else the checkpoint's ``generate_config.json``, else upstream's
    hard defaults (``Qwen3TTSModel._merge_generate_kwargs``)."""
    return model._merge_generate_kwargs(**over)


def decode(model, codes: list[torch.Tensor], ref_codes: list[torch.Tensor]) -> list[np.ndarray]:
    """Upstream's decode for a group: reference codes + generated codes through ``speech_tokenizer.decode`` (one
    padded batch), the reference part cut off proportionally. A timbre-only prompt has no reference codes (None)."""
    full = [c if r is None else torch.cat([r.to(c.device), c], dim=0) for r, c in zip(ref_codes, codes)]
    wavs, _ = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in full])
    out = []
    for w, r, c in zip(wavs, ref_codes, full):
        cut = 0 if r is None else int(int(r.shape[0]) / max(int(c.shape[0]), 1) * w.shape[0])
        out.append(w[cut:])
    return out


def synthesize(model, items: list[dict], *, slots: int = 64, dec_batch: int = 8, dec_pool: int = 64,
               overlap: bool = True, gen: dict | None = None, timing: dict | None = None,
               codes_out: list | None = None, log_every: float = 60.0) -> tuple[list[np.ndarray], int]:  # fmt: skip
    """Waveforms (float32 numpy) and the sample rate for ``items`` = [{"text", "language", "prompt" (``prompt()``),
    "seed", optional "max_frames"}]. ``gen``: overrides of upstream's sampling settings (``gen_kwargs``), e.g.
    ``do_sample=False, subtalker_dosample=False`` for greedy. ``codes_out``: filled with each item's frames [n, 16]."""
    import queue
    import threading

    if not items:
        return [], 24000
    tm = timing if timing is not None else {}
    t0 = time.time()
    gk = gen_kwargs(model, **(gen or {}))
    prompts = [prompt_embeds(model, it["text"], it["language"], it["prompt"]) for it in items]
    caps = [int(it.get("max_frames") or default_max_frames(it["text"], it["language"])) for it in items]
    tm["prompt_s"] = time.time() - t0
    dec = ContinuousTalker(model, min(slots, len(items)), max_prompt=max(p[0].shape[0] for p in prompts),
                           max_text=max(p[1].shape[0] for p in prompts), max_frames=max(caps), gen=gk)  # fmt: skip
    dec.tts_pad = prompts[0][2]
    out: list = [None] * len(items)
    if codes_out is not None:
        codes_out[:] = [None] * len(items)
    pend = []
    tm.setdefault("decode_s", 0.0)
    last = [time.time()]
    sr = int(model.model.speech_tokenizer.get_output_sample_rate())

    def run_dec(batch):
        ts = time.time()
        batch.sort(key=lambda p: p[1].shape[0])
        for a in range(0, len(batch), dec_batch):
            grp = batch[a : a + dec_batch]
            ws = decode(model, [c for _, c in grp], [items[i]["prompt"].ref_code for i, _ in grp])
            for (i, _), w in zip(grp, ws):
                out[i] = np.asarray(w, np.float32)
        tm["decode_s"] += time.time() - ts
        if time.time() - last[0] > log_every:
            last[0] = time.time()
            print(f"[qwen3_tts_fast] {sum(o is not None for o in out)}/{len(items)} utterances, "
                  f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip

    jq: queue.Queue = queue.Queue()
    err = []

    def worker():
        side = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(side):
            while (job := jq.get()) is not None:
                ev, batch = job
                side.wait_event(ev)
                try:
                    run_dec(batch)
                except Exception as e:  # noqa: BLE001 -- re-raised by the caller
                    err.append(e)

    def flush():
        if not pend:
            return
        batch = list(pend)
        pend.clear()
        if not overlap:
            run_dec(batch)
            return
        ev = torch.cuda.Event()
        ev.record()
        jq.put((ev, batch))

    nbins = model.model.talker.code_predictor.config.vocab_size

    def done(i, fr):
        if fr.numel() and (int(fr.max()) >= nbins or int(fr.min()) < 0):
            raise RuntimeError(f"qwen3_tts_fast: item {i} has codec ids outside [0, {nbins})")
        if codes_out is not None:
            codes_out[i] = fr.cpu()
        pend.append((i, fr))
        if len(pend) >= dec_pool:
            flush()

    th = None
    t1 = time.time()
    overlap = overlap and torch.cuda.is_available() and dec.K.is_cuda
    if overlap:
        dec.precapture()
        th = threading.Thread(target=worker, daemon=True)
        th.start()
    tm["capture_s"] = time.time() - t1
    try:
        dec.run(prompts, [it["seed"] for it in items], caps, on_done=done)
        flush()
    finally:
        if th is not None:
            jq.put(None)
            th.join()
    if err:
        raise err[0]
    tm["total_s"] = time.time() - t0
    return out, sr
