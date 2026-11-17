"""Batched Chatterbox Multilingual synthesis (German, Chinese, ... 23 languages): many utterances at once.

Upstream ``ChatterboxMultilingualTTS.generate`` runs one utterance at a time: T3 (Llama, 0.5B) decodes the
classifier-free-guidance pair (conditional row + the row with its text embedding zeroed) at batch 2, then S3Gen
(10-step flow matching with CFG + HiFiGAN). The reference here is upstream's GitHub master (2026-11-04, V3 weights
``t3_mtl23ls_v3``): no alignment analyzer (the PyPI 0.1.7 analyzer force-ends utterances on any two equal consecutive
speech tokens; removed upstream in #516), repetition penalty 1.2, the final speech token's audio dropped. Only the
model modules of the installed ``chatterbox-tts`` (PyPI 0.1.7; tokenizer / S3Gen code identical to master) are used.

* ``ContinuousMtlT3``: continuous batching as ``chatterbox_fast.ContinuousT3``, each utterance a CFG row pair over a
  static bf16 KV cache, the step captured as a CUDA graph per 128-position window. Sampling is upstream's chain (CFG
  mix, repetition penalty over the ids incl. the start token, temperature, min-p, top-p) with the exponential-race
  draw from a generator seeded per utterance.
* ``chatterbox_fast.s3gen_batch`` with the regular S3Gen settings (10 steps, CFG, no appended silence tokens), in
  groups of 16 (faster than 32 here: the 10-step flow decoder is ~85% of S3Gen's time), the flow estimator in bf16
  weights (``speed_s3gen``).
* No watermark.

Measured (German FLEURS, one H100, 2026-11-17): 84.7x realtime vs upstream V3's 3.5x, Parakeet WER vs upstream
+0.005 [95% -0.016, +0.025] on 96 sentences; peak ~39-49 GB, and the card holds Parakeet transcribing alongside
(62 GB peak together). Chinese: Whisper CER 0.003-0.005 vs upstream 0.003-0.009 (8 sentences).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from . import chatterbox_fast as cf
except ImportError:  # run as the worker's sibling module
    import chatterbox_fast as cf

REPO = "ResembleAI/chatterbox"
T3_FILES = {"v2": "t3_mtl23ls_v2.safetensors", "v3": "t3_mtl23ls_v3.safetensors"}
SOS, EOS = 6561, 6562  # S3 speech start / stop (``chatterbox.models.s3tokenizer``)
TOKEN_SAMPLES = 24000 // 25  # S3Gen output samples per speech token


def load_mtl(device: str = "cuda", t3: str = "v3"):
    """``ChatterboxMultilingualTTS`` with the ``t3`` weights (upstream master's ``from_pretrained(t3_model=...)``)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

    t3_file = T3_FILES[t3]
    ckpt = Path(snapshot_download(REPO, allow_patterns=["ve.pt", t3_file, "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
                                                        "conds.pt", "Cangjie5_TC.json"]))  # fmt: skip
    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(ckpt / "ve.pt", weights_only=True))
    ve.to(device).eval()
    m = T3(T3Config.multilingual())
    sd = load_file(ckpt / t3_file)
    m.load_state_dict(sd["model"][0] if "model" in sd else sd)
    m.to(device).eval()
    s3 = S3Gen()
    s3.load_state_dict(torch.load(ckpt / "s3gen.pt", weights_only=True))
    s3.to(device).eval()
    tok = MTLTokenizer(str(ckpt / "grapheme_mtl_merged_expanded_v1.json"))
    fix_cangjie(tok)
    conds = Conditionals.load(ckpt / "conds.pt").to(device) if (ckpt / "conds.pt").exists() else None
    return ChatterboxMultilingualTTS(m, s3, ve, tok, device, conds=conds)


def fix_cangjie(tok) -> None:
    """The tokenizer looks the Chinese Cangjie table up with ``cache_dir=<the checkpoint snapshot dir>``, which is not a
    hub cache: offline (our jobs) that fails with only a log warning, Chinese characters then become ``[UNK]`` and
    Chinese speech is gibberish (Whisper CER ~1.0, 2026-11-17). Reload it from the regular hub cache; fail if empty."""
    cj = tok.cangjie_converter
    if not cj.word2cj:
        cj._load_cangjie_mapping(None)
    if not cj.word2cj:
        raise RuntimeError("Chatterbox Cangjie5_TC.json not loadable: Chinese would synthesize as [UNK] tokens")


def speed_s3gen(model) -> None:
    """``chatterbox_fast.speed_s3gen`` plus the flow estimator in bf16 weights (``solve_euler`` casts its inputs to the
    estimator's dtype and the result back). German, 288 sentences: 84.7x vs 74.9x realtime under bf16 autocast,
    Parakeet WER vs upstream +0.005 [95% -0.016, +0.025]; compiling the estimator on top gave nothing (85.9x)."""
    cf.speed_s3gen(model)
    model.s3gen.flow.decoder.estimator.to(torch.bfloat16)


def text_tokens(model, text: str, language_id: str) -> torch.Tensor:
    """Upstream's text path: ``punc_norm``, the multilingual tokenizer (language tag), start / stop text tokens."""
    from chatterbox.mtl_tts import punc_norm

    t = model.tokenizer.text_to_tokens(punc_norm(text), language_id=language_id.lower())[0]
    hp = model.t3.hp
    return F.pad(F.pad(t, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)


def finish_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Upstream's ``drop_invalid_tokens`` (after the last-generated start token, before the stop token) -> (tokens S3Gen
    can embed, the count upstream trims the waveform by). Ids >= 6561 other than start / stop would crash upstream's
    S3Gen; they are dropped here."""
    hit = (tokens == SOS).nonzero()
    if len(hit):
        tokens = tokens[int(hit[0]) + 1 :]
    hit = (tokens == EOS).nonzero()
    if len(hit):
        tokens = tokens[: int(hit[0])]
    return tokens[tokens < cf.S3_VOCAB], int(tokens.numel())


class ContinuousMtlT3:
    """Continuous-batching multilingual T3 decode (module docstring). ``run`` returns each utterance's sampled tokens up
    to (excluding) its stop token, at most ``max_gen_len``."""

    def __init__(self, t3, slots: int, max_prompt: int, *, cfg_weight=0.5, temperature=0.8, min_p=0.05, top_p=1.0,
                 repetition_penalty=1.2, max_gen_len=1000, bucket: int = 128, kv_dtype=torch.bfloat16):  # fmt: skip
        tf = t3.tfmr
        self.t3, self.tf, self.b, self.r = t3, tf, slots, 2 * slots
        dev = t3.device
        cfg = tf.config
        self.h, self.d = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        assert cfg.num_key_value_heads == self.h, "grouped KV heads: not handled"
        self.max_gen, self.bucket = max_gen_len, bucket
        self.lmax = -(-(max_prompt + max_gen_len + 2) // bucket) * bucket
        L = len(tf.layers)
        self.K = torch.zeros(L, self.r, self.h, self.lmax, self.d, device=dev, dtype=kv_dtype)
        self.V = torch.zeros_like(self.K)
        self.bias = torch.full((self.r, self.lmax), float("-inf"), device=dev, dtype=kv_dtype)
        self.ar = torch.arange(slots, device=dev)
        self.arR = torch.arange(self.r, device=dev)
        self.zeros = torch.zeros(self.r, device=dev, dtype=kv_dtype)
        self.c = torch.zeros(slots, dtype=torch.long, device=dev)  # next cache position (= next position id)
        self.si = torch.zeros(slots, dtype=torch.long, device=dev)  # tokens generated so far
        self.tok = torch.zeros(slots, 1, dtype=torch.long, device=dev)
        vocab = t3.speech_head.out_features
        self.seen = torch.zeros(slots, vocab, dtype=torch.bool, device=dev)
        self.noise = torch.ones(slots, max_gen_len + 1, vocab, device=dev)
        self.stop = t3.hp.stop_speech_token
        self.out = torch.full((slots, max_gen_len + 1), self.stop, dtype=torch.long, device=dev)
        self.done = torch.ones(slots, dtype=torch.bool, device=dev)  # empty slots count as done
        self.cfg, self.temp, self.minp, self.topp, self.pen = cfg_weight, temperature, min_p, top_p, repetition_penalty
        self.graphs = {}

    def _pick(self, lg, seen, noise):
        """CFG pair logits [2n, V] -> next tokens [n, 1]: upstream's processor chain, exponential-race draw."""
        lg = lg.float()
        cond, unc = lg[0::2], lg[1::2]
        s = cond + self.cfg * (cond - unc)
        if self.pen != 1.0:
            s = torch.where(seen, torch.where(s < 0, s * self.pen, s / self.pen), s)
        if self.temp != 1.0:
            s = s / self.temp
        if self.minp > 0:
            p = F.softmax(s, dim=-1)
            s = s.masked_fill(p < self.minp * p.amax(-1, keepdim=True), float("-inf"))
        if self.topp < 1.0:
            srt, idx = torch.sort(s, descending=False)
            rm = srt.softmax(-1).cumsum(-1) <= (1 - self.topp)
            rm[..., -1:] = False
            s = s.masked_fill(rm.scatter(1, idx, rm), float("-inf"))
        return (F.softmax(s, dim=-1) / noise).argmax(-1, keepdim=True)

    def _step(self, w):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        t3, tf, r = self.t3, self.tf, self.r
        live = ~self.done
        cR = self.c.repeat_interleave(2)
        self.bias.index_put_((self.arR, cR), self.zeros)
        x = t3.speech_emb(self.tok) + t3.speech_pos_emb.emb(self.si)[:, None]  # [B, 1, D]
        x = x.repeat_interleave(2, dim=0)
        cos, sin = tf.rotary_emb(x, cR[:, None])
        kd = self.K.dtype
        for i, layer in enumerate(tf.layers):
            at = layer.self_attn
            hs = layer.input_layernorm(x)
            q, k, v = (p(hs).view(r, 1, self.h, self.d).transpose(1, 2) for p in (at.q_proj, at.k_proj, at.v_proj))
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            self.K[i][self.arR, :, cR] = k[:, :, 0].to(kd)
            self.V[i][self.arR, :, cR] = v[:, :, 0].to(kd)
            a = F.scaled_dot_product_attention(q.to(kd), self.K[i][:, :, :w], self.V[i][:, :, :w],
                                               attn_mask=self.bias[:, None, None, :w])  # fmt: skip
            x = x + at.o_proj(a.to(x.dtype).transpose(1, 2).reshape(r, 1, self.h * self.d))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        lg = t3.speech_head(tf.norm(x))[:, -1, :]
        nxt = self._pick(lg, self.seen, self.noise[self.ar, self.si])
        nxt = torch.where(live[:, None], nxt, torch.full_like(nxt, self.stop))
        self.out[self.ar, self.si] = nxt[:, 0]
        self.seen.scatter_(1, nxt, True)
        self.tok.copy_(nxt)
        self.c += live.long()
        self.si.add_(live.long()).clamp_(max=self.max_gen)
        self.done |= (nxt[:, 0] == self.stop) | (self.si >= self.max_gen)

    def _graph(self, w):
        if w not in self.graphs:
            state = (self.bias, self.tok, self.seen, self.done, self.out, self.c, self.si)
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
        """Capture the decode graph of every attention window up front (before another stream starts work)."""
        for w in range(self.bucket, self.lmax + 1, self.bucket):
            self._graph(w)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def _refill(self, slots: list[int], conds: list, texts: list[torch.Tensor], seeds: list[int]):
        """Prefill the new utterances' CFG pairs (one left-padded batch: conditioning, text, start token at speech
        position 0, and upstream's appended start token) and move each pair into its slot."""
        t3, dev = self.t3, self.t3.device
        sos = torch.full((2, 1), t3.hp.start_speech_token, dtype=torch.long, device=dev)
        bos = t3.speech_emb(sos[:1]) + t3.speech_pos_emb.get_fixed_embedding(0)
        rows = []
        for c, t in zip(conds, texts):
            e, _ = t3.prepare_input_embeds(t3_cond=c, text_tokens=t.view(1, -1).to(dev).expand(2, -1).clone(),
                                           speech_tokens=sos, cfg_weight=self.cfg)  # fmt: skip
            e = torch.cat([e, bos.expand(2, -1, -1)], dim=1)
            rows += [e[0], e[1]]
        m, n = len(rows), max(x.shape[0] for x in rows)
        emb = rows[0].new_zeros(m, n, rows[0].shape[-1])
        mask = torch.zeros(m, n, dtype=torch.long, device=dev)
        for i, x in enumerate(rows):
            emb[i, n - x.shape[0] :] = x
            mask[i, n - x.shape[0] :] = 1
        out = self.tf(inputs_embeds=emb, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                      use_cache=True)  # fmt: skip
        cache = out.past_key_values
        layers = cache.layers if hasattr(cache, "layers") else None
        kv = [
            (layers[i].keys, layers[i].values) if layers is not None else cache[i] for i in range(len(self.tf.layers))
        ]
        lg = t3.speech_head(out[0][:, -1])
        seen = torch.zeros(len(slots), self.seen.shape[1], dtype=torch.bool, device=dev)
        seen[:, t3.hp.start_speech_token] = True
        noise = []
        for s, sd in zip(slots, seeds):
            g = torch.Generator(device=dev).manual_seed(int(sd))
            self.noise[s].exponential_(generator=g)
            noise.append(self.noise[s, 0])
        first = self._pick(lg, seen, torch.stack(noise))[:, 0]
        for j, s in enumerate(slots):
            nr = rows[2 * j].shape[0]
            for rr in (2 * s, 2 * s + 1):
                src = 2 * j + (rr - 2 * s)
                for i, (k, v) in enumerate(kv):
                    self.K[i, rr, :, :nr] = k[src, :, n - nr :].to(self.K.dtype)
                    self.V[i, rr, :, :nr] = v[src, :, n - nr :].to(self.V.dtype)
                self.bias[rr] = float("-inf")
                self.bias[rr, :nr] = 0.0
            f = int(first[j])
            self.c[s] = nr
            self.seen[s] = seen[j]
            self.seen[s, f] = True
            self.tok[s, 0] = f
            self.out[s] = self.stop
            self.out[s, 0] = f
            self.si[s] = 1
            self.done[s] = f == self.stop

    @torch.inference_mode()
    def run(self, conds: list, texts: list[torch.Tensor], seeds: list[int], on_done=None, harvest_every: int = 8):
        """Decode every utterance, longest text first; ``on_done(i, tokens)`` as each finishes (else returned)."""
        q = sorted(range(len(texts)), key=lambda i: -texts[i].numel())
        who = [None] * self.b
        res = [None] * len(texts)
        filled = [0] * self.b

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
                    toks = row[: int(hit[0]) + 1].clone() if len(hit) else row[: self.max_gen].clone()
                    i, who[s] = who[s], None
                    if on_done is not None:
                        on_done(i, toks)
                    else:
                        res[i] = toks
                    free.append(s)
            if q and free:
                refill(free)
        return res


def synthesize(model, items: list[dict], *, slots: int = 64, s3_batch: int = 16, s3_pool: int = 64,
               log_every: float = 60.0, timing: dict | None = None, flow_autocast: torch.dtype | None = None,
               overlap: bool = True) -> list:  # fmt: skip
    """Waveforms (float32 numpy at ``model.sr``) for ``items`` = [{"text", "lang", "cond" (a ``model.conds``),
    "seed"}]: ``ContinuousMtlT3``, finished utterances pooled ``s3_pool`` at a time, sorted by length, S3Gen in groups
    of ``s3_batch``, each waveform cut to upstream's length (the final token's audio dropped). ``overlap``: S3Gen runs
    in a second thread on its own CUDA stream while T3 keeps decoding (every decode graph is captured first, so no
    capture runs concurrently with it)."""
    import queue
    import threading

    if not items:
        return []
    toks = [text_tokens(model, it["text"], it["lang"]) for it in items]
    dec = ContinuousMtlT3(model.t3, min(slots, len(items)),
                          max_prompt=model.t3.hp.speech_cond_prompt_len + max(t.numel() for t in toks) + 8)  # fmt: skip
    out = [None] * len(items)
    pend = []
    t0, last = time.time(), [time.time()]
    tm = timing if timing is not None else {}
    tm.setdefault("s3_s", 0.0)

    def s3(batch):
        ts = time.time()
        batch.sort(key=lambda p: p[1].numel())
        for a in range(0, len(batch), s3_batch):
            grp = batch[a : a + s3_batch]
            ws = cf.s3gen_batch(model, [t for _, t, _ in grp], [items[i]["cond"] for i, _, _ in grp],
                                [items[i]["seed"] for i, _, _ in grp], sil_tokens=0, n_timesteps=10,
                                meanflow=False, flow_autocast=flow_autocast)  # fmt: skip
            for (i, _, n), w in zip(grp, ws):
                out[i] = w[: max(1, n - 1) * TOKEN_SAMPLES]
        tm["s3_s"] += time.time() - ts
        if time.time() - last[0] > log_every:
            last[0] = time.time()
            print(f"[chatterbox_fast_mtl] {sum(o is not None for o in out)}/{len(items)} utterances, "
                  f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip

    q: queue.Queue = queue.Queue()
    err = []

    def worker():
        side = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(side):
            while (job := q.get()) is not None:
                ev, batch = job
                side.wait_event(ev)
                try:
                    s3(batch)
                except Exception as e:  # noqa: BLE001 -- re-raised by the caller
                    err.append(e)

    def flush():
        if not pend:
            return
        batch = list(pend)
        pend.clear()
        if not overlap:
            s3(batch)
            return
        ev = torch.cuda.Event()
        ev.record()
        q.put((ev, batch))

    def done(i, t):
        t, n = finish_tokens(t)
        pend.append((i, t, n))
        if len(pend) >= s3_pool:
            flush()

    th = None
    if overlap:
        dec.precapture()
        th = threading.Thread(target=worker, daemon=True)
        th.start()
    try:
        dec.run([it["cond"].t3 for it in items], toks, [it["seed"] for it in items], on_done=done)
        flush()
    finally:
        if th is not None:
            q.put(None)
            th.join()
    if err:
        raise err[0]
    tm["total_s"] = time.time() - t0
    return out
