"""OWSM-CTC (ESPnet S2T-CTC) adapter for the grad-based forced-alignment pipeline.

Model: ``espnet/owsm_ctc_v4_1B`` -- a GENERAL graphemic (BPE-50k) transcription CTC, the
representative general CTC (vs torchaudio MMS_FA = an aligner, vitouphy/POWSM = phonetic).
Encoder is a self-conditioned-CTC E-Branchformer with an OWSM prompt; loaded via the CTC variant
of ``S2TTask`` (``espnet2.tasks.s2t_ctc``, NOT the AED ``espnet2.tasks.s2t``).

Per-token score is the CTC forward PARTIAL score (``_ctc_partial_scores``), differentiable w.r.t.
the emission log-probs -- the same scoring as the Wav2Vec2-CTC adapter. Grad target = the log-mel
frontend output (128-mel, 100 Hz), the OWSM analog of Whisper's log-mel leaf. The CTC emission grid
is ~12.5 Hz (80 ms/frame; conv2d8 subsampling). Subword (BPE) scoring grouped to words via ``▁``.
Batch 1.

ESPnet env plumbing mirrors the OWLS adapter (see reference_owls_espnet_env): add a local deps dir
to sys.path and stub numba + a few TTS-only deps BEFORE importing espnet.
"""

from __future__ import annotations

import os
import sys
import glob
import types
import importlib.machinery
import time
from typing import Optional, Union, List

import numpy as np
import torch

from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from .base import BaseModelInterface, ForwardOutput

# --- ESPnet import plumbing (must run before `import espnet*`) ----------------------
_OWLS_DEPS = "/home/az668407/work/owls-deps"
if _OWLS_DEPS not in sys.path:
    sys.path.insert(0, _OWLS_DEPS)

if "numba" not in sys.modules:
    _nb = types.ModuleType("numba")

    def _noop_jit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f

    _nb.jit = _nb.njit = _nb.generated_jit = _noop_jit
    _nb.prange = range
    sys.modules["numba"] = _nb


def _stub_mod(name: str):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []
    m.__getattr__ = lambda attr: __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    sys.modules[name] = m
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(_stub_mod(parent), child, m)
    return m


for _m in [
    "g2p_en",
    "pyopenjtalk",
    "pypinyin",
    "jamo",
    "phonemizer",
    "g2pk2",
    "g2pk",
    "jaconv",
    "tacotron_cleaner",
    "tacotron_cleaner.cleaners",
]:
    _stub_mod(_m)


class OwsmCtc(BaseModelInterface):
    """ESPnet OWSM-CTC (graphemic BPE-50k S2T-CTC). See module docstring."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        language: str = "eng",
        version: int = 1,
        layer: Optional[int] = None,
    ):
        """:param layer: emit from this inter-CTC self-conditioning block (one of interctc_layer_idx,
        e.g. 6/12/15/21) instead of the final encoder output. None = final (the default; keeps the
        existing finished jobs' hashes since the kwarg is then not passed to build_dict)."""
        super().__init__()
        assert version >= 2, "v1 used the encoder frame count (not the log-mel leaf grid) as the alignment time axis"
        self.device = device
        self.model_dir = model_dir
        self.language = language
        self.version = version
        self.layer = layer

        print("Import / load OWSM-CTC (ESPnet S2T-CTC)...")
        start_time = time.time()
        from espnet2.tasks.s2t_ctc import S2TTask  # CTC variant (name clash with the AED S2TTask)

        snap = get_content_dir_from_hub_cache_dir(model_dir)
        cfg = glob.glob(os.path.join(snap, "exp*", "s2t_train_*", "config.yaml"))
        assert len(cfg) == 1, f"expected 1 config.yaml under {snap}/exp*, got {cfg}"
        ckpts = glob.glob(os.path.join(snap, "exp*", "s2t_train_*", "*.pth"))
        ckpts = [c for c in ckpts if "ave" in os.path.basename(c)] or ckpts
        assert ckpts, f"no .pth under {snap}/exp*"
        ckpt = sorted(ckpts, key=len)[0]

        _prev_cwd = os.getcwd()
        os.chdir(snap)
        try:
            self.model, train_args = S2TTask.build_model_from_file(cfg[0], ckpt, str(device))
        finally:
            os.chdir(_prev_cwd)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        token_list = list(train_args.token_list)
        self._tok2id = {t: i for i, t in enumerate(token_list)}
        self.token_list = token_list
        self.vocab_size = len(token_list)
        self.blank_idx = self._tok2id.get("<blank>", 0)

        from espnet2.text.build_tokenizer import build_tokenizer

        bpemodels = glob.glob(os.path.join(snap, "data", "token_list", "*", "bpe.model"))
        assert bpemodels, f"no bpe.model under {snap}/data/token_list"
        self._tokenizer = build_tokenizer(token_type="bpe", bpemodel=bpemodels[0])

        # OWSM-CTC prompt: prefix = <lang><task> (ASR); prev-context = <na> (none). Both are audio-
        # independent -> precompute the encoder-side prompt tensors once (constant across utterances).
        def _sid(tok):
            assert tok in self._tok2id, f"prompt token {tok!r} not in OWSM-CTC vocab"
            return self._tok2id[tok]

        from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

        self._make_pad_mask = make_pad_mask
        _na = torch.tensor([[_sid("<na>")]], dtype=torch.long, device=device)
        _prefix = torch.tensor([[_sid(f"<{language}>"), _sid("<asr>")]], dtype=torch.long, device=device)
        with torch.no_grad():
            _mem, _mlen, _ = self.model.prompt_encoder(self.model.pos_enc(self.model.embed(_na)), torch.tensor([1]))
            self._memory = self.model.prompt_proj(_mem)
            self._memory_mask = (~make_pad_mask(_mlen)[:, None, :]).to(device)
            self._prefix_embeds = self.model.embed_proj(self.model.embed(_prefix))
        print(f"  ({time.time() - start_time:.1f}s) vocab={self.vocab_size} blank={self.blank_idx}")

    def _ctc_partial_scores(self, lp: torch.Tensor, target_ids: List[int]) -> torch.Tensor:
        """CTC forward partial scores Delta_i, differentiable w.r.t. ``lp`` ([T,V]). See Wav2Vec2Ctc."""
        device, dtype = lp.device, lp.dtype
        T = lp.shape[0]
        blank = self.blank_idx
        neg = -1.0e9
        ext: List[int] = [blank]
        for c in target_ids:
            ext.append(int(c))
            ext.append(blank)
        sx = len(ext)
        ext_t = torch.tensor(ext, device=device, dtype=torch.long)
        emit = lp[:, ext_t]  # [T, sx]
        same = torch.zeros(sx, dtype=torch.bool, device=device)
        same[2:] = ext_t[2:] == ext_t[:-2]
        log_state = torch.full((sx,), neg, device=device, dtype=dtype)
        log_state[0] = emit[0, 0]
        if sx > 1:
            log_state[1] = emit[0, 1]
        acc = log_state.clone()
        for t in range(1, T):
            stay = log_state
            prev1 = torch.cat([torch.full((1,), neg, device=device, dtype=dtype), log_state[:-1]])
            prev2 = torch.cat([torch.full((2,), neg, device=device, dtype=dtype), log_state[:-2]])
            prev2 = torch.where(same, prev2, torch.full_like(prev2, neg))
            log_state = emit[t] + torch.logsumexp(torch.stack([stay, prev1, prev2], 0), 0)
            acc = torch.logaddexp(acc, log_state)
        # telescoping difference: state after token i (odd index 2i+1) minus the blank before it.
        partial = []
        for i in range(len(target_ids)):
            partial.append(acc[2 * i + 1])
        return torch.stack(partial) if partial else lp.new_zeros(0)

    def _encode(self, wav: torch.Tensor, n_samples: int):
        """log-mel leaf -> normalize -> prompt-conditioned self-cond-CTC encoder -> CTC emission logits.
        Returns (leaf [1,T,128], logits [1,T_enc,V], T_enc)."""
        speech = wav.to(self.device)[None]
        slens = torch.tensor([n_samples], device=self.device)
        with torch.no_grad():
            feats, flens = self.model._extract_feats(speech, slens)  # [1, T, 128] log-mel @100Hz
        leaf = feats.detach().requires_grad_(True)
        leaf.retain_grad()
        with torch.enable_grad():
            nfeats, nflens = (
                self.model.normalize(leaf, flens) if getattr(self.model, "normalize", None) else (leaf, flens)
            )
            enc_out, enc_lens, _ = self.model.encoder(
                nfeats,
                nflens,
                ctc=self.model.ctc,
                prefix_embeds=self._prefix_embeds,
                memory=self._memory,
                memory_mask=self._memory_mask,
            )
            if isinstance(enc_out, tuple):  # self-conditioned CTC returns (final, [(layer_idx, hidden), ...])
                final, inter = enc_out
                if self.layer is None:
                    enc_out = final
                else:
                    inter_d = dict(inter)
                    assert self.layer in inter_d, f"layer {self.layer} not an inter-CTC layer {sorted(inter_d)}"
                    enc_out = inter_d[self.layer]  # raw (pre-after_norm) self-cond block output
            logits = self.model.ctc.ctc_lo(enc_out)  # [1, T_enc, V]
        return leaf, logits, int(leaf.shape[1])

    def forward(
        self,
        *,
        raw_inputs: Union[np.ndarray, torch.Tensor, List[List[str]]],
        raw_inputs_sample_rate: Optional[int] = None,
        raw_input_seq_lens: torch.Tensor,
        raw_targets: List[List[str]],
        raw_target_seq_lens: torch.Tensor,
        omitted_prev_context: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        assert raw_inputs_sample_rate == 16000, "OWSM-CTC expects 16 kHz"
        assert len(raw_inputs) == 1 and isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("OWSM-CTC chunked context not implemented")
        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])

        leaf, logits, t_feat = self._encode(raw_inputs[0].float(), orig_n_samples)

        # Subword (BPE) target ids, grouped to words via the '▁' word-start marker.
        flat_ids: List[int] = []
        word_start_end: List[List[int]] = []
        for word in words:
            pieces = self._tokenizer.text2tokens(word)
            ids = [self._tok2id[p] for p in pieces if p in self._tok2id]
            assert ids, f"word {word!r} -> no in-vocab BPE pieces"
            cstart = len(flat_ids)
            flat_ids.extend(ids)
            word_start_end.append([cstart, len(flat_ids)])
        s_len = len(flat_ids)
        assert s_len > 0, f"empty target for {words!r}"

        lp = torch.log_softmax(logits[0].float(), dim=-1)  # [T_enc, V]
        partial = self._ctc_partial_scores(lp, flat_ids)  # [S]
        partial_padded = torch.cat([partial, partial.new_zeros(1)])  # [S+1]

        targets = torch.tensor([flat_ids + [self.blank_idx]], dtype=torch.long, device=dev)
        word_start_end = word_start_end + [[s_len, s_len + 1]]
        input_slice = (torch.tensor([0], dtype=torch.int64), torch.tensor([t_feat], dtype=torch.int64))
        edges = torch.arange(t_feat + 1, dtype=torch.float64) * (orig_n_samples / max(t_feat, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(0)

        print(f"[fwd] words={len(words)} subwords={s_len} T_enc={t_feat} text={' '.join(words)!r}", flush=True)
        return ForwardOutput(
            inputs=leaf,
            input_seq_lens=torch.tensor([t_feat]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(word_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
            outputs=dict(partial_padded=partial_padded, vocab_size=self.vocab_size),
        )

    def log_probs(
        self,
        *,
        forward_output: ForwardOutput,
        start: Union[int, torch.Tensor],
        end: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        partial_padded = forward_output.outputs["partial_padded"]
        v = forward_output.outputs["vocab_size"]
        start_i = int(start[0]) if isinstance(start, torch.Tensor) else int(start)
        end_i = int(end[0]) if isinstance(end, torch.Tensor) else int(end)
        n = end_i - start_i
        device = partial_padded.device
        vals = partial_padded[start_i:end_i]
        targets_slice = forward_output.targets[0, start_i:end_i].to(device)
        out = partial_padded.new_zeros((1, n, v))
        out[0, torch.arange(n, device=device), targets_slice] = vals
        return out

    def recog(
        self,
        *,
        raw_inputs: torch.Tensor,
        raw_inputs_sample_rate: int,
        raw_input_seq_lens: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> List[List[str]]:
        """CTC greedy: argmax per frame, collapse repeats, drop blanks + special tokens, BPE-detok."""
        assert len(raw_inputs) == 1 and raw_inputs_sample_rate == 16000
        with torch.no_grad():
            _, logits, _ = self._encode(raw_inputs[0].float(), int(raw_input_seq_lens[0]))
            ids = logits[0].argmax(-1).tolist()
        out_ids: List[int] = []
        prev = None
        for i in ids:
            if i != prev and i != self.blank_idx:
                out_ids.append(i)
            prev = i
        pieces = [self.token_list[i] for i in out_ids]
        pieces = [p for p in pieces if not (p.startswith("<") and p.endswith(">"))]  # drop lang/task/specials
        text = self._tokenizer.tokens2text(pieces) if pieces else ""
        return [text.split()]

    def forced_align_words(self, *, audio: torch.Tensor, sample_rate: int, words: List[str]):
        """torchaudio CTC forced-alignment of ``words`` on this model's emission (the selected layer).

        Returns per-word (start, end) in seconds -- the 'posteriors' baseline vs grad-align. The
        emission forced-aligns well at every block even though grad-align does not.
        """
        import torchaudio

        assert sample_rate == 16000, "OWSM-CTC expects 16 kHz"
        dev = self.device
        dur = int(audio.shape[0]) / sample_rate
        with torch.no_grad():
            _, logits, _ = self._encode(audio.float(), int(audio.shape[0]))
            lp = torch.log_softmax(logits[0].float(), dim=-1)  # [T_enc, V]
        flat_ids, word_ranges = [], []
        for w in words:
            a = len(flat_ids)
            flat_ids.extend(self._tok2id[p] for p in self._tokenizer.text2tokens(w) if p in self._tok2id)
            word_ranges.append((a, len(flat_ids)))
        assert len(word_ranges) == len(words)
        n_frames = int(lp.shape[0])
        spf = dur / max(n_frames, 1)
        targets = torch.tensor([flat_ids], dtype=torch.int32, device=dev)
        aligned, scores = torchaudio.functional.forced_align(lp[None], targets, blank=self.blank_idx)
        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0], blank=self.blank_idx)
        assert len(spans) == len(flat_ids), f"{len(spans)} vs {len(flat_ids)}"
        pred_sub = [(s.start * spf, s.end * spf) for s in spans]
        return [(pred_sub[a][0], pred_sub[b - 1][1]) for a, b in word_ranges]
