"""OWLS (ESPnet Whisper-style AED) adapter for the grad-based forced-alignment pipeline.

Model: the ESPnet OWLS suite (``espnet/owls_<size>_180K``) -- a CONTROLLED
Whisper-style encoder-decoder scaling ladder (0.25B..18B, same recipe/data),
purpose-built for scaling-law studies. Also loads OWSM models (same S2T task).

Per-token score is the direct autoregressive ``log p(y_i | y_<i, audio)``
(teacher-forced), same as the Whisper / LLM adapters. Grad target = the log-mel
frontend output (80-mel, 100 Hz), the OWLS analog of Whisper's log-mel leaf.
Subword (BPE) scoring, grouped to words via the ``▁`` word-start marker. Batch 1.

ESPnet env plumbing (see reference_owls_espnet_env): the core espnet2 is in the
job env; this module adds a local deps dir to sys.path and stubs numba (numpy 2.4
vs numba conflict) + a few TTS-only deps that S2T imports but never runs, all
BEFORE importing espnet.
"""

from __future__ import annotations

import os
import sys
import glob
import types
import importlib.machinery
import time
from typing import Optional, Union, List
from unittest.mock import MagicMock

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

    _nb.jit = _noop_jit
    _nb.njit = _noop_jit
    _nb.prange = range
    _nb.generated_jit = _noop_jit
    sys.modules["numba"] = _nb


def _stub_mod(name: str):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []
    m.__getattr__ = lambda attr: MagicMock()
    sys.modules[name] = m
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(_stub_mod(parent), child, m)
    return m


for _m in [
    "g2p_en", "pyopenjtalk", "pypinyin", "jamo", "phonemizer", "g2pk2", "g2pk", "jaconv",
    "tacotron_cleaner", "tacotron_cleaner.cleaners",
]:
    _stub_mod(_m)


class Owls(BaseModelInterface):
    """ESPnet OWLS/OWSM S2T (Whisper-style AED). See module docstring."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        language: str = "eng",
        char_level: bool = False,
        char_level_sep: Optional[str] = "▁",
        version: int = 1,
    ):
        super().__init__()
        assert version >= 1
        self.device = device
        self.model_dir = model_dir
        self.language = language
        self._char_level = char_level
        self._char_level_sep = char_level_sep
        self.version = version

        print("Import / load OWLS (ESPnet S2T)...")
        start_time = time.time()
        from espnet2.tasks.s2t import S2TTask

        snap = get_content_dir_from_hub_cache_dir(model_dir)
        # exp dir is exp_owsm OR exp_<tag> (data-scale variants); config and the ave .pth may live
        # in DIFFERENT s2t_train_* subdirs -> glob them independently.
        cfgs = glob.glob(os.path.join(snap, "exp_*", "s2t_train_*", "config.yaml"))
        assert len(cfgs) == 1, f"expected 1 config.yaml under {snap}/exp_*, got {cfgs}"
        cfg = cfgs[0]
        ckpts = glob.glob(os.path.join(snap, "exp_*", "s2t_train_*", "*.pth"))
        ckpts = [c for c in ckpts if "ave" in os.path.basename(c)] or ckpts
        assert ckpts, f"no .pth checkpoint under {snap}/exp_*"
        ckpt = sorted(ckpts, key=len)[0]  # valid.total_count.ave_5best.pth (the inference ckpt)

        # config.yaml references stats/token_list by RELATIVE path -> build from the snapshot root.
        _prev_cwd = os.getcwd()
        os.chdir(snap)
        try:
            self.model, train_args = S2TTask.build_model_from_file(cfg, ckpt, str(device))
        finally:
            os.chdir(_prev_cwd)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        token_list = list(train_args.token_list)
        self._tok2id = {t: i for i, t in enumerate(token_list)}
        # BPE tokenizer (ESPnet) -> word-start marker '▁'.
        from espnet2.text.build_tokenizer import build_tokenizer

        bpemodels = glob.glob(os.path.join(snap, "data", "token_list", "*", "bpe.model"))
        assert bpemodels, f"no bpe.model under {snap}/data/token_list"
        self._tokenizer = build_tokenizer(token_type="bpe", bpemodel=bpemodels[0])

        def _sid(tok):
            assert tok in self._tok2id, f"special token {tok!r} not in OWLS vocab"
            return self._tok2id[tok]

        # OWSM ASR prompt: <sos> <lang> <asr> <notimestamps> <text...>.
        self.prefix_ids = [_sid("<sos>"), _sid(f"<{language}>"), _sid("<asr>"), _sid("<notimestamps>")]
        self.eos_id = _sid("<eos>")
        self.vocab_size = len(token_list)
        print(f"  ({time.time() - start_time:.1f}s) prefix={self.prefix_ids} eos={self.eos_id} vocab={self.vocab_size}")

    def _encode_leaf(self, wav: torch.Tensor, n_samples: int):
        """Run the frontend, leaf-ify the log-mel, then normalize+encode. Returns
        (leaf [1,T,80], enc_out [1,T_enc,D], enc_lens [1], T)."""
        speech = wav.to(self.device)[None]  # [1, T_samples]
        slens = torch.tensor([n_samples], device=self.device)
        with torch.no_grad():
            feats, flens = self.model._extract_feats(speech, slens)  # [1, T, 80] log-mel @100Hz
        leaf = feats.detach().requires_grad_(True)
        leaf.retain_grad()
        with torch.enable_grad():
            nfeats, nflens = self.model.normalize(leaf, flens) if getattr(self.model, "normalize", None) else (leaf, flens)
            enc = self.model.encoder(nfeats, nflens)
        return leaf, enc[0], enc[1], int(leaf.shape[1])

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
        assert raw_inputs_sample_rate == 16000, "OWLS expects 16 kHz"
        assert len(raw_inputs) == 1 and isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("OWLS chunked context not implemented")

        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])

        leaf, enc_out, enc_lens, t_feat = self._encode_leaf(raw_inputs[0].float(), orig_n_samples)

        transc_ids: List[int] = []
        words_start_end: List[List[int]] = []
        if self._char_level:
            # Per-char scoring (like the Whisper char adapter): map each char to its BARE
            # vocab piece directly (every ascii letter + apostrophe is a single OWLS piece;
            # sp.encode would otherwise prepend the word-marker). Optional '▁' separator before
            # each word as autoregressive context (never scored).
            for word in words:
                if self._char_level_sep:
                    transc_ids.append(self._tok2id[self._char_level_sep])
                cstart = len(transc_ids)
                for ch in word:
                    assert ch in self._tok2id, f"char {ch!r} (word {word!r}) not an OWLS vocab piece"
                    transc_ids.append(self._tok2id[ch])
                words_start_end.append([cstart, len(transc_ids)])
        else:
            # Subword scoring, grouped to words via the '▁' word-start marker.
            for word in words:
                pieces = self._tokenizer.text2tokens(word)
                ids = [self._tok2id[p] for p in pieces if p in self._tok2id]
                assert ids, f"word {word!r} -> no in-vocab BPE pieces"
                cstart = len(transc_ids)
                transc_ids.extend(ids)
                words_start_end.append([cstart, len(transc_ids)])
        n_targets = len(transc_ids)
        assert n_targets > 0, f"empty target for {words!r}"
        assert len(words_start_end) == len(words)

        ys_in = torch.tensor([self.prefix_ids + transc_ids], dtype=torch.long, device=dev)
        ys_lens = torch.tensor([ys_in.shape[1]], device=dev)
        dst_text_start = len(self.prefix_ids)
        with torch.enable_grad():
            dec = self.model.decoder(enc_out, enc_lens, ys_in, ys_lens)
            logits = dec[0]  # [1, L, V] (output_layer applied)

        targets = torch.tensor([transc_ids + [self.eos_id]], dtype=torch.long, device=dev)
        words_start_end = words_start_end + [[n_targets, n_targets + 1]]  # exit slot

        input_slice = (torch.tensor([0], dtype=torch.int64), torch.tensor([t_feat], dtype=torch.int64))
        edges = torch.arange(t_feat + 1, dtype=torch.float64) * (orig_n_samples / max(t_feat, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(0)

        print(f"[fwd] words={len(words)} subwords={n_targets} t_feat={t_feat} text={' '.join(words)!r}", flush=True)
        return ForwardOutput(
            inputs=leaf,
            input_seq_lens=torch.tensor([t_feat]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(words_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
            outputs=dict(logits=logits, dst_text_start=dst_text_start),
        )

    def log_probs(
        self,
        *,
        forward_output: ForwardOutput,
        start: Union[int, torch.Tensor],
        end: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice

        logits = forward_output.outputs["logits"]
        dst_text_start = forward_output.outputs["dst_text_start"]
        # Decoder position P+i-1 predicts target token i; slice [start-1, end-1] (offset by the prompt).
        sl = batch_slice(logits, (dst_text_start + start - 1, dst_text_start + end - 1))
        return sl.float().log_softmax(-1)
