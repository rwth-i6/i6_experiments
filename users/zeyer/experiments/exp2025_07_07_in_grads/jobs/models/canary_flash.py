"""NVIDIA Canary-1B-Flash adapter (NeMo ``EncDecMultiTaskModel``, AED).

A current top-10 Open ASR Leaderboard model:
FastConformer encoder,
Transformer decoder with learned absolute positional embeddings,
i.e. a hard decoder-length cap (the Whisper failure mode in a modern model);
the model card states audio inputs under 40 s, long-form only via external chunking.
The adapter asserts the decoder budget explicitly,
so the long-form run fails with the structural reason instead of an opaque index error.

Per-token score: teacher-forced next-token log-probs
from the model's single public forward (``transcript`` = prompt + text ids),
whisper-adapter style.
No gradient extraction, no attention collection (chunk-align scoring only).
"""

from __future__ import annotations

from typing import List, Optional, Union
import glob
import os
import sys
import time

import numpy as np
import torch

from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from .base import BaseModelInterface, ForwardOutput


class CanaryFlash(BaseModelInterface):
    """NVIDIA Canary-1B-Flash (FastConformer encoder + Transformer AED decoder)."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        overlay_path: str,
        source_lang: str = "en",
        target_lang: str = "en",
        pnc: str = "no",
        grad_wrt: Optional[str] = None,
        version: int = 1,
    ):
        """
        :param model_dir: hub cache dir for ``nvidia/canary-1b-flash``.
        :param overlay_path: NeMo env overlay to activate on sys.path (passed from the recipe).
        :param pnc: punctuation/capitalization slot; "no" matches the lowercased Buckeye reference.
        :param grad_wrt: only ``None`` -- gradient extraction is not implemented for this adapter.
        """
        super().__init__()
        assert grad_wrt is None, f"CanaryFlash: gradient extraction not implemented ({grad_wrt=})"
        assert version >= 1
        self.device = device
        self.model_dir = model_dir
        self.overlay_path = overlay_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.pnc = pnc
        self.version = version

        if overlay_path not in sys.path:
            sys.path.insert(0, overlay_path)

        print("Import NeMo / EncDecMultiTaskModel (from overlay)...")
        start_time = time.time()
        import nemo
        from nemo.collections.asr.models import EncDecMultiTaskModel
        from nemo.collections.common.prompts import PromptFormatter

        print(f"  nemo={nemo.__version__} from {nemo.__file__}")

        content = get_content_dir_from_hub_cache_dir(model_dir)
        nemo_files = glob.glob(os.path.join(content, "**", "*.nemo"), recursive=True)
        assert len(nemo_files) == 1, f"expected exactly one .nemo under {content}, got {nemo_files}"
        print(f"Restoring Canary from {nemo_files[0]}...")
        self.model = EncDecMultiTaskModel.restore_from(nemo_files[0], map_location=device)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.tokenizer = self.model.tokenizer

        prompt_format = self.model.prompt_format
        self.formatter = PromptFormatter.resolve(prompt_format)(self.tokenizer)
        # Slot values by name, resolved against whatever slots this prompt format declares;
        # an undeclared-slot value is a hard error (no silent guessing across canary versions).
        self._slot_defaults = {
            "source_lang": f"<|{source_lang}|>",
            "target_lang": f"<|{target_lang}|>",
            "task": "<|transcribe|>",
            "taskname": "<|transcribe|>",
            "pnc": {"no": "<|nopnc|>", "yes": "<|pnc|>"}[pnc],
            "itn": "<|noitn|>",
            "timestamp": "<|notimestamp|>",
            "diarize": "<|nodiarize|>",
            "emotion": "<|emo:undefined|>",
            "decodercontext": "",
            "context": "",
        }

        # The decoder's learned absolute positional embedding sets a hard token budget.
        dec_cfg = self.model.cfg.transf_decoder
        self.max_dec_len = int(dec_cfg.config_dict.max_sequence_length)
        self.target_sr = int(self.model.cfg.sample_rate)
        self.vocab_size = int(self.model.cfg.head.num_classes)
        print(
            f"  ({time.time() - start_time:.1f}s) prompt_format={prompt_format!r} "
            f"max_dec_len={self.max_dec_len} vocab={self.vocab_size} sr={self.target_sr}"
        )

    # ---- Helpers --------------------------------------------------------

    def _fill_slots(self, role: str, extra: Optional[dict] = None) -> dict:
        slots = {}
        for slot in self.formatter.get_slots(role):
            if extra and slot in extra:
                slots[slot] = extra[slot]
            else:
                assert slot in self._slot_defaults, f"no default for prompt slot {slot!r} (role {role!r})"
                slots[slot] = self._slot_defaults[slot]
        # aggregate-tokenizer meta slot: the prompt turn uses the special-token sub-tokenizer,
        # the assistant text turn the target-language sub-tokenizer
        slots["prompt_language"] = "spl_tokens" if role == "user" else self.target_lang
        return slots

    def _build_ids(self, transcription: str):
        """Prompt-only and full teacher-forced decoder ids via the canary prompt formatter."""
        user_turn = {"role": "user", "slots": self._fill_slots("user")}
        prefix = self.formatter.encode_dialog([user_turn])["input_ids"]
        full = self.formatter.encode_dialog(
            [user_turn, {"role": "assistant", "slots": self._fill_slots("assistant", {"text": transcription})}]
        )["input_ids"]
        assert torch.equal(full[: prefix.shape[0]], prefix), "prompt ids not a prefix of full ids"
        return full.unsqueeze(0), int(prefix.shape[0])

    # ---- Forward (forced alignment) -------------------------------------

    def forward(
        self,
        *,
        raw_inputs: Union[np.ndarray, torch.Tensor, List[List[str]]],
        raw_inputs_sample_rate: Optional[int] = None,
        raw_input_seq_lens: torch.Tensor,
        raw_targets: List[List[str]],
        raw_target_seq_lens: torch.Tensor,
        omitted_prev_context: Optional[torch.Tensor] = None,
        collect_attentions: Optional[list] = None,
    ) -> ForwardOutput:
        assert collect_attentions is None, "CanaryFlash: attention collection not implemented"
        assert len(raw_inputs) == 1, "CanaryFlash wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        # Omitted prev context: the decoder prompt has no free-text slot in the asr task,
        # so chunked runs score each chunk without prev-text conditioning (documented approximation).

        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])
        transcription = " ".join(words)
        print(f"[fwd] start; words={len(words)} text={transcription!r}", flush=True)

        wav = raw_inputs[0, :orig_n_samples].to(dev).float()
        if raw_inputs_sample_rate != self.target_sr:
            import torchaudio

            wav = torchaudio.functional.resample(wav[None], raw_inputs_sample_rate, self.target_sr)[0]

        input_ids, dst_text_start = self._build_ids(transcription)
        input_ids = input_ids.to(dev)
        n_total = int(input_ids.shape[1])
        # The structural long-form limit: learned absolute decoder positions.
        assert n_total <= self.max_dec_len, (
            f"decoder input of {n_total} tokens exceeds the learned-positional-embedding budget "
            f"({self.max_dec_len}): Canary-1B-Flash cannot take this transcript length (long-form input)"
        )

        with torch.no_grad():
            transf_log_probs, encoded_len, enc_states, enc_mask = self.model.forward(
                input_signal=wav[None],
                input_signal_length=torch.tensor([wav.shape[0]], device=dev),
                transcript=input_ids,
                transcript_length=torch.tensor([n_total], device=dev),
            )
        del enc_states, enc_mask
        # [1, n_total, V] log-probs; position i predicts token i+1.
        assert transf_log_probs.shape[:2] == input_ids.shape
        t_enc = int(encoded_len[0])

        # Targets: text tokens after the prompt, incl. the trailing EOS as the chunk-exit slot.
        targets = input_ids[:, dst_text_start:]
        n_targets = int(targets.shape[1]) - 1  # last = EOS
        self.assistant_end_token_id = int(targets[0, -1])
        assert n_targets > 0, f"empty target for words={words!r}"

        # Per-word grouping via the SentencePiece word-start marker.
        toks = self.tokenizer.ids_to_tokens([int(t) for t in targets[0, :n_targets]])
        words_start_end: List[List[int]] = []
        words_: List[str] = []
        for j, s in enumerate(toks):
            if j == 0 or s.startswith("▁"):
                words_start_end.append([j, j + 1])
                words_.append(s.lstrip("▁"))
            else:
                words_[-1] += s
                words_start_end[-1][1] = j + 1
        assert len(words_start_end) == len(words), (
            f"word-grouping mismatch: {len(words_start_end)} groups ({words_!r}) vs {len(words)} words ({words!r})"
        )
        words_start_end = words_start_end + [[n_targets, n_targets + 1]]  # exit slot

        # Encoder frame -> raw sample mapping (FastConformer 8x subsampled grid).
        input_slice = (torch.tensor([0], dtype=torch.int64), torch.tensor([t_enc], dtype=torch.int64))
        edges = torch.arange(t_enc + 1, dtype=torch.float64) * (orig_n_samples / max(t_enc, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(0)

        print(f"[fwd] ok; n_total={n_total} n_targets={n_targets} t_enc={t_enc}", flush=True)
        return ForwardOutput(
            inputs=transf_log_probs,
            input_seq_lens=torch.tensor([t_enc]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(words_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
            outputs=dict(log_probs=transf_log_probs, dst_text_start=dst_text_start),
        )

    # ---- log_probs -------------------------------------------------------

    def log_probs(
        self,
        *,
        forward_output: ForwardOutput,
        start: Union[int, torch.Tensor],
        end: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        lp = forward_output.outputs["log_probs"]
        dst_text_start = forward_output.outputs["dst_text_start"]
        # Position dst_text_start + u - 1 predicts target token u.
        return batch_slice(lp, (dst_text_start + start - 1, dst_text_start + end - 1)).float()

    # ---- Recog (open recognition, greedy) -------------------------------

    def recog(
        self,
        *,
        raw_inputs: torch.Tensor,
        raw_inputs_sample_rate: int,
        raw_input_seq_lens: torch.Tensor,
        max_new_tokens: int = 512,
    ) -> List[List[str]]:
        import soundfile as sf
        import tempfile

        assert len(raw_inputs) == 1
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            sf.write(path, raw_inputs[0].detach().cpu().numpy().astype(np.float32), raw_inputs_sample_rate)
            with torch.no_grad():
                hyp = self.model.transcribe([path], batch_size=1, verbose=False)[0]
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        return [text.split()]
