from typing import Optional, Union, Any, Sequence, List, Dict
import time
import numpy as np
import torch
from i6_experiments.users.zeyer.torch.report_dev_memory_stats import report_dev_memory_stats
from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from ..logits_transform import make_logits_transform
from .base import BaseModelInterface, ForwardOutput


class Phi4MM(BaseModelInterface):
    """
    Phi4MM model interface.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        speech_prompt: str = "Transcribe the audio clip into text.",
        grad_wrt: str = "speech_embeddings",
        logits_transform: Union[None, str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]] = None,
    ):
        """
        :param model_dir: hub cache dir of model e.g. via DownloadHuggingFaceRepoJob.out_hub_cache_dir
        :param speech_prompt: text-only part of the prompt
        """
        super().__init__()

        self.device = device
        self.model_dir = model_dir
        self.speech_prompt = speech_prompt
        self.grad_wrt = grad_wrt
        self.logits_transform = make_logits_transform(logits_transform)

        print("Import Transformers...")
        start_time = time.time()

        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"({time.time() - start_time} secs)")

        print("Loading model...")
        start_time = time.time()
        model_dir = get_content_dir_from_hub_cache_dir(self.model_dir)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype="auto", trust_remote_code=True, device_map=str(device)
        ).to(device)

        from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalForCausalLM

        model: Phi4MultimodalForCausalLM  # just as an example...
        print(model)
        print("model.dtype:", model.dtype)
        print(f"({time.time() - start_time} secs)")

        self.processor = processor
        self.model = model

        tokenizer = self.processor.tokenizer
        (self.assistant_token_id,) = tokenizer.convert_tokens_to_ids(["<|assistant|>"])
        (self.assistant_end_token_id,) = tokenizer.convert_tokens_to_ids(["<|end|>"])

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
        assert raw_inputs_sample_rate is not None  # assume audio input
        assert (len(raw_inputs),) == raw_input_seq_lens.shape == (len(raw_targets),) == raw_target_seq_lens.shape, (
            f"batch size inconsistent, got {len(raw_inputs)=}, {len(raw_input_seq_lens)=},"
            f" {len(raw_targets)=}, {len(raw_target_seq_lens)=}"
        )
        assert len(raw_inputs) == 1, f"Phi4MM currently supports only batch size 1, got {len(raw_inputs)=}"
        assert isinstance(raw_inputs, torch.Tensor), f"raw_inputs expected to be torch.Tensor, got {type(raw_inputs)=}"
        assert raw_inputs.ndim == 2, f"raw_inputs expected to be 2D, got {raw_inputs.ndim=}"
        assert raw_input_seq_lens[0] == raw_inputs.shape[1], (
            f"raw_input_seq_lens[0]={raw_input_seq_lens[0]} != {raw_inputs.shape[1]}"
        )

        speech_prompt = self.speech_prompt
        dev = self.device

        # TODO maybe monkey patch some modules, e.g. Phi4MMRMSNorm,
        #   via liger_kernel.transformers.monkey_patch._patch_rms_norm_module?

        tokenizer = self.processor.tokenizer

        words = raw_targets[0]
        transcription = " ".join(words)
        added_prefix = False
        if omitted_prev_context is not None and omitted_prev_context[0] > 0:
            added_prefix = True
            transcription = "... " + transcription
        prompt = f"<|user|><|audio_1|>{speech_prompt}<|end|><|assistant|>{transcription}<|end|>"
        inputs = self.processor(text=prompt, audios=[(raw_inputs[0], raw_inputs_sample_rate)], return_tensors="pt")
        input_ids = inputs["input_ids"]
        (dst_text_start,) = torch.nonzero(input_ids[0] == self.assistant_token_id).squeeze(dim=1)
        dst_text_start = int(dst_text_start) + 1  # one past the assistant token
        dst_text_end = input_ids.shape[-1] - 1  # right before the <end> token. excluding.
        inputs = inputs.to(dev)
        input_ids = inputs["input_ids"]
        inputs_embeds = inputs["input_audio_embeds"]
        print("inputs_embeds:", inputs_embeds)
        inputs_embeds.requires_grad = True
        inputs_embeds.retain_grad()
        # We don't need the logits here. There is currently no way to not compute them,
        # so num_logits_to_keep=1 is the best we can do.
        # We then will compute only the needed logits below,
        # and for that, we need the last layer output, thus output_hidden_states=True.
        res = self.model(**inputs, output_hidden_states=True, num_logits_to_keep=1)
        last_out = res.hidden_states[-1]  # [B,T,D]
        del res
        assert last_out.shape[:2] == input_ids.shape
        report_dev_memory_stats(dev)

        targets = input_ids[:, dst_text_start:dst_text_end]  # [B,T']

        words_start_end = [[0, 1]]
        tokens = [tokenizer.decode(targets[0, :1])]
        words_ = [tokens[-1]]
        for t in range(1, targets.shape[1]):
            s = tokenizer.decode(targets[0, t : t + 1])
            tokens.append(s)
            if s.startswith(" "):  # new word
                words_.append(s[1:])
                words_start_end[-1][1] = t
                words_start_end.append([t, t + 1])
            else:
                words_[-1] += s
                words_start_end[-1][1] = t + 1
        if added_prefix:
            assert words_[0] == "..."
            words_start_end = words_start_end[1:]
            words_ = words_[1:]
        assert len(words_start_end) == len(words_) == len(words), f"got {tokens=}"
        assert words_ == words, f"{tokens=} {words=} {words_=}"

        assert self.grad_wrt == "speech_embeddings", f"{self.grad_wrt=!r}"
        return ForwardOutput(
            inputs=inputs_embeds,
            input_seq_lens=torch.tensor([inputs_embeds.shape[1]]),  # [B]
            input_slice_start_end=None,
            input_raw_start_end=None,  # TODO...
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),  # [B]
            target_start_end=torch.tensor(words_start_end, dtype=torch.int64).unsqueeze(0),  # [B, T, 2]
            outputs=dict(dst_text_start=dst_text_start, last_out=last_out),
        )

    def log_probs(
        self, *, forward_output: ForwardOutput, start: Union[int, torch.Tensor], end: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        last_out = forward_output.outputs["last_out"]
        dst_text_start = forward_output.outputs["dst_text_start"]
        last_out = batch_slice(last_out, (dst_text_start + start - 1, dst_text_start + end - 1))

        logits = self.model.lm_head(last_out)  # [B, T', V]
        logits = logits.float()
        for f in self.logits_transform:
            logits = f(logits)

        return logits.log_softmax(-1)  # [B, T', V]
