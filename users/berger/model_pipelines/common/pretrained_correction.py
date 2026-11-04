"""Opt-in pretrained-AM correction; existing benchmark runtimes stay unchanged."""

import json
from collections import OrderedDict
from contextlib import contextmanager
from itertools import islice

import numpy as np
import torch
from sisyphus import Job, Task

from . import hf_benchmark
from .dynamic_qwen_tuned import TunedDynamicQwenJob
from .rnnt_llm import RnntQwenRecogJob
from .qwen_speech import QwenSpeechRecogJob, QwenSpeechScorer

BaseAdapter = hf_benchmark.HfBenchmarkAdapter


class EstimateGranitePriorJob(Job):
    """Frame-weighted posterior mean over a bounded training-only HF stream."""
    def __init__(self, repo, training_source, *, code_version, max_segments=512):
        self.repo, self.training_source = repo, training_source
        self.code_version, self.max_segments = code_version, max_segments
        if not training_source["split"].startswith("train"):
            raise ValueError("Prior estimation must use training data")
        self.out_prior = self.output_path("prior.json")
        self.rqmt = dict(cpu=2, mem=32, gpu=1, gpu_mem=24, time=8)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from datasets import Audio, load_dataset
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        source = self.training_source
        dataset = load_dataset(source["repo"], source["config"], split=source["split"],
                               revision=source["revision"], streaming=True)
        dataset = dataset.cast_column(source["audio_key"], Audio(sampling_rate=16000))
        adapter = BaseAdapter(self.repo.get_path(), "ctc", "float32")
        total, frames, seconds, ids = None, 0, 0.0, []
        with torch.inference_mode():
            for index, sample in enumerate(islice(dataset, self.max_segments)):
                audio = hf_benchmark.audio_array(sample[source["audio_key"]])
                costs = adapter.encode_inputs(adapter.inputs(audio))
                mass = np.exp(-costs.astype(np.float64)).sum(axis=0)
                total = mass if total is None else total + mass
                frames += costs.shape[0]
                seconds += len(audio) / 16000
                ids.append(str(sample.get("id", sample.get("ID", index))))
                print(f"Prior {index + 1}/{self.max_segments}: frames={frames}", flush=True)
        if len(ids) != self.max_segments or frames == 0:
            raise ValueError("Insufficient training data for reproducible prior sample")
        probs = np.maximum(total / frames, 1e-12)
        probs /= probs.sum()
        with open(self.out_prior.get_path(), "w") as dst:
            json.dump(dict(log_prior=np.log(probs).tolist(), blank=adapter.blank,
                           pieces=[adapter.tokenizer.convert_ids_to_tokens(i) for i in range(len(probs))],
                           source=source, sample_ids=ids, frames=frames, audio_seconds=seconds,
                           model=self.repo.get_path(), frame_weighted=True), dst)


class CorrectedAdapter(BaseAdapter):
    def __init__(self, *args, prior=None, prior_scale=0.0, ilm_scale=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_scale, self.ilm_scale = prior_scale, ilm_scale
        self.validation = dict(prior_scale=prior_scale, ilm_scale=ilm_scale)
        self.ilm_cache = OrderedDict()
        if prior_scale < 0 or ilm_scale < 0 or not np.isfinite([prior_scale, ilm_scale]).all():
            raise ValueError("Invalid correction scales")
        if prior_scale:
            if self.family != "ctc" or prior is None:
                raise ValueError("CTC correction requires an estimated prior")
            with open(prior.get_path()) as src:
                data = json.load(src)
            self.log_prior = np.asarray(data["log_prior"], dtype=np.float32)
            pieces = [self.tokenizer.convert_ids_to_tokens(i) for i in range(len(self.log_prior))]
            if data["blank"] != self.blank or data["pieces"] != pieces:
                raise ValueError("Prior/model acoustic vocabulary indices differ")
            if not np.isfinite(self.log_prior).all() or not np.isclose(np.exp(self.log_prior).sum(), 1, atol=1e-5):
                raise ValueError("Invalid prior distribution")
            self.validation["prior_inventory_checked"] = True
        if ilm_scale and self.family != "rnnt":
            raise ValueError("ILM subtraction is only supported for RNN-T")

    def encode_inputs(self, inputs):
        self.ilm_cache.clear()
        features = super().encode_inputs(inputs)
        if self.prior_scale:
            if features.shape[-1] != len(self.log_prior):
                raise ValueError("Prior dimension differs from AM output")
            features = features + self.prior_scale * self.log_prior
            self.validation["prior_shape_checked"] = True
        return features

    def ilm_log_probs(self, prefixes):
        """Zero encoder, full predictor history; conditional nonblank distribution."""
        pending = list(dict.fromkeys(p for p in prefixes if p not in self.ilm_cache))
        buckets = {}
        for prefix in pending:
            buckets.setdefault(len(prefix), []).append(prefix)
        for group in buckets.values():
            for start in range(0, len(group), 8):
                batch = group[start:start + 8]
                ids = torch.tensor([[self.blank] + list(p) for p in batch], device=self.device)
                predictor = self.model.decoder(ids, cache=None)[:, -1, :]
                zero = torch.zeros((len(batch), self.encoder.shape[-1]), device=predictor.device,
                                   dtype=self.encoder.dtype)
                logits = self.model.joint(decoder_hidden_states=predictor, encoder_hidden_states=zero).float()
                nonblank = torch.arange(logits.shape[-1], device=logits.device) != self.blank
                log_probs = torch.zeros_like(logits)
                log_probs[:, nonblank] = logits[:, nonblank].log_softmax(-1)
                if not torch.isfinite(log_probs).all() or not torch.allclose(
                    log_probs[:, nonblank].exp().sum(-1), torch.ones(len(batch), device=logits.device), atol=1e-5
                ):
                    raise ValueError("Invalid zero-encoder conditional ILM probabilities")
                for prefix, value in zip(batch, log_probs.cpu().numpy()):
                    self.ilm_cache[prefix] = value
        # Read this batch before bounding the CPU-only cache.
        result = [self.ilm_cache[p] for p in prefixes]
        while len(self.ilm_cache) > 2048:
            self.ilm_cache.popitem(last=False)
        return result

    @torch.inference_mode()
    def score_contexts(self, contexts):
        scores = super().score_contexts(contexts)
        if not self.ilm_scale:
            return scores
        active = [i for i, answer in enumerate(scores) if answer is not None]
        ilms = self.ilm_log_probs([contexts[i][1] for i in active])
        for i, log_probs in zip(active, ilms):
            costs, frame = scores[i]
            corrected = np.asarray(costs) + self.ilm_scale * log_probs
            if corrected[self.blank] != costs[self.blank] or not np.isfinite(corrected).all():
                raise ValueError("ILM changed blank cost or produced nonfinite scores")
            scores[i] = (corrected.tolist(), frame)
        self.validation["ilm_nonblank_normalized"] = True
        self.validation["blank_unchanged"] = True
        return scores


@contextmanager
def corrected_adapter(prior=None, prior_scale=0.0, ilm_scale=0.0):
    """Scoped adapter injection in an isolated worker; no shared runtime edits."""
    original, instances = hf_benchmark.HfBenchmarkAdapter, []
    def factory(*args, **kwargs):
        adapter = CorrectedAdapter(*args, prior=prior, prior_scale=prior_scale, ilm_scale=ilm_scale, **kwargs)
        instances.append(adapter)
        return adapter
    hf_benchmark.HfBenchmarkAdapter = factory
    try:
        yield instances
    finally:
        hf_benchmark.HfBenchmarkAdapter = original


def _save_validation(path, adapters):
    if len(adapters) != 1:
        raise ValueError("Expected exactly one corrected acoustic adapter")
    data = adapters[0].validation
    if data.get("prior_scale") and not data.get("prior_shape_checked"):
        raise ValueError("CTC prior validation never ran")
    if data.get("ilm_scale") and not data.get("blank_unchanged"):
        raise ValueError("ILM validation never ran")
    with open(path.get_path(), "w") as dst:
        json.dump(dict(passed=True, **data), dst)


class CorrectedGraniteQwenJob(TunedDynamicQwenJob):
    def __init__(self, *, prior, prior_scale, **kwargs):
        self.prior, self.prior_scale = prior, prior_scale
        super().__init__(**kwargs)
        self.out_correction_validation = self.output_path("correction_validation.json")

    def run(self):
        with corrected_adapter(self.prior, self.prior_scale) as adapters:
            if self.zero:
                from .dynamic_llm_jobs import DynamicGraniteQwenJob
                DynamicGraniteQwenJob.run(self)
            else:
                super().run()
        _save_validation(self.out_correction_validation, adapters)


class CorrectedRnntQwenJob(RnntQwenRecogJob):
    def __init__(self, *, ilm_scale, **kwargs):
        self.ilm_scale = ilm_scale
        super().__init__(**kwargs)
        self.out_correction_validation = self.output_path("correction_validation.json")

    def run(self):
        with corrected_adapter(ilm_scale=self.ilm_scale) as adapters:
            super().run()
        _save_validation(self.out_correction_validation, adapters)


class CorrectedGraniteSpeechJob(QwenSpeechRecogJob):
    def __init__(self, *, prior, prior_scale, **kwargs):
        self.prior, self.prior_scale = prior, prior_scale
        super().__init__(**kwargs)
        self.out_correction_validation = self.output_path("correction_validation.json")

    def run(self):
        with corrected_adapter(self.prior, self.prior_scale) as adapters:
            super().run()
        _save_validation(self.out_correction_validation, adapters)


@torch.inference_mode()
def speech_beam_native(scorer, inputs, *, beam, length_penalty=1.0, max_new_tokens=500):
    raw = scorer.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=beam,
                                do_sample=False, length_penalty=length_penalty)
    raw = raw.sequences if hasattr(raw, "sequences") else raw
    ids = raw[0, inputs["input_ids"].shape[1]:].tolist()
    return dict(text=scorer.processor.decode(ids, return_format="transcription_only"),
                raw_token_ids=ids, truncated=bool(ids and ids[-1] not in scorer.eos_ids),
                beam=beam, length_penalty=length_penalty)


class SpeechQwenBeamJob(QwenSpeechRecogJob):
    def __init__(self, *, beam, length_penalty=1.0, **kwargs):
        self.beam, self.length_penalty = beam, length_penalty
        if beam not in (4, 8) or kwargs.get("granite_repo") is not None:
            raise ValueError("Standalone Speech-Qwen beam baseline only")
        super().__init__(**kwargs)

    def run(self):
        from . import qwen_speech
        beam, length_penalty = self.beam, self.length_penalty
        class BeamScorer(QwenSpeechScorer):
            @torch.inference_mode()
            def native(self, inputs, max_new_tokens=500):
                return speech_beam_native(self, inputs, beam=beam, length_penalty=length_penalty,
                                          max_new_tokens=max_new_tokens)
        original = qwen_speech.QwenSpeechScorer
        qwen_speech.QwenSpeechScorer = BeamScorer
        try:
            super().run()
        finally:
            qwen_speech.QwenSpeechScorer = original
