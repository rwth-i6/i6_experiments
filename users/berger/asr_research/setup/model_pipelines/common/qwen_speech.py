"""Audio-conditioned Qwen3-ASR baseline and dynamic Granite shallow fusion.

The auxiliary score is -log P(text | audio), not a text-only LM probability.
Audio embeddings are computed once per segment. GPU scoring recomputes the
decoder history in small batches, without assembling a KV cache.
"""

import json
import time

import numpy as np
import torch
from sisyphus import Job, Task

from .dynamic_llm import QwenContinuationScorer


class QwenSpeechScorer(QwenContinuationScorer):
    def __init__(self, repo, *, device="cuda", batch_size=2):
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.device, self.batch_size = device, batch_size
        self.formatting = "original"
        self.processor = AutoProcessor.from_pretrained(repo)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(repo, dtype=torch.float32).to(device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos)
        # Native generation accepts document EOS and chat-turn EOS. Fusion
        # scores the assistant-turn terminator, not both tokens sequentially.
        turn_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if turn_end not in self.eos_ids:
            raise ValueError("Speech-LLM chat-turn EOS is missing from generation_config")
        self.end, self.initial = [turn_end], []
        self.reset()

    def tokenize(self, texts):
        # Preserve the surface supplied by search. Granite supplies its AM
        # spelling; RNN-T supplies separately enumerated casing variants.
        from .dynamic_qwen_tuned import TunedQwenScorer

        return TunedQwenScorer.tokenize(self, texts)

    def prepare(self, waveform):
        return self.processor.apply_transcription_request(
            audio=waveform, language="English", return_tensors="pt", sampling_rate=16000
        ).to(self.device)

    @torch.inference_mode()
    def set_audio(self, inputs):
        if inputs["input_ids"].shape[0] != 1 or not inputs["attention_mask"].all():
            raise ValueError("Speech fusion requires one unpadded audio prompt")
        self.inputs = inputs
        self.initial = inputs["input_ids"][0].tolist()
        embeddings = self.model.get_input_embeddings()(inputs["input_ids"])
        audio = self.model.get_audio_features(
            inputs["input_features"], inputs["input_features_mask"], return_dict=True
        ).pooler_output
        mask = self.model.model.get_placeholder_mask(inputs["input_ids"], embeddings, audio)
        self.prompt_embeddings = embeddings.masked_scatter(mask, audio.to(embeddings.dtype))
        self.reset()

    @torch.inference_mode()
    def native(self, inputs, max_new_tokens=500):
        raw = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        if hasattr(raw, "sequences"):
            raw = raw.sequences
        ids = raw[0, inputs["input_ids"].shape[1] :].tolist()
        return dict(
            text=self.processor.decode(ids, return_format="transcription_only"),
            raw_token_ids=ids,
            truncated=bool(ids and ids[-1] not in self.eos_ids),
        )

    @torch.inference_mode()
    def score_continuations(self, prefixes, continuations):
        if len(prefixes) != len(continuations):
            raise ValueError("Mismatched Speech-LLM requests")
        prompt_len = len(self.initial)
        if not prompt_len or any(p[:prompt_len] != self.initial for p in prefixes):
            raise ValueError("History does not start with the current audio prompt")
        result = [[] for _ in prefixes]
        order = sorted(
            (i for i, c in enumerate(continuations) if c), key=lambda i: len(prefixes[i]) + len(continuations[i])
        )
        self.requests += len(order)
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            length = max(len(prefixes[i]) + len(continuations[i]) for i in batch)
            if length > self.model.config.text_config.max_position_embeddings:
                raise ValueError("Speech-LLM context limit exceeded")
            self.forward_batches += 1
            self.max_request_length = max(self.max_request_length, length)
            ids = torch.full((len(batch), length), self.end[0], dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for b, i in enumerate(batch):
                seq = prefixes[i] + continuations[i]
                ids[b, : len(seq)] = torch.tensor(seq, device=self.device)
                mask[b, : len(seq)] = 1
            embeddings = self.model.get_input_embeddings()(ids)
            embeddings[:, :prompt_len] = self.prompt_embeddings.expand(len(batch), -1, -1)
            logits = self.model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).logits
            for b, i in enumerate(batch):
                p, c = prefixes[i], continuations[i]
                costs = torch.nn.functional.cross_entropy(
                    logits[b, len(p) - 1 : len(p) + len(c) - 1].float(),
                    torch.tensor(c, device=self.device),
                    reduction="none",
                )
                if not torch.isfinite(costs).all():
                    raise ValueError("Non-finite Speech-LLM costs")
                result[i] = costs.cpu().tolist()
        return result

    @torch.inference_mode()
    def validate(self):
        """Check cached audio and batched continuation scores against full forwarding."""
        tokens = self.tokenizer.encode("Hello world", add_special_tokens=False) + self.end
        prefixes = [self.initial, self.initial + tokens[:1]]
        continuations = [tokens, tokens[1:]]
        actual = self.score_continuations(prefixes, continuations)
        ids = torch.tensor([self.initial + tokens], device=self.device)
        inputs = dict(self.inputs, input_ids=ids, attention_mask=torch.ones_like(ids))
        logits = self.model(**inputs, use_cache=False).logits[0, len(self.initial) - 1 : -1]
        expected = torch.nn.functional.cross_entropy(
            logits.float(), torch.tensor(tokens, device=self.device), reduction="none"
        )
        for row, offset in zip(actual, (0, 1)):
            np.testing.assert_allclose(row, expected[offset:].cpu().numpy(), atol=2e-3, rtol=2e-4)
        return dict(passed=True, checked_tokens=len(tokens), audio_conditioned=True)


def register_speech_scorer(librasr, backend):
    class Scorer(librasr.LlmScorer):
        def __init__(self, config):
            super().__init__(config)

        def reset(self):
            backend.reset()  # Retain current audio; reset only request statistics.

        def initial_tokens(self):
            return backend.initial_tokens()

        def sentence_end_tokens(self):
            return backend.sentence_end_tokens()

        def tokenize(self, texts):
            return backend.tokenize(texts)

        def score_continuations(self, prefixes, continuations):
            return backend.score_continuations(prefixes, continuations)

    librasr.register_llm_scorer_type("qwen-speech", Scorer)


class QwenSpeechRecogJob(Job):
    # Preserve existing standalone/default-batch job identities.
    __sis_hash_exclude__ = {"batch_size": 2}

    def __init__(
        self,
        dataset,
        speech_repo,
        *,
        code_version,
        granite_repo=None,
        binary=None,
        vocabulary=None,
        rasr_config=None,
        baseline=None,
        max_new_tokens=500,
        batch_size=2,
    ):
        self.dataset, self.speech_repo, self.code_version = dataset, speech_repo, code_version
        self.granite_repo, self.binary, self.vocabulary = granite_repo, binary, vocabulary
        self.rasr_config, self.baseline, self.max_new_tokens = rasr_config, baseline, max_new_tokens
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if granite_repo is not None and any(x is None for x in (binary, vocabulary, rasr_config, baseline)):
            raise ValueError("Fusion requires a standalone baseline, binary, vocabulary and config")
        self.out_hypotheses = self.output_path("hypotheses.jsonl")
        self.out_runtime = self.output_path("runtime.json")
        self.out_validation = self.output_path("validation.json")
        self.out_log = self.output_path("rasr.log")
        self.rqmt = dict(cpu=2, mem=64, gpu=1, gpu_mem=24, time=48)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sys
        import transformers
        from datasets import load_from_disk
        from .hf_benchmark import HfBenchmarkAdapter, audio_array, traceback_result

        if transformers.__version__ != "5.16.1":
            raise RuntimeError("Use benchmark transformers==5.16.1")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        backend = QwenSpeechScorer(self.speech_repo.get_path(), batch_size=self.batch_size)
        search, adapter, vocabulary = None, None, {}
        if self.granite_repo is not None:
            with open(self.baseline.get_path()) as src:
                if not json.load(src)["ref_len"]:
                    raise ValueError("Standalone Speech-LLM scoring must complete first")
            sys.path.insert(0, self.binary.get_path())
            import librasr

            register_speech_scorer(librasr, backend)
            cfg = librasr.Configuration()
            if not cfg.set_from_file(self.rasr_config.get_path()):
                raise ValueError("Failed to read fusion config")
            cfg.set("lib-rasr.search-algorithm.log-stepwise-statistics", "false")
            for channel in ("log", "warning", "error", "statistics"):
                cfg.set(f"*.{channel}.channel", self.out_log.get_path())
            librasr.init_logging(cfg)
            search = librasr.SearchAlgorithm(config=cfg)
            adapter = HfBenchmarkAdapter(self.granite_repo.get_path(), "ctc", "float32")
            with open(self.vocabulary.get_path()) as src:
                vocabulary = {p.rstrip("\n"): i for i, p in enumerate(src)}
        dataset = load_from_disk(self.dataset.get_path())
        elapsed = audio_seconds = 0
        validation = dict(passed=True, native=True)
        with open(self.out_hypotheses.get_path(), "w") as dst:
            for i, sample in enumerate(dataset):
                waveform = audio_array(sample["audio"])
                torch.cuda.synchronize()
                started = time.perf_counter()
                inputs = backend.prepare(waveform)
                if search is None:
                    result = backend.native(inputs, self.max_new_tokens)
                else:
                    backend.set_audio(inputs)
                    if i == 0:
                        validation = backend.validate()
                    with torch.inference_mode():
                        features = adapter.encode_inputs(adapter.inputs(waveform))
                    trace = search.recognize_segment(np.ascontiguousarray(features, dtype=np.float32))
                    result = traceback_result(trace, adapter, vocabulary, True)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                elapsed += seconds
                audio_seconds += len(waveform) / 16000
                result.update(id=str(sample["id"]), elapsed_seconds=seconds)
                dst.write(json.dumps(result, ensure_ascii=False) + "\n")
                dst.flush()
                print(f"{i + 1}/{len(dataset)}: RTF={elapsed / audio_seconds:.3f}", flush=True)
        with open(self.out_validation.get_path(), "w") as dst:
            json.dump(validation, dst)
        with open(self.out_runtime.get_path(), "w") as dst:
            json.dump(
                dict(
                    audio_seconds=audio_seconds,
                    elapsed_seconds=elapsed,
                    rtf=elapsed / audio_seconds,
                    num_segments=len(dataset),
                    dtype="float32",
                    audio_conditioned=True,
                ),
                dst,
            )
