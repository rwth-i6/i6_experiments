"""Native HF and RASR adapters for the public benchmark.

Python label scorers are intentional first-integration implementations: no ONNX
conversion or cache-layout changes are mixed into the native/RASR comparison.
Full-history GPU forwards use bounded batches, and no scores survive a segment.
"""

import json
import time
from collections import defaultdict

import numpy as np
import torch
from sisyphus import Job, Task


def audio_array(audio):
    """Handle both datasets' dict interface and torchcodec AudioDecoder."""
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        array, rate = samples.data, samples.sample_rate
        array = array.cpu().numpy()
        if array.ndim == 2:
            array = array.mean(axis=0)
    else:
        array, rate = np.asarray(audio["array"]), audio["sampling_rate"]
    if rate != 16000 or array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"Invalid benchmark waveform: rate={rate}, shape={array.shape}")
    return np.asarray(array, dtype=np.float32)


def collapse_ctc(ids, blank):
    return [x for i, x in enumerate(ids) if x != blank and (i == 0 or x != ids[i - 1])]


class HfBenchmarkAdapter:
    def __init__(self, repo, family, dtype="float32", device="cuda", max_new_tokens=500):
        from transformers import AutoModelForCTC, AutoProcessor, CohereAsrForConditionalGeneration, ParakeetForRNNT

        cls = {"ctc": AutoModelForCTC, "rnnt": ParakeetForRNNT, "aed": CohereAsrForConditionalGeneration}[family]
        self.family, self.device, self.max_new_tokens = family, device, max_new_tokens
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = cls.from_pretrained(repo, dtype=getattr(torch, dtype)).to(device).eval()
        self.tokenizer = self.processor.tokenizer
        self.blank = getattr(self.model.config, "blank_token_id", 0) if family != "aed" else None
        self.eos = self.model.generation_config.eos_token_id if family == "aed" else None
        if isinstance(self.eos, list):
            if len(self.eos) != 1:
                raise ValueError("RASR adapter requires a single EOS")
            self.eos = self.eos[0]

    def inputs(self, waveform):
        """Single-chunk interface for CTC/RNNT and standalone encoder callers."""
        chunks = list(self.input_chunks(waveform))
        if len(chunks) != 1:
            raise ValueError("Multiple processor chunks: use native() or rasr() to decode and reassemble all chunks")
        return chunks[0][1]

    def input_chunks(self, waveform):
        """Keep HF's chunk boundaries/features; forward one chunk at a time.

        Do not split the waveform ourselves or preprocess chunks a second time:
        Cohere chooses energy-based boundaries and supplies audio_chunk_index.
        Padding and attention masks stay exactly as produced by the processor.
        """
        kwargs = dict(sampling_rate=16000, return_tensors="pt")
        if self.family == "aed":
            kwargs.update(language="en", punctuation=False)  # official Cohere benchmark prompt
        inputs = self.processor([waveform], **kwargs)
        num_chunks = inputs["input_features"].shape[0]
        indices = inputs.get("audio_chunk_index")
        if indices is None and num_chunks == 1:
            indices = [(0, None)]
        if indices is None or len(indices) != num_chunks or num_chunks == 0:
            raise ValueError("Processor chunk count does not match audio_chunk_index")
        indices = [tuple(index) for index in indices]
        if indices != [(0, None)] and indices != [(0, i) for i in range(num_chunks)]:
            raise ValueError(f"Invalid chunk indices for one benchmark segment: {indices!r}")
        if num_chunks > 1 and self.family != "aed":
            raise ValueError(f"Unexpected multi-chunk processor output for {self.family}")
        for i, index in enumerate(indices):
            chunk = {}
            for key, value in inputs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if value.ndim == 0 or value.shape[0] != num_chunks:
                    raise ValueError(f"Processor tensor {key!r} lacks the expected chunk batch dimension")
                chunk[key] = value[i : i + 1].to(self.device)
            if self.family == "aed":
                chunk = self.prepare_aed_policy(chunk)
            yield index, chunk

    def decode_chunks(self, waveform, decode):
        """One hypothesis per original segment; retain chunk-local diagnostics."""
        results, indices = [], []
        for index, inputs in self.input_chunks(waveform):
            indices.append(index)
            results.append(decode(inputs))
        if len(results) == 1:
            return results[0]
        texts = [result["text"] for result in results]
        # HF 5.16.1 indexes non_empty[0] for split audio. An all-empty decode is
        # a legitimate deletion hypothesis, so handle that corner case explicitly.
        text = self.processor._reassemble_chunk_texts(texts, indices, " ")[0] if any(t.strip() for t in texts) else ""
        result = dict(text=text, chunks=[dict(r, audio_chunk_index=index) for r, index in zip(results, indices)])
        if all("token_ids" in r for r in results):
            result["token_ids"] = [token for r in results for token in r["token_ids"]]
            result["token_chunk_lengths"] = [len(r["token_ids"]) for r in results]
        # Do not flatten traceback times or raw prompt/EOS paths across chunks.
        # They are chunk-local and are preserved separately in result['chunks'].
        return result

    def prepare_aed_policy(self, inputs):
        """Use generate()'s actual prefix (including any automatically prepended BOS).

        Keep generation-time token suppression/forced-token policy identical on
        the RASR side. These private interfaces are covered by adapter tests and
        require the recorded Transformers version; do not silently approximate.
        """
        config, _ = self.model._prepare_generation_config(
            None, max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False
        )
        self.model._prepare_special_tokens(config, kwargs_has_attention_mask=True, device=self.device)
        prompt, inputs = self.model._prepare_decoder_input_ids_for_generation(
            batch_size=1,
            model_input_name="input_features",
            model_kwargs=dict(inputs),
            decoder_start_token_id=config._decoder_start_token_tensor,
            device=self.device,
        )
        config.max_length = prompt.shape[1] + self.max_new_tokens
        self.logits_processors = self.model._get_logits_processor(
            config, input_ids_seq_length=prompt.shape[1], device=self.device, model_kwargs=inputs
        )
        inputs["decoder_input_ids"] = prompt
        return inputs

    def ctc_logits(self, inputs):
        logits = self.model(**inputs).logits[0]
        # Native generate() masks frames past the downsampled input length.
        # Remove exactly those frames for both decoders, even with batch size 1.
        if inputs.get("attention_mask") is not None:
            mask = self.model._get_output_attention_mask(inputs["attention_mask"], target_length=logits.shape[0])[0]
            logits = logits[mask.bool()]
        return logits

    def detokenize(self, ids):
        # Already collapsed / nonblank ids. Never collapse repeated RNNT/AED symbols.
        kwargs = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if self.family in ("ctc", "rnnt"):
            kwargs["group_tokens"] = False
        text = self.tokenizer.decode(ids, **kwargs)
        return self.postprocess(text)

    def postprocess(self, text):
        if self.family == "aed":
            # The official Cohere runner removes bracket characters BEFORE the
            # normalizer, otherwise text inside them would be dropped as an aside.
            text = text.translate(str.maketrans("", "", "()[]"))
        return text

    @torch.inference_mode()
    def native(self, waveform):
        return self.decode_chunks(waveform, self.native_inputs)

    def native_inputs(self, inputs):
        if self.family == "ctc":
            # Native CTC benchmark: argmax, then the model processor's CTC decoder.
            raw = self.ctc_logits(inputs).argmax(-1).unsqueeze(0)
            text = self.processor.batch_decode(raw, skip_special_tokens=True)[0]
            ids = collapse_ctc(raw[0].tolist(), self.blank)
        else:
            generated = self.model.generate(
                **inputs,
                **(
                    dict(max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False)
                    if self.family == "aed"
                    else {}
                ),
            )
            raw = generated.sequences if hasattr(generated, "sequences") else generated
            text = self.processor.batch_decode(raw, skip_special_tokens=True)[0]
            ids = raw[0].tolist()
            if self.family == "aed":
                ids = ids[inputs["decoder_input_ids"].shape[1] :]
                if not ids or ids[-1] != self.eos:
                    if len(ids) != self.max_new_tokens:
                        raise ValueError("Native AED stopped before its token limit without EOS")
                    print(f"Native AED: returning partial hypothesis after {len(ids)} labels", flush=True)
                    return dict(
                        text=self.postprocess(text), token_ids=ids, raw_token_ids=raw[0].tolist(), truncated=True
                    )
                ids = ids[:-1]
            else:
                ids = [i for i in ids if i != self.blank]
        return dict(text=self.postprocess(text), token_ids=ids, raw_token_ids=raw[0].tolist())

    @torch.inference_mode()
    def encode(self, waveform):
        return self.encode_inputs(self.inputs(waveform))

    def encode_inputs(self, inputs):
        inputs = dict(inputs)
        if self.family == "ctc":
            return -self.ctc_logits(inputs).float().log_softmax(-1).cpu().numpy()
        if self.family == "rnnt":
            output = self.model.get_audio_features(**inputs)
            self.encoder = output.pooler_output
            length = int(output.attention_mask[0].sum())
        else:
            self.prompt = inputs.pop("decoder_input_ids")[0].tolist()
            self.encoder = self.model.get_encoder()(**inputs)
            length = self.encoder.last_hidden_state.shape[1]
        self.num_frames = length
        # The Python scorer owns the encoder tensor on GPU. AED attends to the
        # entire encoder output, not a particular RASR feature frame. Send one
        # placeholder so max-labels-per-timestep is an absolute output cap,
        # equal to native max_new_tokens (not cap * encoder length).
        # RNNT still needs the actual encoder frame count.
        return np.zeros((1 if self.family == "aed" else length, 1), dtype=np.float32)

    @torch.inference_mode()
    def rasr(self, waveform, search, vocabulary, lexiconfree):
        def decode(inputs):
            features = self.encode_inputs(inputs)
            # recognize_segment resets both ASR and external LM history for each
            # processor chunk, matching independent native generate() calls.
            trace = search.recognize_segment(np.ascontiguousarray(features, dtype=np.float32))
            if self.family == "aed":
                ended = any(
                    vocabulary.get(item.lemma) == self.eos if lexiconfree else item.lemma == "[SENTENCE-END]"
                    for item in trace
                )
            result = traceback_result(trace, self, vocabulary, lexiconfree)
            if self.family == "aed" and not ended:
                print("RASR AED: returning partial hypothesis without EOS", flush=True)
                result["truncated"] = True
            return result

        return self.decode_chunks(waveform, decode)

    @torch.inference_mode()
    def score_contexts(self, contexts):
        """Context = (encoder time, output-token tuple); score in length buckets."""
        answers = [None] * len(contexts)
        buckets = defaultdict(list)
        for i, (t, prefix) in enumerate(contexts):
            if self.family == "rnnt" and t >= self.num_frames:
                continue
            buckets[len(prefix)].append(i)
        for indices in buckets.values():
            for start in range(0, len(indices), 8):
                batch = indices[start : start + 8]
                tokens = [(self.prompt if self.family == "aed" else [self.blank]) + list(contexts[i][1]) for i in batch]
                ids = torch.tensor(tokens, dtype=torch.long, device=self.device)
                if self.family == "rnnt":
                    predictor = self.model.decoder(ids, cache=None)[:, -1, :]
                    encoder = torch.cat([self.encoder[:, contexts[i][0], :] for i in batch])
                    logits = self.model.joint(decoder_hidden_states=predictor, encoder_hidden_states=encoder)
                else:
                    hidden = self.encoder.last_hidden_state.expand(len(batch), -1, -1)
                    mask = self.encoder.attention_mask.expand(len(batch), -1)
                    decoded = self.model.model.decoder(
                        input_ids=ids, encoder_hidden_states=hidden, encoder_attention_mask=mask, use_cache=False
                    )
                    logits = self.model.proj_out(decoded.last_hidden_state[:, -1])
                    logits = self.logits_processors(ids, logits.float())
                costs = -logits.float().log_softmax(-1).cpu().numpy()
                for i, scores in zip(batch, costs):
                    answers[i] = (scores.tolist(), contexts[i][0] if self.family == "rnnt" else len(contexts[i][1]))
        return answers


def register_hf_scorer(librasr, adapter):
    """Register against the explicitly selected binary, before creating search."""
    trans = librasr.TransitionType
    aed_types = [trans.INITIAL_LABEL, trans.LABEL_TO_LABEL, trans.LABEL_LOOP, trans.SENTENCE_END]
    rnnt_types = aed_types[:-1] + [trans.INITIAL_BLANK, trans.LABEL_TO_BLANK, trans.BLANK_TO_LABEL, trans.BLANK_LOOP]

    class Scorer(librasr.LabelScorer):
        def __init__(self, config):
            super().__init__(config)

        def reset(self):
            pass  # No score/history cache; encode() replaces the utterance tensors.

        def signal_no_more_features(self):
            pass

        def get_initial_scoring_context(self):
            return (0, ())

        def allowed_transition_types(self):
            return aed_types if adapter.family == "aed" else rnnt_types

        def extended_scoring_context(self, context, token, transition_type):
            t, prefix = context
            if adapter.family == "rnnt" and token == adapter.blank:
                return (t + 1, prefix)
            if transition_type == trans.SENTENCE_END:
                return context
            return (t, prefix + (token,))

        def add_inputs(self, inputs):
            pass

        def compute_scores_with_times(self, contexts):
            return adapter.score_contexts(contexts)

    librasr.register_label_scorer_type("hf-benchmark", Scorer)


def traceback_result(trace, adapter, vocabulary, lexiconfree):
    items = [
        dict(
            lemma=x.lemma,
            am_score=float(x.am_score),
            lm_score=float(x.lm_score),
            start_time=int(x.start_time),
            end_time=int(x.end_time),
        )
        for x in trace
    ]
    if lexiconfree:
        unexpected = {x.lemma for x in trace if x.lemma and x.lemma not in vocabulary}
        if unexpected:
            raise ValueError(f"Traceback lemmata missing from the acoustic vocabulary: {sorted(unexpected)}")
        ids = [vocabulary[x.lemma] for x in trace if x.lemma in vocabulary]
        ids = [i for i in ids if i not in (adapter.blank, adapter.eos)]
        return dict(text=adapter.detokenize(ids), token_ids=ids, traceback=items)
    if any(item.lemma and item.lemma.endswith("&&") for item in trace):
        if adapter.family != "ctc":
            raise ValueError("Benchmark piece fallback is only supported for Granite CTC")
        from .benchmark_oov import granite_fallback_result

        result = granite_fallback_result(trace, adapter.tokenizer)
        result["text"] = adapter.postprocess(result["text"])
        return dict(result, traceback=items)
    # Closed-tree output is whole word lemmata, not BPE text. Do not accidentally
    # strip literal @@ or other characters using a subword traceback formatter.
    words = []
    for item in trace:
        name = item.lemma
        if name.startswith("qwen-v1:"):
            fields = name.split(":", 3)
            if len(fields) != 4 or not fields[1].isdigit() or not fields[2].isdigit():
                raise ValueError(f"Malformed Qwen lemma {name!r}")
            name = fields[3]
        if name and name not in ("[BLANK]", "[SENTENCE-BEGIN]", "[SENTENCE-END]", "[SILENCE]"):
            words.append(name)
    text = " ".join(words)
    return dict(text=adapter.postprocess(text), traceback=items)


class HfBenchmarkRecogJob(Job):
    def __init__(
        self,
        dataset,
        repo,
        family,
        *,
        code_version: str,
        native=True,
        rasr_config=None,
        rasr_binary=None,
        vocabulary=None,
        dtype="float32",
        max_new_tokens=500,
        transformers_version="5.16.1",
    ):
        self.dataset, self.repo, self.family = dataset, repo, family
        self.code_version, self.native = code_version, native
        self.rasr_config, self.rasr_binary, self.vocabulary = rasr_config, rasr_binary, vocabulary
        self.dtype, self.max_new_tokens = dtype, max_new_tokens
        self.transformers_version = transformers_version
        if not native and (rasr_config is None or rasr_binary is None):
            raise ValueError("RASR recognition requires a config and pinned binary")
        self.out_hypotheses = self.output_path("hypotheses.jsonl")
        self.out_runtime = self.output_path("runtime.json")
        self.out_log = self.output_path("recognition.log")
        self.out_rasr_log = self.output_path("rasr.recog.log") if not native else None
        self.rqmt = {"cpu": 2, "mem": 64, "gpu": 1, "gpu_mem": 24, "time": 48}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import transformers
        from datasets import load_from_disk

        if transformers.__version__ != self.transformers_version:
            raise RuntimeError(
                f"Benchmark requires transformers=={self.transformers_version}; "
                f"found {transformers.__version__}. Use the benchmark container/environment."
            )
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"Loading {self.family} model {self.repo} ({self.dtype}); native={self.native}", flush=True)
        adapter = HfBenchmarkAdapter(self.repo.get_path(), self.family, self.dtype, max_new_tokens=self.max_new_tokens)
        search, vocabulary = None, {}
        if not self.native:
            sys.path.insert(0, self.rasr_binary.get_path())
            import librasr

            if self.family != "ctc":
                register_hf_scorer(librasr, adapter)
            cfg = librasr.Configuration()
            if not cfg.set_from_file(self.rasr_config.get_path()):
                raise ValueError("Failed to load RASR config")
            # Logging policy only: also overrides previously written configs,
            # without requiring completed recognition/scoring jobs to rerun.
            cfg.set("lib-rasr.search-algorithm.log-stepwise-statistics", "false")
            if self.family == "aed":
                # One placeholder frame per chunk: native and RASR have the
                # same absolute generated-label cap, excluding the prompt.
                cfg.set("lib-rasr.search-algorithm.max-labels-per-timestep", str(self.max_new_tokens))
            for channel in ("log", "warning", "error", "statistics"):
                cfg.set(f"*.{channel}.channel", self.out_rasr_log.get_path())
            librasr.init_logging(cfg)
            search = librasr.SearchAlgorithm(config=cfg)
            if self.vocabulary is not None:
                with open(self.vocabulary.get_path()) as f:
                    vocabulary = {s.rstrip("\n"): i for i, s in enumerate(f)}
        ds = load_from_disk(self.dataset.get_path())
        print(f"Recognizing {len(ds)} benchmark segments", flush=True)
        audio_seconds = elapsed = 0.0
        with open(self.out_hypotheses.get_path(), "w") as dst, open(self.out_log.get_path(), "w") as log:
            for i, sample in enumerate(ds):
                waveform = audio_array(sample["audio"])
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    if self.native:
                        result = adapter.native(waveform)
                    else:
                        result = adapter.rasr(waveform, search, vocabulary, self.vocabulary is not None)
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                audio_seconds += len(waveform) / 16000
                elapsed += duration
                result.update(id=str(sample["id"]), elapsed_seconds=duration, audio_seconds=len(waveform) / 16000)
                dst.write(json.dumps(result, ensure_ascii=False) + "\n")
                dst.flush()
                message = (
                    f"{i + 1}/{len(ds)} {sample['id']}: {duration:.2f}s, aggregate RTF={elapsed / audio_seconds:.3f}"
                )
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
        with open(self.out_runtime.get_path(), "w") as f:
            json.dump(
                dict(
                    num_segments=len(ds),
                    audio_seconds=audio_seconds,
                    elapsed_seconds=elapsed,
                    rtf=elapsed / audio_seconds,
                    dtype=self.dtype,
                    torch_version=torch.__version__,
                    transformers_version=transformers.__version__,
                    cuda_device=torch.cuda.get_device_name(),
                    native=self.native,
                    pid=os.getpid(),
                ),
                f,
                indent=2,
            )
