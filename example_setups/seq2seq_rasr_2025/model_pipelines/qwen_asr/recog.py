"""Qwen-ASR through a Python RASR label scorer, without an external LM.

This deliberately simple example recomputes short decoder histories on GPU.
It requires a Transformers build with Qwen3ASR support (tested research runtime:
5.16.1). Model download, vocabulary preparation and recognition are separate jobs.
Scoring uses the corpus's ordinary STM/CTM pipeline, without leaderboard rules.
"""

import json
import re
from collections import defaultdict

from sisyphus import Job, Task

MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"
MODEL_REVISION = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"


class DownloadQwenAsrJob(Job):
    def __init__(self, *, cache_dir=None):
        self.cache_dir = cache_dir
        self.out_repo = self.output_path("model", directory=True)

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=4))

    def run(self):
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, local_dir=self.out_repo.get_path(), cache_dir=self.cache_dir
        )


class QwenAsrVocabJob(Job):
    def __init__(self, repo):
        self.repo = repo
        self.out_vocab = self.output_path("labels.txt")
        self.out_info = self.output_path("vocab.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=1))

    def run(self):
        from transformers import AutoConfig, AutoProcessor

        tokenizer = AutoProcessor.from_pretrained(self.repo.get_path()).tokenizer
        config = AutoConfig.from_pretrained(self.repo.get_path())
        size = config.text_config.vocab_size
        if any(i < 0 or i >= size for i in tokenizer.get_vocab().values()):
            raise ValueError("Tokenizer ids exceed model output vocabulary")
        eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos is None or eos == tokenizer.unk_token_id or not 0 <= eos < size:
            raise ValueError("Missing Qwen assistant-turn terminator")
        # Safe, unique RASR orthographies; ids are the model output indices.
        with open(self.out_vocab.get_path(), "w") as dst:
            dst.writelines(f"t{i}\n" for i in range(size))
        with open(self.out_info.get_path(), "w") as dst:
            json.dump(dict(size=size, eos=eos), dst)


class QwenAsrBackend:
    def __init__(self, repo, *, device="cuda", batch_size=2):
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.device, self.batch_size = device, batch_size
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(repo, dtype=torch.float32).to(device).eval()

    def set_audio(self, waveform, sample_rate):
        import torch

        inputs = self.processor.apply_transcription_request(
            audio=waveform, language="English", sampling_rate=sample_rate, return_tensors="pt"
        ).to(self.device)
        if inputs["input_ids"].shape[0] != 1 or not inputs["attention_mask"].all():
            raise ValueError("Expected one unpadded audio prompt")
        with torch.inference_mode():
            prompt = self.model.get_input_embeddings()(inputs["input_ids"])
            audio = self.model.get_audio_features(
                inputs["input_features"], inputs["input_features_mask"], return_dict=True
            ).pooler_output
            mask = self.model.model.get_placeholder_mask(inputs["input_ids"], prompt, audio)
            self.prompt_embeddings = prompt.masked_scatter(mask, audio.to(prompt.dtype))
        self.prompt_ids = inputs["input_ids"][0].tolist()

    def score(self, contexts):
        import torch

        result = [None] * len(contexts)
        buckets = defaultdict(list)
        for i, prefix in enumerate(contexts):
            buckets[len(prefix)].append(i)
        with torch.inference_mode():
            for indices in buckets.values():
                for start in range(0, len(indices), self.batch_size):
                    batch = indices[start : start + self.batch_size]
                    ids = torch.tensor([self.prompt_ids + list(contexts[i]) for i in batch], device=self.device)
                    embeddings = self.model.get_input_embeddings()(ids)
                    embeddings[:, : len(self.prompt_ids)] = self.prompt_embeddings.expand(len(batch), -1, -1)
                    logits = (
                        self.model(inputs_embeds=embeddings, attention_mask=torch.ones_like(ids), use_cache=False)
                        .logits[:, -1]
                        .float()
                    )
                    costs = -logits.log_softmax(-1)
                    if not torch.isfinite(costs).all():
                        raise ValueError("Non-finite Qwen scores")
                    for i, row in zip(batch, costs.cpu().tolist()):
                        result[i] = (row, len(contexts[i]))
        return result


def register_scorer(librasr, backend):
    transition = librasr.TransitionType

    class Scorer(librasr.LabelScorer):
        def __init__(self, config):
            super().__init__(config)

        def reset(self):
            pass

        def signal_no_more_features(self):
            pass

        def add_inputs(self, inputs):
            pass

        def get_initial_scoring_context(self):
            return ()

        def allowed_transition_types(self):
            return [transition.INITIAL_LABEL, transition.LABEL_TO_LABEL, transition.LABEL_LOOP, transition.SENTENCE_END]

        def extended_scoring_context(self, context, token, transition_type):
            return context if transition_type == transition.SENTENCE_END else context + (token,)

        def compute_scores_with_times(self, contexts):
            return backend.score(contexts)

    librasr.register_label_scorer_type("qwen-asr", Scorer)


def format_for_corpus(text, *, uppercase=False, strip_punctuation=False):
    """Optional output formatting only; reference text is never rewritten."""
    if strip_punctuation:
        import unicodedata

        text = text.replace("’", "'").replace("‘", "'")
        text = re.sub(r"[-‐-―]", " ", text)
        text = "".join(c for c in text if c == "'" or not unicodedata.category(c).startswith("P"))
        text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return " ".join((text.upper() if uppercase else text).split())


class QwenAsrRecogJob(Job):
    def __init__(
        self,
        corpus,
        repo,
        vocab,
        vocab_info,
        rasr_binary,
        *,
        beam=4,
        threshold=12.0,
        max_labels=500,
        uppercase=False,
        strip_punctuation=False,
    ):
        if beam < 1 or threshold < 0 or max_labels < 1:
            raise ValueError("Invalid search budget")
        self.corpus, self.repo, self.vocab, self.vocab_info, self.rasr_binary = (
            corpus,
            repo,
            vocab,
            vocab_info,
            rasr_binary,
        )
        self.beam, self.threshold, self.max_labels = beam, threshold, max_labels
        self.uppercase, self.strip_punctuation = uppercase, strip_punctuation
        self.out_ctm = self.output_path("hyp.ctm")
        self.out_hypotheses = self.output_path("hypotheses.jsonl")
        self.out_log = self.output_path("rasr.log")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=2, mem=24, gpu=1, gpu_mem=24, time=24))

    def run(self):
        import sys
        import numpy as np
        import soundfile as sf
        import torch
        from i6_core.lib.corpus import Corpus

        sys.path.insert(0, self.rasr_binary.get_path())
        import librasr

        torch.set_num_threads(2)
        backend = QwenAsrBackend(self.repo.get_path())
        register_scorer(librasr, backend)
        with open(self.vocab_info.get_path()) as src:
            info = json.load(src)
        cfg = librasr.Configuration()
        settings = {
            "lib-rasr.lexicon.file": "vocab-text:" + self.vocab.get_path(),
            "lib-rasr.search-algorithm.type": "lexiconfree-labelsync-beam-search",
            "lib-rasr.label-scorer.type": "qwen-asr",
            "lib-rasr.label-scorer.scale": "1.0",
            "lib-rasr.search-algorithm.max-beam-size": str(self.beam),
            "lib-rasr.search-algorithm.score-threshold": str(self.threshold),
            "lib-rasr.search-algorithm.sentence-end-label-index": str(info["eos"]),
            "lib-rasr.search-algorithm.max-labels-per-timestep": str(self.max_labels),
            "lib-rasr.search-algorithm.log-stepwise-statistics": "false",
            "lib-rasr.search-algorithm.recombination-mode": "off",
            "*.encoding": "UTF-8",
        }
        for channel in ("log", "warning", "error", "statistics"):
            settings[f"*.{channel}.channel"] = self.out_log.get_path()
        for key, value in settings.items():
            cfg.set(key, value)
        librasr.init_logging(cfg)
        search = librasr.SearchAlgorithm(config=cfg)
        corpus = Corpus()
        corpus.load(self.corpus.get_path())
        with open(self.out_ctm.get_path(), "w") as ctm, open(self.out_hypotheses.get_path(), "w") as output:
            for segment in corpus.segments():
                with sf.SoundFile(segment.recording.audio) as audio:
                    rate = audio.samplerate
                    audio.seek(round(segment.start * rate))
                    waveform = audio.read(round((segment.end - segment.start) * rate), dtype="float32")
                if rate != 16000 or waveform.ndim != 1 or not len(waveform):
                    raise ValueError("This example expects mono 16 kHz Bliss audio")
                backend.set_audio(waveform, rate)
                trace = search.recognize_segment(np.zeros((1, 1), dtype=np.float32))
                ids = [int(item.lemma[1:]) for item in trace if item.lemma and re.fullmatch(r"t\d+", item.lemma)]
                raw = backend.processor.decode(ids, return_format="transcription_only")
                text = format_for_corpus(raw, uppercase=self.uppercase, strip_punctuation=self.strip_punctuation)
                output.write(
                    json.dumps(
                        dict(
                            id=segment.fullname(),
                            raw=raw,
                            hypothesis=text,
                            token_ids=ids,
                            truncated=info["eos"] not in ids,
                        )
                    )
                    + "\n"
                )
                words = text.split()
                # AED supplies token order, not word alignment. Uniform placement
                # within each segment is for STM/CTM scoring only.
                duration = (segment.end - segment.start) / max(1, len(words))
                track = segment.track + 1 if segment.track else 1
                for i, word in enumerate(words):
                    ctm.write(
                        f"{segment.recording.name} {track} {segment.start + i*duration:.3f} {duration:.3f} {word}\n"
                    )
                print(f"{segment.fullname()}: {text}", flush=True)
