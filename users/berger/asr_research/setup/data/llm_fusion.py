"""Pinned Qwen preparation and numerical gates for closed-tree shallow fusion."""

import copy
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter

from i6_core.util import uopen, write_xml
from sisyphus import Job, Task

QWEN_REPO = "Qwen/Qwen3-0.6B-Base"
QWEN_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
QWEN_BASE_MODELS = {
    "0.6b": (QWEN_REPO, QWEN_REVISION),
    "1.7b": ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),
}
SCORE_CASES = (("HELLO", "WORLD"), ("UNCHARACTERISTICALLY", "QUIET"), ("DON'T", "STOP"), ("HELLO",))


def token_name(index):
    return f"llm_{index:06d}"


def encode_word(tokenizer, word):
    """One leading space for EVERY word, including the first; lowercase only."""
    if not word or word != word.strip() or any(c.isspace() for c in word):
        raise ValueError(f"Expected a single surface word, got {word!r}")
    surface = " " + word.lower()
    ids = tokenizer.encode(surface, add_special_tokens=False)
    if not ids or set(ids) & set(tokenizer.all_special_ids):
        raise ValueError(f"Invalid/special LLM tokens for {word!r}: {ids}")
    if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != surface:
        raise ValueError(f"Non-round-tripping LLM tokenization for {word!r}")
    return ids


def encode_words(tokenizer, words):
    ids = [i for w in words for i in encode_word(tokenizer, w)]
    whole = tokenizer.encode("".join(" " + w.lower() for w in words), add_special_tokens=False)
    if ids != whole:
        raise ValueError(f"Wordwise and full-string tokenization differ: {words!r}")
    return ids


def rewrite_lexicon(root, tokenizer, config):
    """Only synt changes. Keep acoustic inventory, pronunciations and orths exact."""
    root = copy.deepcopy(root)
    sizes, words = Counter(), []
    for lemma in root.findall("lemma"):
        special = lemma.get("special", "")
        if special.startswith("unknown-"):
            raise ValueError("LLM prototype is closed-tree only; do not silently map fallback to UNK")
        synt = lemma.find("synt")
        if special in ("sentence-begin", "sentence-end", "sentence-boundary"):
            ids = [config["bos_token_id"] if special == "sentence-begin" else config["eos_token_id"]]
        elif special == "unknown":
            if any((p.text or "").strip() for p in lemma.findall("phon")):
                raise ValueError("Closed-tree unknown anchor must have no acoustic pronunciation")
            ids = []  # This LLM has no linguistic UNK; never substitute EOS for it.
        elif (synt is not None and not synt.findall("tok")) or special in ("blank", "silence", "nonword"):
            ids = []  # Preserve neutral/punctuation lemma semantics.
        else:
            word = lemma.findtext("orth")
            ids = encode_word(tokenizer, word)
            words.append(word)
            sizes[len(ids)] += 1
        if any(not isinstance(i, int) or not 0 <= i < config["vocab_size"] for i in ids):
            raise ValueError(f"LLM id outside softmax inventory: {ids}")
        if synt is not None:
            lemma.remove(synt)
        synt = ET.SubElement(lemma, "synt")
        for i in ids:
            ET.SubElement(synt, "tok").text = token_name(i)
    # Exhaustively check adjacent dictionary words, not just isolated encodings.
    for offset in range(0, len(words), 64):
        encode_words(tokenizer, words[offset : offset + 64])
    return root, dict(num_words=len(words), syntactic_lengths=dict(sizes), policy="lowercase-space-prefixed-every-word")


class QwenLexiconJob(Job):
    def __init__(self, lexicon, repo, validation_text=None):
        self.lexicon, self.repo, self.validation_text = lexicon, repo, validation_text
        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_vocab = self.output_path("lm.vocab.txt")
        self.out_report = self.output_path("tokenization.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 12, "time": 2})

    def run(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.repo.get_path(), local_files_only=True)
        with open(os.path.join(self.repo.get_path(), "config.json")) as f:
            config = json.load(f)
        with uopen(self.lexicon, "rt") as f:
            source = ET.parse(f).getroot()
        print(f"Checking Qwen tokenization for {len(source.findall('lemma'))} lemmata", flush=True)
        rewritten, report = rewrite_lexicon(source, tokenizer, config)
        assert ET.tostring(source.find("phoneme-inventory")) == ET.tostring(rewritten.find("phoneme-inventory"))
        count = 0
        if self.validation_text is not None:
            with uopen(self.validation_text, "rt") as f:
                for line in f:
                    encode_words(tokenizer, line.split())
                    count += 1
        write_xml(self.out_lexicon.get_path(), rewritten)
        with open(self.out_vocab.get_path(), "w") as f:
            # Includes unused padded output columns: ids are NEVER compacted.
            for i in range(config["vocab_size"]):
                f.write(f"{token_name(i)} {i}\n")
        report.update(
            validation_sentences=count,
            vocab_size=config["vocab_size"],
            bos_id=config["bos_token_id"],
            eos_id=config["eos_token_id"],
        )
        with open(self.out_report.get_path(), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Qwen lexical tokenization passed: {report}", flush=True)


class ExportQwenLmJob(Job):
    def __init__(self, repo):
        self.repo = repo
        self.out_model = self.output_path("model.onnx")
        self.out_report = self.output_path("hf_comparison.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 48, "time": 4})

    def run(self):
        import numpy as np
        import torch
        import onnxruntime as ort
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ..model_pipelines.transformer_lm.qwen import Qwen3RasrLm, export_qwen_onnx, verify_qwen_export

        torch.set_num_threads(2)
        print("Loading pinned Qwen Base checkpoint in float32", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(self.repo.get_path(), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.repo.get_path(), local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
        sequences = [[model.config.bos_token_id] + encode_words(tokenizer, words) for words in SCORE_CASES]
        cases = []
        with torch.inference_mode():
            for words, sequence in zip(SCORE_CASES, sequences):
                scores = -model(torch.tensor([sequence]), use_cache=False).logits[0].float().log_softmax(-1)
                targets = sequence[1:] + [model.config.eos_token_id]
                nll = scores[torch.arange(len(sequence)), torch.tensor(targets)]
                cases.append(
                    dict(
                        words=list(words),
                        input_ids=sequence,
                        target_ids=targets,
                        token_costs=nll.tolist(),
                        total_cost=float(nll.double().sum()),
                    )
                )
        print("Exporting Qwen with the RASR KV-cache interface", flush=True)
        export_qwen_onnx(Qwen3RasrLm(model), self.out_model.get_path())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        session = ort.InferenceSession(self.out_model.get_path(), sess_options=opts, providers=["CPUExecutionProvider"])
        print("Comparing cached ONNX scores against full-prefix Hugging Face scores", flush=True)
        error = verify_qwen_export(model, session, sequences)
        if not all(np.isfinite(c["total_cost"]) for c in cases):
            raise ValueError("Nonfinite HF scores")
        with open(self.out_report.get_path(), "w") as f:
            json.dump(dict(passed=True, max_abs_error=error, dtype="float32", cases=cases), f, indent=2)
        print(f"Qwen ONNX/HF validation passed: max absolute error {error:g}", flush=True)


NATIVE_RUNNER = r"""
import sys,json,numpy as np
sys.path.insert(0,sys.argv[1])
import librasr
cfg=librasr.Configuration()
assert cfg.set_from_file(sys.argv[2])
librasr.init_logging(cfg)
search=librasr.SearchAlgorithm(config=cfg)
case=json.loads(sys.argv[3])
path=[0]
for label in case["labels"]: path.extend([label,label,0])
scores=np.full((len(path),case["vocab_size"]),10000,dtype=np.float32)
scores[np.arange(len(path)),path]=0
trace=search.recognize_segment(scores)
trace_items=[dict(lemma=x.lemma,am_score=float(x.am_score),lm_score=float(x.lm_score),
                  start_time=x.start_time,end_time=x.end_time) for x in trace]
with open("traceback.json","w") as f: json.dump(trace_items,f,indent=2)
assert trace and trace[-1].lemma=="[EOS]",("Missing EOS traceback entry",trace_items)
orths=[x.lemma for x in trace if x.lemma and not x.lemma.startswith(("[","<"))]
assert orths==case["words"],(orths,case)
cost=float(trace[-1].am_score+trace[-1].lm_score)
assert abs(cost-case["expected_cost"])<0.01,(cost,case)
with open("result.json","w") as f: json.dump(dict(cost=cost,case=case),f)
"""


class ValidateQwenRasrJob(Job):
    """Gate the real lexicon on HF vs native multi-token word-exit scoring."""

    __sis_hash_exclude__ = {"variant_aliases": None, "score_cases": SCORE_CASES}

    def __init__(
        self,
        lexicon,
        repo,
        model,
        vocab,
        hf_report,
        binary,
        correctness_report,
        pruning_report,
        python_exe,
        runner=NATIVE_RUNNER,
        variant_aliases=None,
        score_cases=SCORE_CASES,
    ):
        from i6_core.lib.lexicon import Lexicon, Lemma
        from i6_core.lexicon.modification import WriteLexiconJob
        from .librispeech.lm import OnnxTokenLmParams, get_word_lm_config
        from ..model_pipelines.common.recog_rasr_config import TreeTimesyncRecogParams, get_tree_timesync_recog_config
        from ..model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config

        self.lexicon, self.repo, self.model, self.vocab = lexicon, repo, model, vocab
        self.hf_report, self.binary, self.python_exe, self.runner = hf_report, binary, python_exe, runner
        self.correctness_report, self.pruning_report = correctness_report, pruning_report
        self.words = sorted({w for words in score_cases for w in words})
        toy = Lexicon()
        toy.add_phoneme("blank", variation="none")
        toy.add_lemma(Lemma(orth=["[BLANK]"], phon=["blank"], synt=[], special="blank"))
        for i, w in enumerate(self.words, 1):
            toy.add_phoneme(f"p{i}", variation="none")
            toy.add_lemma(Lemma(orth=[w], phon=[f"p{i}"], synt=[] if not any(c.isalnum() for c in w) else [w]))
        # A zero-length pronunciation is different from no pronunciation:
        # Python RASR drops trace entries whose pronunciation pointer is null.
        # Retain the EOS item so trace[-1] includes the final LM contribution.
        toy.add_lemma(Lemma(orth=["[BOS]"], phon=[""], synt=["<s>"], special="sentence-begin"))
        toy.add_lemma(Lemma(orth=["[EOS]"], phon=[""], synt=["</s>"], special="sentence-end"))
        if variant_aliases is None:
            toy_lex = QwenLexiconJob(WriteLexiconJob(toy).out_bliss_lexicon, repo).out_lexicon
        else:
            from .qwen_text_variants import QwenTextVariantLexiconJob

            toy_lex = QwenTextVariantLexiconJob(
                WriteLexiconJob(toy).out_bliss_lexicon, repo, alias_manifest=variant_aliases
            ).out_lexicon
        self.configs = {}
        for scale in (1.0, 0.3):
            lm = get_word_lm_config(toy_lex, OnnxTokenLmParams(model, vocab, scale=scale, use_gpu=False))
            self.configs[scale] = get_tree_timesync_recog_config(
                toy_lex,
                [get_no_op_label_scorer_config()],
                TreeTimesyncRecogParams(
                    collapse_repeated_labels=True,
                    max_beam_sizes=[256],
                    score_thresholds=[200.0],
                    max_word_end_beam_size=256,
                    word_end_score_threshold=200.0,
                    sentence_end_fallback=False,
                    unknown_word_fallback="disabled",
                ),
                lm_config=lm,
            )
        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_report = self.output_path("native_comparison.json")
        self.out_logs = self.output_path("logs", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 2})

    def run(self):
        for report in (self.hf_report, self.correctness_report, self.pruning_report):
            with open(report.get_path()) as f:
                if not json.load(f).get("passed"):
                    raise ValueError(f"Preflight did not pass: {report}")
        with open(self.hf_report.get_path()) as f:
            cases = json.load(f)["cases"]
        results = []
        for scale, config in self.configs.items():
            for i, case in enumerate(cases):
                print(f"Native Qwen scoring check: scale={scale}, words={case['words']}", flush=True)
                directory = os.path.join(self.out_logs.get_path(), f"scale{scale}-case{i}")
                os.makedirs(directory, exist_ok=True)
                payload = dict(
                    words=case["words"],
                    labels=[1 + self.words.index(w) for w in case["words"]],
                    vocab_size=1 + len(self.words),
                    expected_cost=scale * case["total_cost"],
                )
                with open(os.path.join(directory, "stdout.log"), "w") as f:
                    try:
                        subprocess.run(
                            [
                                self.python_exe.get_path(),
                                "-c",
                                self.runner,
                                self.binary.get_path(),
                                config.get_path(),
                                json.dumps(payload),
                            ],
                            cwd=directory,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=300,
                        )
                    except subprocess.SubprocessError:
                        f.flush()
                        with open(f.name) as captured:
                            print(
                                f"Native scoring failed; tail of {f.name}:\n" + "".join(captured.readlines()[-40:]),
                                flush=True,
                            )
                        raise
                with open(os.path.join(directory, "result.json")) as f:
                    results.append(json.load(f))
        with open(self.out_report.get_path(), "w") as f:
            json.dump(dict(passed=True, results=results), f, indent=2)
        shutil.copyfile(self.lexicon.get_path(), self.out_lexicon.get_path())
