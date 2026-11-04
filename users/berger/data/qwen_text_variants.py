"""Alternative LM spellings of the same acoustic word, with start-only aliases.

Aliases are extra LM columns, not new acoustic labels. They copy Qwen's probability
of an unspaced first token only immediately after BOS, and map back to that token
on history input. No renormalization or length normalization is performed.
"""

import copy
import itertools
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter

from i6_core.util import uopen, write_xml
from sisyphus import Job, Task

from .llm_fusion import token_name, NATIVE_RUNNER


SCORE_CASES = (("HELLO", "WORLD"), ("DON'T", "STOP"), ("EUGENIUS",), ("NETTLES", ",", "HERE"))
VARIANT_NATIVE_RUNNER = NATIVE_RUNNER.replace(
    "orths=[x.lemma for x in trace",
    'orths=[(x.lemma.split(":",3)[3] if x.lemma.startswith("qwen-v1:") else x.lemma) for x in trace',
)


def word_spellings(word):
    if not word or word != word.strip() or any(c.isspace() for c in word):
        raise ValueError(f"Expected a single lexical word: {word!r}")
    # capitalize(), not title(): DON'T -> Don't, rather than Don'T.
    return tuple(dict.fromkeys((word.lower(), word.capitalize(), word.upper())))


def encode_surface(tokenizer, surface):
    ids = tokenizer.encode(surface, add_special_tokens=False)
    if not ids or set(ids) & set(tokenizer.all_special_ids):
        raise ValueError(f"Invalid/special Qwen tokens for {surface!r}: {ids}")
    if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != surface:
        raise ValueError(f"Non-round-tripping Qwen surface: {surface!r}")
    return ids


def lemma_surfaces(lemma):
    """Return (surface, initial-only) choices; None means a structural lemma.

    Punctuation is only scored when its existing acoustic lemma is emitted. We do
    not invent acoustically empty punctuation loops or multiply each word by a
    Cartesian product of punctuation decorations. Separator '_' remains neutral.
    """
    if lemma.get("special"):
        return None
    word = lemma.findtext("orth")
    synt = lemma.find("synt")
    if synt is not None and not synt.findall("tok"):
        if word and all(not c.isalnum() and not c.isspace() for c in word) and word != "_":
            return [(word, False)]
        return None
    return [(prefix + spelling, not prefix) for spelling in word_spellings(word) for prefix in (" ", "")]


def rewrite_variant_lexicon(source, tokenizer, config, alias_ids=None):
    """Clone lemmata, retaining orth/phon/eval; only synt and unique names differ."""
    vocab_size = config["vocab_size"]
    if alias_ids is None:
        alias_ids = sorted(
            {
                encode_surface(tokenizer, text)[0]
                for lemma in source.findall("lemma")
                for text, initial in (lemma_surfaces(lemma) or [])
                if initial
            }
        )
    if alias_ids != sorted(set(alias_ids)) or any(not 0 <= i < vocab_size for i in alias_ids):
        raise ValueError("Invalid initial-token alias map")
    aliases = {token: vocab_size + i for i, token in enumerate(alias_ids)}
    root = ET.Element(source.tag, source.attrib)
    for child in source:
        if child.tag != "lemma":
            root.append(copy.deepcopy(child))
    counts, lengths = Counter(), Counter()
    for index, lemma in enumerate(source.findall("lemma")):
        special = lemma.get("special", "")
        if special.startswith("unknown-"):
            raise ValueError("Qwen text variants are closed-tree only")
        surfaces = lemma_surfaces(lemma)
        if surfaces is None:
            if special in ("sentence-begin", "sentence-end", "sentence-boundary"):
                ids = [config["bos_token_id"] if special == "sentence-begin" else config["eos_token_id"]]
            else:
                if special == "unknown" and any((p.text or "").strip() for p in lemma.findall("phon")):
                    raise ValueError("Unknown anchor must not have an acoustic pronunciation")
                ids = []
            choices = [(ids, "structural")]
        else:
            choices = []
            for surface, initial in surfaces:
                ids = encode_surface(tokenizer, surface)
                if initial:
                    ids = [aliases[ids[0]], *ids[1:]]
                choices.append((ids, "initial" if initial else "regular"))
        seen = set()
        for ids, kind in choices:
            if tuple(ids) in seen:
                continue
            seen.add(tuple(ids))
            if any(not 0 <= i < vocab_size + len(alias_ids) for i in ids):
                raise ValueError("Token index outside extended LM vocabulary")
            out = copy.deepcopy(lemma)
            # librasr exposes the unique lemma NAME, not <orth>, in traceback.
            # Make the mapping reversible; never leak RASR's auto '[1]' suffixes
            # into scoring, and retain the route identity in raw logs.
            if surfaces is not None:
                out.set("name", f"qwen-v1:{index}:{len(seen)}:{lemma.findtext('orth')}")
            if out.find("synt") is not None:
                out.remove(out.find("synt"))
            synt = ET.SubElement(out, "synt")
            for token in ids:
                ET.SubElement(synt, "tok").text = token_name(token)
            root.append(out)
            counts[kind] += 1
            lengths[len(ids)] += 1
    report = dict(
        policy="case-variants-position-gated-initial-v1",
        base_vocab_size=vocab_size,
        alias_ids=alias_ids,
        counts=dict(counts),
        syntactic_lengths=dict(lengths),
        num_source_lemmata=len(source.findall("lemma")),
        num_output_lemmata=sum(counts.values()),
        punctuation="existing acoustic punctuation only; underscore separator stays neutral",
    )
    return root, report


class QwenTextVariantLexiconJob(Job):
    def __init__(self, lexicon, repo, alias_manifest=None):
        self.lexicon, self.repo, self.alias_manifest = lexicon, repo, alias_manifest
        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_vocab = self.output_path("lm.vocab.txt")
        self.out_report = self.output_path("text_variants.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 2})

    def run(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.repo.get_path(), local_files_only=True)
        with open(os.path.join(self.repo.get_path(), "config.json")) as f:
            config = json.load(f)
        with uopen(self.lexicon, "rt") as f:
            source = ET.parse(f).getroot()
        aliases = None
        if self.alias_manifest is not None:
            with open(self.alias_manifest.get_path()) as f:
                aliases = json.load(f)["alias_ids"]
        root, report = rewrite_variant_lexicon(source, tokenizer, config, aliases)
        assert ET.tostring(source.find("phoneme-inventory")) == ET.tostring(root.find("phoneme-inventory"))
        write_xml(self.out_lexicon.get_path(), root)
        with open(self.out_vocab.get_path(), "w") as f:
            for i in range(config["vocab_size"] + len(report["alias_ids"])):
                f.write(f"{token_name(i)} {i}\n")
        with open(self.out_report.get_path(), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Text variants: {report['counts']}; {len(report['alias_ids'])} initial-only aliases", flush=True)


def oracle_sequences(tokenizer, words):
    """All legal toy-lexicon routes, including the original leading-space start."""
    choices = []
    for position, word in enumerate(words):
        if all(not c.isalnum() for c in word):
            choices.append([encode_surface(tokenizer, word)])
        else:
            choices.append(
                [
                    encode_surface(tokenizer, prefix + surface)
                    for surface in word_spellings(word)
                    for prefix in ((" ", "") if position == 0 else (" ",))
                ]
            )
    return list(dict.fromkeys(tuple(i for piece in pieces for i in piece) for pieces in itertools.product(*choices)))


class ExportQwenTextVariantsJob(Job):
    """Same pretrained scores with a positional mask, checked against HF and KV reuse."""

    def __init__(self, repo, alias_manifest, base_report):
        self.repo, self.alias_manifest, self.base_report = repo, alias_manifest, base_report
        self.out_model = self.output_path("model.onnx")
        self.out_report = self.output_path("hf_comparison.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 48, "time": 4})

    def run(self):
        import torch
        import onnxruntime as ort
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ..model_pipelines.transformer_lm.qwen import (
            Qwen3TextVariantsRasrLm,
            export_qwen_onnx,
            verify_qwen_text_variants,
        )

        with open(self.base_report.get_path()) as f:
            if not json.load(f).get("passed"):
                raise ValueError("Original Qwen export gate failed")
        with open(self.alias_manifest.get_path()) as f:
            aliases = json.load(f)["alias_ids"]
        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(self.repo.get_path(), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.repo.get_path(), local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
        wrapper = Qwen3TextVariantsRasrLm(model, aliases)
        print("Exporting position-gated initial-token aliases", flush=True)
        export_qwen_onnx(wrapper, self.out_model.get_path())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 2
        session = ort.InferenceSession(self.out_model.get_path(), sess_options=opts, providers=["CPUExecutionProvider"])
        cases, sequences = [], []
        with torch.inference_mode():
            for words in SCORE_CASES:
                candidates = []
                for ids in oracle_sequences(tokenizer, words):
                    seq = [model.config.bos_token_id, *ids]
                    scores = -model(torch.tensor([seq]), use_cache=False).logits[0].float().log_softmax(-1)
                    targets = [*ids, model.config.eos_token_id]
                    cost = float(scores[torch.arange(len(seq)), torch.tensor(targets)].double().sum())
                    candidates.append((cost, seq))
                cost, seq = min(candidates)
                cases.append(dict(words=list(words), total_cost=cost, input_ids=seq, num_candidates=len(candidates)))
                sequences.append(seq)
                print(f"Variant oracle {words}: {cost:.5f} ({len(candidates)} candidates)", flush=True)
        error = verify_qwen_text_variants(model, session, sequences, aliases)
        with open(self.out_report.get_path(), "w") as f:
            json.dump(dict(passed=True, max_abs_error=error, cases=cases, num_aliases=len(aliases)), f, indent=2)
