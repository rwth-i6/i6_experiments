"""Offline self-distillation data-gen for AudexDuplex ("bolt full-duplex onto turn-based models" frame).

Base Audex-2B answers each question in its OWN words (HF `generate`, in-process — Audex's custom arch is
not vLLM-servable), and we emit a minimal ``[user: question, assistant: audex_answer]`` dialogue in the
exact schema ``make_speech_pipeline`` consumes (a ``dialogue`` column = JSON string of ``{speaker, text}``
turns). Audex is a ``<think>...</think>`` reasoning model, so the concise spoken answer is the text AFTER
``</think>``. Rationale under the reframe: distil the base model's OWN behaviour so the FD graft learns
duplex without fighting the frozen trunk (knowledge preservation, not injection). Runs in the setup .venv
(verified: Audex loads + generates there). GPU (fits c25g's 80 GB at 2B).

Downstream: ``make_speech_pipeline(gen.out_hf, ...)`` (Chatterbox TTS + MoshiAnnotate) -> Mimi-encode ->
train AudexDuplex on the pre-encoded codes.
"""

import glob
import json
import os
import re

from sisyphus import Job, Task, tk


def _resolve_audex_dir() -> str:
    # Match moshi_family.audex.backbone.audex_snapshot_dir: the model lives in the repo SUBFOLDER
    # `checkpoint_folder_full`; snapshot_download resolves it from the HF cache (HF_HUB_OFFLINE in-job).
    from huggingface_hub import snapshot_download

    root = snapshot_download("nvidia/Nemotron-Labs-Audex-2B", allow_patterns=["checkpoint_folder_full/*"])
    return os.path.join(root, "checkpoint_folder_full")


def _extract_answer(decoded: str) -> str:
    """Concise spoken answer from Audex's reasoning output: text after ``</think>``, control tokens
    stripped, whitespace collapsed to one short line."""
    s = decoded.split("</think>")[-1] if "</think>" in decoded else decoded
    for t in ("<|im_end|>", "<|im_start|>", "<s>", "</s>"):
        s = s.replace(t, "")
    s = re.sub(r"[*_`#]+", "", s)  # strip markdown emphasis/formatting (TTS would read the symbols)
    return re.sub(r"\s+", " ", s).strip()


def _triviaqa_qa(example: dict):
    """(question, gold_answer, uid) from a TriviaQA row (answer is a {value, aliases} dict)."""
    ans = example.get("answer", {})
    value = ans.get("value", "") if isinstance(ans, dict) else str(ans)
    uid = example.get("question_id") or example["question"][:64]
    return example["question"], value, str(uid)


class AudexSelfDistillGen(Job):
    """Base Audex answers questions -> minimal Q/A dialogues (self-distillation data)."""

    __sis_hash_exclude__ = {"batch_report": 25}

    def __init__(
        self,
        *,
        dataset_split_path: tk.Path,
        max_examples: int | None = None,
        max_new_tokens: int = 256,
        system_prompt: str = "You are a helpful assistant. Answer the question in one short, natural spoken sentence. Do not explain your reasoning out loud.",
        batch_report: int = 25,
    ):
        self.dataset_split_path = dataset_split_path
        self.max_examples = max_examples
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.batch_report = batch_report
        self.out_hf = self.output_path("dialogue_dataset", directory=True)
        self.out_json = self.output_path("preview.json")
        # 2B fits c25g's 80 GB comfortably; sequential gen over a pilot set is well under 8h.
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import torch
        from datasets import Dataset, load_from_disk
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ds = load_from_disk(self.dataset_split_path.get())
        if self.max_examples is not None:
            ds = ds.select(range(min(self.max_examples, len(ds))))
        n = len(ds)
        print(f"[audex-selfdistill] {n} questions", flush=True)

        snap = _resolve_audex_dir()
        tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(snap, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()

        out = []
        for i, ex in enumerate(ds):
            q, gold, uid = _triviaqa_qa(ex)
            text = tok.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": q},
                ],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,  # direct concise answer (no <think> trace) -> clean spoken targets
            )
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.3,
                    pad_token_id=tok.eos_token_id,
                )
            dec = tok.decode(gen[0, ids.shape[1] :], skip_special_tokens=False)
            ans = _extract_answer(dec)
            dialogue = [
                {"speaker": "user", "text": q},
                {"speaker": "assistant", "text": ans},
            ]
            out.append(
                {
                    "uid": uid,
                    "question": q,
                    "gold": gold,
                    "audex_answer": ans,
                    "dialogue": json.dumps(dialogue),
                    "llm_name": "nvidia/Nemotron-Labs-Audex-2B",
                    "template_name": "selfdistill",
                    "truncated": False,
                }
            )
            if (i + 1) % self.batch_report == 0 or i + 1 == n:
                print(f"[audex-selfdistill] {i + 1}/{n} | last: {ans[:70]!r}", flush=True)

        n_empty = sum(1 for r in out if not r["audex_answer"].strip())
        assert n_empty < max(1, n // 10), f"too many empty answers: {n_empty}/{n}"
        Dataset.from_list(out).save_to_disk(self.out_hf.get())
        with open(self.out_json.get(), "w") as f:
            json.dump(out[:20], f, indent=2)
        print(f"[audex-selfdistill] saved {n} dialogues ({n_empty} empty)", flush=True)
