"""MRQA: the MoshiRAG paper's factual-QA benchmark (arXiv 2604.12928, §5.1), run faithfully on our models.

Sets (exactly the paper's): OpenAudioBench's spoken Llama Questions (300), Web Questions (1000) and TriviaQA (1000)
(``baichuan-inc/OpenAudioBench``), and Kyutai's ``HaluEvalAudio_1000`` (1000; ships the gold ``knowledge`` document).
Scoring: the rules of ``kyutai-labs/moshi-rag``'s own evaluator, ported verbatim in :mod:`.mrqa_upstream` -- per-set
judge prompts, ``valid_answers=str(list)``, only the FIRST LINE of the judge reply is parsed, and empty replies or
unparseable verdicts drop OUT of the denominator. The judge model is ours and configurable (upstream: GPT-4o for the
three OpenAudioBench sets, Gemma 3 27B for HaluEval; the paper's appendix Tables 6-9 use Gemma 3 27B throughout).

Protocol, matching upstream ``run_inference.py`` / ``InferenceJob``:

* the user speaks first, from frame 0 (``lead_in_s=0``);
* the reply ends after MORE than ``max_tail_silence`` consecutive ``<pad>`` text frames once the question has been
  fed. Upstream stops generating there; we capture ``capture_s`` and cut at the same frame, which is the same
  transcript because the input after the question is silence either way (:func:`.mrqa_upstream.tail_stop_length`).
  The paper does not state its N; ours is a knob (default 25 frames = 2 s), applied at scoring time, so changing it
  re-runs no inference.

Columns. ``monologue`` grades the inner-monologue text cut at the stop frame -- upstream's protocol for Moshi-family
models (its ``model_text``); ``asr`` grades Whisper's transcript of the reply audio cut at the same frame (the column
every backend has); ``reference`` grades MoshiRAG's retrieved reference (upstream's "ref." column). Never rank one
column against another. Each reports ``acc_upstream`` (upstream's denominator: judged rows only) and ``acc_strict``
(all rows; empty and unjudgeable count as wrong), with ``n_empty`` and ``n_judge_fail``.

Two reconstructions, documented rather than hidden: the per-set gold-answer field (upstream's sidecar builder is not
released -- see :data:`ANSWER_FIELDS`) and the stop rule's N.

Jobs: :class:`MRQADownload` -> :class:`MRQAData` -> (``SpeechInference`` via ``sharded_knowledge_inference``) ->
:class:`MRQAResponses` -> ``WhisperTranscription`` -> :class:`MRQAScore`; :func:`mrqa_benchmark_py` wires one model.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import time

from sisyphus import Job, Task, tk

from .common import vllm_gpu_mem_gb

OAB_REPO = "baichuan-inc/OpenAudioBench"
OAB_REVISION = "98f8fa0d5fa7e530ec8e6f097b55f10b003d111b"
HALU_REPO = "kyutai/HaluEvalAudio_1000"
HALU_REVISION = "82a88135fbda632d5739ffeacfc794365f10e6bf"

#: Benchmark order = clip-index order in the combined container.
SUBSETS = ("llama_questions", "web_questions", "trivia_qa", "halueval")
OAB_SUBSETS = SUBSETS[:3]

#: Gold-answer field per set, in the form upstream's ``extract_answer_variants`` reads (str, or a dict with
#: ``normalized_aliases``/``normalized_value``). OUR RECONSTRUCTION: moshi-rag's evaluator reads ``answer`` from a
#: per-clip sidecar whose builder is not released. These are the columns OpenAudioBench's own scorers judge against
#: (``tasks/*_audio.py``); TriviaQA's comma-joined alias string is split into the alias list it was joined from.
ANSWER_FIELDS = {
    "llama_questions": "Answer",
    "web_questions": "answers",
    "trivia_qa": ("answer_normalized_value", "answer_normalized_aliases"),
    "halueval": "answer",
}
QUESTION_FIELDS = {
    "llama_questions": "Questions",
    "web_questions": "question",
    "trivia_qa": "question",
    "halueval": "text",
}

#: Mimi: 24 kHz, 1920 samples per 12.5 Hz frame.
MIMI_SR = 24000
MIMI_FRAME = 1920

DEFAULT_JUDGE = "google/gemma-3-27b-it"
#: Path-safe short names of the judges we serve. A judge not listed here is refused, so an output path can
#: never be ambiguous about which model graded it.
JUDGE_SHORT = {
    "google/gemma-3-27b-it": "gemma3_27b",
    "openai/gpt-oss-120b": "gpt_oss_120b",
    "google/gemma-4-31B-it": "gemma4_31b",
}
#: Base of the per-request judge seed (attempt k uses base + k): reproducible, and a retry still re-samples, as an
#: unseeded upstream retry would.
JUDGE_SEED = 1234

DEFAULT_MAX_TAIL_SILENCE = 25
DEFAULT_CAPTURE_S = 20.0

SOURCES = ("monologue", "asr", "reference")

#: Upstream's explicit verdict for the SimpleQA / LlamaQ prompts; anything else falls back to a substring test.
_EXPLICIT_VERDICT = r"[Tt]he score is \[(Correct|Incorrect)\]"


def unwrap_fence(content: str | None) -> str | None:
    """Drop a surrounding Markdown code fence (```` ```json ... ``` ````) from a judge reply.

    OUR ONE DEVIATION from upstream's client, and only for fenced replies: upstream parses the first line of the
    reply, and Gemma 3 27B fences every OpenAudioBench JSON verdict, so its first line is ```` ```json ```` and all
    TriviaQA / Web Questions verdicts became unparseable -> dropped (smoke 2026-10-01: 40/40). The JSON inside the
    fence is one line, so after unwrapping, upstream's first-line rule and parser apply unchanged.
    """
    if not content:
        return content
    s = content.strip()
    if s.startswith("```"):
        lines = s.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return content


class MRQADownload(Job):
    """Snapshot the two HF dataset repos at pinned revisions (login-node mini_task: needs internet)."""

    def __init__(self, *, oab_revision: str = OAB_REVISION, halu_revision: str = HALU_REVISION):
        self.oab_revision = oab_revision
        self.halu_revision = halu_revision
        self.out_dir = self.output_path("data", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from huggingface_hub import snapshot_download

        out = self.out_dir.get()
        snapshot_download(
            repo_id=OAB_REPO,
            repo_type="dataset",
            revision=self.oab_revision,
            local_dir=os.path.join(out, "OpenAudioBench"),
            allow_patterns=[f"eval_datas/{s}/*" for s in OAB_SUBSETS]
            + [f"eval_datas/{s}/audios/*" for s in OAB_SUBSETS],
        )
        snapshot_download(
            repo_id=HALU_REPO,
            repo_type="dataset",
            revision=self.halu_revision,
            local_dir=os.path.join(out, "HaluEvalAudio_1000"),
        )


def _answer_field(subset: str, row: dict):
    f = ANSWER_FIELDS[subset]
    if isinstance(f, tuple):
        value, aliases = row[f[0]], row[f[1]]
        return {"normalized_value": value, "normalized_aliases": [a for a in aliases.split(", ") if a]}
    return row[f]


class MRQAData(Job):
    """The four sets as ONE question-clip container (index order = :data:`SUBSETS`) + an index-aligned reference.

    ``out_clips``: arrow clip store (``clip_store``), audio at its native rate (drivers resample to 24 kHz).
    ``out_reference``: HF dataset with ``index, subset, question, answer, aliases, category`` (what
    ``WhisperTranscription`` reads; ``category`` = set) plus ``answer_json`` (upstream's gold field),
    ``gt_reference_text`` (HaluEval ``knowledge``), ``source_id``, ``question_samples``, ``sampling_rate``.
    ``n_per_subset``: the first N rows of each set (a smoke run); None = the full sets.
    """

    def __init__(self, *, data_dir: tk.Path, n_per_subset: int | None = None):
        self.data_dir = data_dir
        self.n_per_subset = n_per_subset
        self.out_clips = self.output_path("clips", directory=True)
        self.out_reference = self.output_path("reference", directory=True)
        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _rows(self):
        import csv
        import io

        import pyarrow.parquet as pq
        import soundfile as sf

        root = self.data_dir.get()
        for subset in OAB_SUBSETS:
            base = os.path.join(root, "OpenAudioBench", "eval_datas", subset)
            with open(os.path.join(base, f"{subset}.csv"), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if self.n_per_subset is not None:
                rows = rows[: self.n_per_subset]
            for row in rows:
                audio, sr = sf.read(os.path.join(base, "audios", row["audio_filename"]), dtype="float32")
                yield subset, row, audio, sr, row["audio_filename"], ""
        files = sorted(
            os.path.join(dp, n)
            for dp, _, ns in os.walk(os.path.join(root, "HaluEvalAudio_1000", "data"))
            for n in ns
            if n.endswith(".parquet")
        )
        assert files, f"no HaluEvalAudio parquet under {root}"
        k = 0
        for path in files:
            for row in pq.read_table(path).to_pylist():
                if self.n_per_subset is not None and k >= self.n_per_subset:
                    return
                audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                yield "halueval", row, audio, sr, f"halueval_{k}", row.get("knowledge") or ""
                k += 1

    def run(self):
        import numpy as np
        from datasets import Dataset

        from .clip_store import write_clips
        from .mrqa_upstream import extract_answer_variants

        ref = []

        def clips():
            for i, (subset, row, audio, sr, source_id, knowledge) in enumerate(self._rows()):
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                answer = _answer_field(subset, row)
                variants = extract_answer_variants(answer)
                assert variants, (subset, source_id, answer)
                ref.append(
                    {
                        "index": i,
                        "subset": subset,
                        "category": subset,
                        "question": row[QUESTION_FIELDS[subset]],
                        "answer": variants[0] if isinstance(answer, str) else answer["normalized_value"],
                        "aliases": variants,
                        "answer_json": json.dumps(answer),
                        "gt_reference_text": knowledge,
                        "source_id": source_id,
                        "question_samples": int(audio.shape[0]),
                        "sampling_rate": int(sr),
                    }
                )
                yield i, audio, int(sr)

        write_clips(self.out_clips.get(), clips())
        counts = {s: sum(r["subset"] == s for r in ref) for s in SUBSETS}
        print(f"[mrqa] {len(ref)} questions: {counts}", flush=True)
        if self.n_per_subset is None:
            assert counts == {"llama_questions": 300, "web_questions": 1000, "trivia_qa": 1000, "halueval": 1000}, (
                counts
            )
        Dataset.from_list(ref).save_to_disk(self.out_reference.get())


def question_end_step(question_samples: int, sampling_rate: int, lead_in_s: float) -> int:
    """Index of the last model step that hears the question (upstream: the last ``feed_step`` of the input)."""
    n24 = int(round(question_samples * MIMI_SR / sampling_rate))
    return (int(round(lead_in_s * MIMI_SR)) + n24) // MIMI_FRAME - 1


class MRQAResponses(Job):
    """Apply upstream's stop rule to each reply; emit the graded texts and the cut reply audio.

    ``out_responses`` (jsonl, one row per clip, index order): the kept ``model_text`` pieces and their plain text
    (``monologue``), the RAG ``reference_text`` and retrieval steps when the engine recorded them,
    ``question_end_step``, ``stop_len`` (None = the capture ended before the stop rule fired), ``n_steps``.
    ``out_clips``: the reply audio cut at the same frame -- what Whisper transcribes for the ``asr`` column.
    A clip without a trace (an engine run without ``--text_trace``) is an error, not an empty answer.
    """

    def __init__(
        self,
        *,
        inference_dir: tk.Path,
        reference: tk.Path,
        lead_in_s: float = 0.0,
        max_tail_silence: int = DEFAULT_MAX_TAIL_SILENCE,
    ):
        self.inference_dir = inference_dir
        self.reference = reference
        self.lead_in_s = lead_in_s
        self.max_tail_silence = max_tail_silence
        self.out_responses = self.output_path("responses.jsonl")
        self.out_clips = self.output_path("clips", directory=True)
        self.rqmt = {"cpu": 2, "mem": 12, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np
        from datasets import load_from_disk

        from .clip_store import open_clips, write_clips
        from .mrqa_upstream import tail_stop_length, tokens_to_plain_text

        ref = load_from_disk(self.reference.get())
        clips = open_clips(self.inference_dir.get())
        have = set(clips)  # iterating keys reads no audio
        missing = [r["index"] for r in ref if r["index"] not in have]
        assert not missing, f"{len(missing)} clips have no reply (first: {missing[:5]})"
        rows = []

        def cut():
            for r in ref:
                i = r["index"]
                raw = clips.sidecar(i, "trace")
                assert raw, f"clip {i} has no text trace -- was the engine run with --text_trace?"
                trace = json.loads(raw)
                model_text = trace["model_text"]
                q_end = question_end_step(r["question_samples"], r["sampling_rate"], self.lead_in_s)
                stop_len = tail_stop_length(model_text, q_end, self.max_tail_silence)
                kept = model_text if stop_len is None else model_text[:stop_len]
                audio, sr = clips[i]
                audio = np.asarray(audio, dtype=np.float32)
                if stop_len is not None:
                    audio = audio[: int(round(stop_len * MIMI_FRAME * sr / MIMI_SR))]
                rows.append(
                    {
                        "index": i,
                        "subset": r["subset"],
                        "question": r["question"],
                        "answer_json": r["answer_json"],
                        "model_text": kept,
                        "monologue": tokens_to_plain_text(kept),
                        "reference_text": trace.get("reference_text") or "",
                        "rag_trigger_step": trace.get("rag_trigger_step", -1),
                        "retrieval_step": trace.get("retrieval_step", -1),
                        "retrieval_latency_s": trace.get("retrieval_latency_s", -1.0),
                        "question_end_step": q_end,
                        "stop_len": stop_len,
                        "n_steps": len(model_text),
                    }
                )
                yield i, audio, int(sr)

        write_clips(self.out_clips.get(), cut())
        with open(self.out_responses.get(), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        capped = sum(r["stop_len"] is None for r in rows)
        print(
            f"[mrqa] {len(rows)} replies; stop rule (> {self.max_tail_silence} pad frames) never fired on {capped} "
            f"({capped / max(1, len(rows)):.1%}) -- those are cut by the capture window instead",
            flush=True,
        )


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summarize(rows: list[dict], sources=SOURCES) -> dict:
    """Per set x column: upstream's accuracy (judged rows only) next to the strict one, with the counts that
    separate them; plus keyword delay, RAG and stop-rule statistics. ``mean`` = the macro average over sets."""
    out: dict = {}
    for subset in SUBSETS + ("mean",):
        sub = [r for r in rows if r["subset"] == subset] if subset != "mean" else None
        if subset != "mean" and not sub:
            continue
        block: dict = {}
        if subset == "mean":
            present = [s for s in SUBSETS if s in out]
            for src in sources:
                vals = [out[s][src] for s in present if src in out[s]]
                if vals:
                    block[src] = {
                        k: round(sum(v[k] for v in vals if v[k] is not None) / len(vals), 4)
                        for k in ("acc_upstream", "acc_strict")
                        if all(v[k] is not None for v in vals)
                    }
            delays = [out[s]["keyword_delay_s"] for s in present if out[s].get("keyword_delay_s") is not None]
            if delays:
                block["keyword_delay_s"] = round(sum(delays) / len(delays), 3)
            out["mean"] = block
            continue
        n = len(sub)
        for src in sources:
            scores = [r["scores"].get(src) for r in sub if src in r["scores"]]
            if not scores:
                continue
            empty = sum(1 for r in sub if r["scores"].get(src) is None and not r["texts"].get(src))
            judged = [s for s in scores if s in (0, 1)]
            k = sum(judged)
            block[src] = {
                "n": n,
                "n_judged": len(judged),
                "n_empty": empty,
                "n_judge_fail": n - len(judged) - empty,
                "acc_upstream": round(k / len(judged), 4) if judged else None,
                "acc_strict": round(k / n, 4),
                "ci95_upstream": _wilson(k, len(judged)),
                "ci95_strict": _wilson(k, n),
                "spoke": round((n - empty) / n, 4),
                "n_verdict_fallback": sum(1 for r in sub if r.get("verdict_fallback", {}).get(src)),
            }
        kd = [r["keyword_delay_steps"] / 12.5 for r in sub if r.get("keyword_delay_steps") is not None]
        block["keyword_delay_s"] = round(sum(kd) / len(kd), 3) if kd else None
        block["n_keyword"] = len(kd)
        rag = [r for r in sub if r["texts"].get("reference")]
        block["rag_percentage"] = round(len(rag) / n, 4)
        lat = sorted(r["retrieval_latency_s"] for r in rag if r.get("retrieval_latency_s", -1) >= 0)
        block["retrieval_latency_s_median"] = lat[len(lat) // 2] if lat else None
        block["stop_rule_not_fired"] = round(sum(r["stop_len"] is None for r in sub) / n, 4)
        out[subset] = block
    return out


class MRQAScore(Job):
    """Grade every reply with upstream's judges, served by a local vLLM ``judge_llm``.

    One server, all requests: the correctness judge of each set (by :data:`.mrqa_upstream.DATASET_JUDGE`) on each
    column in ``sources`` that has text, plus the keyword judge on the monologue (upstream's keyword delay). Every
    raw reply is kept in ``out_eval`` (upstream parses only its first line -- the full text is what shows why a
    verdict was unparseable).
    """

    def __init__(
        self,
        *,
        responses: tk.Path,
        asr: tk.Path | None = None,
        judge_llm: str = DEFAULT_JUDGE,
        sources: tuple = SOURCES,
        concurrency: int = 128,
        unwrap_fence: bool = True,
    ):
        assert judge_llm in JUDGE_SHORT, f"unknown judge {judge_llm!r}; add it to JUDGE_SHORT"
        assert set(sources) <= set(SOURCES), sources
        assert ("asr" in sources) == (asr is not None), "the asr column needs the Whisper transcripts and vice versa"
        self.responses = responses
        self.asr = asr
        self.judge_llm = judge_llm
        self.sources = tuple(sources)
        self.concurrency = concurrency
        # see :func:`unwrap_fence`; False = upstream's client byte for byte
        self.unwrap_fence = unwrap_fence
        self.out_eval = self.output_path("eval_results.jsonl")
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 6, "mem": 24, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(judge_llm)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

        from . import mrqa_upstream as up
        from .common import map_concurrent, vllm_server

        rows = [json.loads(line) for line in open(self.responses.get())]
        asr = {}
        if self.asr is not None:
            for line in open(self.asr.get()):
                t = json.loads(line)
                asr[int(t["index"])] = t["transcription"]
            assert len(asr) == len(rows), (len(asr), len(rows))
        for r in rows:
            r["texts"] = {
                "monologue": r["monologue"],
                "asr": asr.get(r["index"], ""),
                "reference": r["reference_text"],
            }

        with vllm_server(self.judge_llm, max_model_len=4096, gpu_memory_utilization=0.85) as url:
            client = OpenAI(api_key="EMPTY", base_url=url, timeout=600)
            client.models.list()

            def make_generate(log: list):
                attempt = [0]

                def generate(prompt, temperature, max_new_tokens):
                    seed = JUDGE_SEED + attempt[0]
                    attempt[0] += 1
                    # Transport retries are ours (a dropped connection is not a verdict); everything after the
                    # reply arrives is upstream's.
                    for k in range(5):
                        try:
                            resp = client.chat.completions.create(
                                model=self.judge_llm,
                                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                                max_tokens=max_new_tokens,
                                temperature=temperature,
                                reasoning_effort="low",
                                seed=seed,
                            )
                            break
                        except (APIConnectionError, APITimeoutError) as e:
                            err = e
                        except APIStatusError as e:
                            if e.status_code < 500:
                                raise
                            err = e
                        time.sleep(2.0 * 2**k)
                    else:
                        raise err
                    content = resp.choices[0].message.content
                    log.append(content)
                    return up.first_line(unwrap_fence(content) if self.unwrap_fence else content)

                return generate

            def judge_row(r):
                kind = up.DATASET_JUDGE[r["subset"]][0]
                variants = up.extract_answer_variants(json.loads(r["answer_json"]))
                scores, raw, fallback = {}, {}, {}
                for src in self.sources:
                    log: list = []
                    scores[src] = up.run_judge(make_generate(log), kind, r["question"], r["texts"][src], variants)
                    raw[src] = log
                    # A verdict decided by upstream's substring fallback ("incorrect"/"correct" anywhere in the
                    # first line) rather than its explicit "the score is [..]" -- counted, never changed.
                    if kind in ("simple_qa", "llama_questions") and scores[src] in (0, 1) and log:
                        first = up.first_line(unwrap_fence(log[-1]) if self.unwrap_fence else log[-1])
                        fallback[src] = re.search(_EXPLICIT_VERDICT, first) is None
                log = []
                keyword = up.run_judge(make_generate(log), "keyword", r["question"], r["monologue"], variants)
                raw["keyword"] = log
                kd = None
                if keyword:
                    step, _, keyword = up.get_keyword_step_delay(tokens=r["model_text"], keyword=keyword)
                    if keyword and step >= 0 and step - r["question_end_step"] >= 0:
                        kd = step - r["question_end_step"]
                return {
                    "scores": scores,
                    "raw": raw,
                    "verdict_fallback": fallback,
                    "keyword": keyword or "",
                    "keyword_delay_steps": kd,
                }

            t0 = time.time()
            results = map_concurrent(judge_row, rows, concurrency=self.concurrency, progress_path="progress.json")
            print(f"[mrqa] judged {len(rows)} replies in {time.time() - t0:.0f}s", flush=True)

        for r, res in zip(rows, results):
            r.update(res)
        with open(self.out_eval.get(), "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k != "model_text"}) + "\n")
        summary = summarize(rows, self.sources)
        summary["judge"] = {
            "model": self.judge_llm,
            "upstream": f"{up.UPSTREAM_REPO}@{up.UPSTREAM_COMMIT}",
            "temperature": up.JUDGE_TEMPERATURE,
            "seed_base": JUDGE_SEED,
            "unwrap_fence": self.unwrap_fence,
        }
        with open(self.out_summary.get(), "w") as f:
            json.dump(summary, f, indent=2)
        for s in SUBSETS + ("mean",):
            if s in summary:
                print(f"[mrqa] {s}: " + json.dumps({k: v for k, v in summary[s].items() if k in self.sources}))


def mrqa_backend(spec):
    """The same backend, asked for the per-frame monologue trace (``--text_trace``) MRQA grades.

    Only the lib engines that write the trace qualify: MoshiRAG always does; base Moshi and PersonaPlex do with the
    flag. Anything else (pip server, cloud, fork drivers) is refused rather than run and found empty afterwards.
    """
    module = spec.offline_module or ""
    if module == "moshi_family.moshirag.offline_inference":
        return spec
    assert module in ("moshi_family.offline_inference", "moshi_family.personaplex.offline_inference"), (
        f"MRQA needs a per-frame text trace; backend {spec.name!r} ({module or 'no offline module'}) has none"
    )
    return dataclasses.replace(spec, offline_extra_args=tuple(spec.offline_extra_args) + ("--text_trace",))


def mrqa_data(n_per_subset: int | None = None) -> MRQAData:
    data = MRQAData(data_dir=MRQADownload().out_dir, n_per_subset=n_per_subset)
    sfx = "" if n_per_subset is None else f"_n{n_per_subset}"
    tk.register_output(f"mrqa/data{sfx}/reference", data.out_reference)
    return data


def mrqa_benchmark_py(
    tag: str,
    speech_backend,
    *,
    judges: tuple = (DEFAULT_JUDGE,),
    n_per_subset: int | None = None,
    max_tail_silence: int = DEFAULT_MAX_TAIL_SILENCE,
    capture_s: float = DEFAULT_CAPTURE_S,
    batch_size: int | None = None,
    inference_seed: int | None = None,
    num_shards: int | None = None,
    grade_asr: bool = True,
    lora_weights: tk.Path | None = None,
    lora_config: tk.Path | None = None,
    notify: bool = True,
) -> dict:
    """Run one model through MRQA and grade it with each judge in ``judges``. Returns ``{judge: MRQAScore}``.

    Outputs: ``mrqa/<tag>/{moshi_output,responses}`` and ``mrqa/<judge short>/<tag>/{eval_results,summary}`` (``_n<N>``
    on the tag for a smoke run). Kept out of ``benchmark/`` on purpose: it is a different benchmark and grader, and
    ``read_benchmarks.py`` must never put it in one table with the TriviaQA judge.
    """
    from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import moshi_venv, whisper_venv

    from .knowledge_benchmark import KNOWLEDGE_INFERENCE_SEED, WhisperTranscription, sharded_knowledge_inference
    from .result_notify import notify_result

    if inference_seed is None:
        inference_seed = KNOWLEDGE_INFERENCE_SEED
    spec = mrqa_backend(speech_backend)
    data = mrqa_data(n_per_subset)
    rag = spec.retrieval_llm is not None
    if rag:
        batch_size = 1
    name = tag if n_per_subset is None else f"{tag}_n{n_per_subset}"
    if num_shards is None:
        num_shards = 1 if rag or n_per_subset is not None else 2
    inference_venv = spec.inference_venv() if spec.inference_venv else moshi_venv()
    moshi_out = sharded_knowledge_inference(
        num_shards=num_shards,
        venv_python_path=inference_venv,
        **({} if batch_size is None else {"batch_size": batch_size}),
        in_dir=data.out_clips,
        storage="hf",
        seed=inference_seed,
        lora_weights=lora_weights,
        lora_config=lora_config,
        offline_script="moshi_offline_inference.py",
        offline_module=spec.offline_module,
        offline_extra_args=spec.offline_extra_args,
        server=spec.server,
        retrieval_llm=spec.retrieval_llm,
        rqmt_override=spec.rqmt_override,
        lead_in_s=0.0,
        capture_s=capture_s,
    )
    tk.register_output(f"mrqa/{name}/moshi_output", moshi_out)
    resp = MRQAResponses(
        inference_dir=moshi_out, reference=data.out_reference, lead_in_s=0.0, max_tail_silence=max_tail_silence
    )
    tk.register_output(f"mrqa/{name}/responses", resp.out_responses)
    asr = None
    if grade_asr:
        asr = WhisperTranscription(
            venv_python_path=whisper_venv(),
            in_dir=resp.out_clips,
            reference_data=data.out_reference,
            model_size="large-v3-turbo",
        ).out_json
        tk.register_output(f"mrqa/{name}/transcription", asr)
    sources = ("monologue",) + (("asr",) if grade_asr else ()) + (("reference",) if rag else ())
    out = {}
    for judge in judges:
        score = MRQAScore(responses=resp.out_responses, asr=asr, judge_llm=judge, sources=sources)
        short = JUDGE_SHORT[judge]
        tk.register_output(f"mrqa/{short}/{name}/eval_results", score.out_eval)
        tk.register_output(f"mrqa/{short}/{name}/summary", score.out_summary)
        if notify:
            notify_result(f"mrqa_{short}_{name}", {"summary": score.out_summary})
        out[judge] = score
    return out
