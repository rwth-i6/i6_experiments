"""Guard: MRQA's scorer is upstream moshi-rag's scorer, byte for byte, and its stop rule is upstream's loop.

``speech_llm/mrqa_upstream.py`` is a GENERATED port of ``kyutai-labs/moshi-rag``'s ``moshi/moshi/evaluate`` (see its
docstring). A hand edit there -- a "harmless" prompt tidy-up, a stricter parser -- would silently move every MRQA
number away from the paper's protocol, and nothing downstream could tell. So this re-derives every ported segment
from the pinned upstream source and asserts it appears verbatim, then drives the behaviour that matters for the
denominator (first-line truncation, None on empty answers, -1 on unparseable verdicts, the pad-run stop rule).

Shown able to fail: ``--self-test`` flips one character of a ported prompt in memory and asserts the check fires.

Run from the setup root (needs internet for the pinned clone, or pass ``--upstream DIR`` of an existing checkout):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_mrqa_faithful.py
"""

import argparse
import ast
import importlib.util
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
PORT = SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/mrqa_upstream.py"


def load_port(text: str):
    spec = importlib.util.spec_from_loader("mrqa_upstream_under_test", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(text, str(PORT), "exec"), mod.__dict__)
    return mod


def upstream_checkout(commit: str, given: str | None) -> Path:
    if given:
        return Path(given)
    d = Path(tempfile.mkdtemp(prefix="mrqa_upstream_"))
    subprocess.run(["git", "clone", "-q", "https://github.com/kyutai-labs/moshi-rag", str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", commit], check=True)
    return d


def segments(ev: Path):
    """(label, expected verbatim text) for every upstream piece the port must contain."""

    def tree(rel):
        s = (ev / rel).read_text()
        return s, ast.parse(s)

    out = []
    s, t = tree("utils.py")
    for node in t.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "normalize_text",
            "tokens_to_plain_text",
            "average",
            "get_keyword_step_delay",
        ):
            out.append((f"utils.{node.name}", ast.get_source_segment(s, node)))
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in (
            "SPECIAL_TOKEN_PATTERN",
            "NON_ALNUM_RE",
            "WHITESPACE_RE",
        ):
            out.append((f"utils.{node.targets[0].id}", ast.get_source_segment(s, node)))
    s, t = tree("score.py")
    for node in t.body:
        if isinstance(node, ast.FunctionDef) and node.name == "extract_answer_variants":
            out.append(("score.extract_answer_variants", ast.get_source_segment(s, node)))
    parsers = {
        ("judge/base.py", "KeywordLLMJudge"): "parse_keyword",
        ("judge/simple_qa.py", "SimpleQALLMJudge"): "parse_simple_qa",
        ("judge/open_audio_bench.py", "LLamaQuestionsJudge"): "parse_llama_questions",
        ("judge/open_audio_bench.py", "TriviaQAJudge"): "parse_trivia_qa",
        ("judge/math_qa.py", "MathQALLMJudge"): "parse_math_qa",
    }
    prompts = {}
    for (rel, cls), fn in parsers.items():
        s, t = tree(rel)
        for node in t.body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for b in node.body:
                    if isinstance(b, ast.Assign) and getattr(b.targets[0], "id", "") == "PROMPT_TEMPLATE":
                        prompts[cls] = ast.literal_eval(b.value)
                    if isinstance(b, ast.FunctionDef) and b.name == "_parse_response":
                        seg = textwrap.dedent("    " + ast.get_source_segment(s, b))
                        seg = f"def {fn}(" + seg[len("def _parse_response(self, ") :]
                        if fn == "parse_trivia_qa":
                            seg = seg.replace("eval_js = eval(", "eval_js = ast.literal_eval(")
                        out.append((f"{cls}._parse_response", seg))
    return out, prompts


def check(port_text: str, ev: Path, up) -> list[str]:
    errs = []
    segs, prompts = segments(ev)
    assert len(segs) == 13, [label for label, _ in segs]
    for label, seg in segs:
        if seg not in port_text:
            errs.append(f"{label}: not verbatim in mrqa_upstream.py")
    for cls, const in {
        "KeywordLLMJudge": "KEYWORD_PROMPT",
        "SimpleQALLMJudge": "SIMPLE_QA_PROMPT",
        "LLamaQuestionsJudge": "LLAMA_QUESTIONS_PROMPT",
        "TriviaQAJudge": "TRIVIA_QA_PROMPT",
        "MathQALLMJudge": "MATH_QA_PROMPT",
    }.items():
        if getattr(up, const) != prompts[cls]:
            errs.append(f"{const} != upstream {cls}.PROMPT_TEMPLATE")
    # judge routing and the judge's sampling temperature (evaluate.py / open_audio_bench.py / llm/client.py)
    ev_src = (ev / "evaluate.py").read_text()
    for subset in ("halueval", "trivia_qa", "llama_questions", "web_questions"):
        if f'"{subset}": (' not in ev_src:
            errs.append(f"evaluate.py no longer routes {subset}")
    if "self.llm.temperature = 0.0" not in (ev / "judge/open_audio_bench.py").read_text():
        errs.append("OpenAudioBench judges no longer sample at 0.0")
    client = (ev.parent / "llm/client.py").read_text()
    if 'text_response = text_response.split("\\n")[0]' not in client or "self.temperature = 1.0" not in client:
        errs.append("llm/client.py first-line truncation / default temperature changed")
    base = (ev / "judge/base.py").read_text()
    for needle in (
        "if not answer_text or not question_text:",
        "valid_answers=str(gt_answers) if gt_answers else",
        "self.max_retries = 3",
    ):
        if needle not in base:
            errs.append(f"judge/base.py changed: {needle!r}")
    job = (ev.parent / "inference_utils/inference_job.py").read_text()
    for needle in (
        'if piece == "<pad>":',
        "if consecutive > self.max_tail_silence:",
        'if i < self.trace.get("question_end_step", -1):',
    ):
        if needle not in job:
            errs.append(f"inference_job._check_tail_silence changed: {needle!r}")
    return errs


def behaviour(up) -> list[str]:
    errs = []
    if up.first_line('```json\n{"judgment": "correct"}\n```') != "```json":
        errs.append("first_line must keep only the first line")
    calls = []

    def gen(prompt, temperature, n):
        calls.append((prompt, temperature, n))
        return '{"analysis": "x", "judgment": "correct"}'

    if up.run_judge(gen, "trivia_qa", "q?", "", ["a"]) is not None or calls:
        errs.append("an empty answer must be None without a judge call")
    if (
        up.run_judge(gen, "trivia_qa", "q?", "a", ["a", "b"]) != 1
        or calls[-1][1] != 0.0
        or "['a', 'b']" not in calls[-1][0]
    ):
        errs.append("trivia_qa verdict / temperature / valid_answers repr")
    if up.run_judge(lambda p, t, n: "```json", "trivia_qa", "q?", "a", ["a"]) is not None:
        errs.append("an unparseable TriviaQA verdict must exhaust its retries -> None")
    if up.run_judge(lambda p, t, n: "no verdict here", "simple_qa", "q?", "a", ["a"]) != -1:
        errs.append("an unparseable SimpleQA verdict must be -1")
    pad = "<pad>"
    mt = ["x", pad, "▁A", pad, pad, "▁B", pad, pad, pad, "▁C"]
    if up.tail_stop_length(mt, 2, 2) != 9 or up.tail_stop_length(mt, 2, 3) is not None:
        errs.append("tail_stop_length disagrees with _check_tail_silence")
    if up.tokens_to_plain_text(["<pad>", "▁The", "<ret>", "▁Angel", "▁Falls"]) != "The Angel Falls":
        errs.append("tokens_to_plain_text")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=None, help="existing moshi-rag checkout at the pinned commit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    text = PORT.read_text()
    up = load_port(text)
    ev = upstream_checkout(up.UPSTREAM_COMMIT, args.upstream) / "moshi/moshi/evaluate"
    errs = check(text, ev, up) + behaviour(up)
    if args.self_test:
        bad = text.replace("You are a professional QA", "You are a professionnal QA", 1)
        assert bad != text
        caught = check(bad, ev, load_port(bad))
        assert caught, "self-test: a mutated prompt was NOT caught"
        print(f"self-test ok: mutated prompt caught ({caught[0]})")
    for e in errs:
        print("FAIL:", e)
    if errs:
        sys.exit(1)
    print(
        f"ok: mrqa_upstream.py matches {up.UPSTREAM_REPO}@{up.UPSTREAM_COMMIT[:10]} (13 segments, 5 prompts) + behaviour"
    )


if __name__ == "__main__":
    main()
