"""Score a VoiceBench response JSONL using VoiceBench's OWN scorers (imported/executed verbatim).

For open/qa subsets: run their `api_judge.py` (unmodified) as a subprocess with the OpenAI client env
pointed at our judge endpoint (served under the `gpt-4o-mini` alias), producing the `result-*.jsonl`
with per-item judge `score`. Then instantiate their `evaluator_mapping[evaluator]` and call `.evaluate()`.
Rule-based subsets skip the judge. The scoring logic is 100% theirs — we only import and invoke it.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def _load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="VoiceBench repo checkout (pinned commit)")
    p.add_argument("--responses", required=True, help="our {model}-{subset}-test-audio.jsonl")
    p.add_argument("--evaluator", required=True)
    p.add_argument("--out_summary", required=True)
    p.add_argument("--run_judge", action="store_true")
    p.add_argument("--judge_base_url", default=None)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    args = p.parse_args()

    sys.path.insert(0, args.repo)

    if args.run_judge:
        # Run their api_judge.py verbatim. It writes 'result-'+src_file in cwd, so stage the responses
        # under a local basename inside the repo and point the OpenAI client at our judge via env.
        env = {**os.environ}
        if args.judge_base_url:
            env["OPENAI_BASE_URL"] = args.judge_base_url
            env.setdefault("OPENAI_API_KEY", "EMPTY")
        local = "vb_resp.jsonl"
        shutil.copy(args.responses, os.path.join(args.repo, local))
        subprocess.run([sys.executable, "api_judge.py", "--src_file", local], cwd=args.repo, env=env, check=True)
        data = _load(os.path.join(args.repo, f"result-{local}"))
    else:
        data = _load(args.responses)

    # Their evaluator, unchanged.
    from src.evaluator import evaluator_mapping

    evaluator = evaluator_mapping[args.evaluator]()
    result = evaluator.evaluate(data)

    summary = {"evaluator": args.evaluator, "n": len(data), "result": result}
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("VB_SCORE", json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
