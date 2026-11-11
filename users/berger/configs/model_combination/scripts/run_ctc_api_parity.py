"""Run the public parity check independently of the live Sisyphus manager.

Each native stage is a fresh process. Outputs stay in a separate directory;
completed public jobs and the live checkout are not edited.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "gate", "recognition", "compare"], default="all")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        for stage in ["gate", "recognition", "compare"]:
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(output), "--stage", stage],
                check=True,
            )
        print((output / "comparison.md").read_text())
        return
    from sisyphus import tk
    from i6_core.rasr import RasrCommand
    from i6_experiments.users.berger.configs.model_combination import bootstrap
    from i6_experiments.users.berger.configs.model_combination.experiments.open_asr.ctc_api_parity import run

    manifest = json.loads((bootstrap.CONFIGS / "experiments/open_asr/frozen_public_conditions.json").read_text())
    jobs = run()["librispeech-dev-other"]
    job, score, comparison = jobs["recognition"], jobs["scoring"], jobs["comparison"]
    binary = manifest["conditions"]["Parakeet RNN-T | Speech Qwen 1.7B | one-pass"]["development"][
        "librispeech-dev-other"
    ]["stages"][0]["parameters"]["binary"]
    if not Path(binary).is_dir():
        raise FileNotFoundError("Pinned public d000386c binary unavailable")
    gate = job.gate.creator
    gate.binary = tk.Path(binary)
    gate.out_report = tk.Path(str(output / "gate.json"))
    gate.out_log = tk.Path(str(output / "gate.log"))
    gate.output_path = lambda name, **kwargs: tk.Path(str(output / name))
    if args.stage == "gate":
        gate.run()
        return
    config = job.config.creator
    RasrCommand.write_config(config.config, config.post_config, str(output / "recog.config"))
    job.config = tk.Path(str(output / "recog.config"))
    job.binary = tk.Path(binary)
    job.gate = gate.out_report
    for attr, name in [
        ("out_hypotheses", "hypotheses.jsonl"),
        ("out_runtime", "runtime.json"),
        ("out_validation", "validation.json"),
        ("out_log", "rasr.log"),
        ("out_correction_validation", "correction_validation.json"),
    ]:
        setattr(job, attr, tk.Path(str(output / name)))
    if args.stage == "recognition":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GPU parity requires CUDA; no CPU fallback")
        job.run()
        return
    score.hypotheses = job.out_hypotheses
    score.out_summary = tk.Path(str(output / "score.json"))
    score.out_segments = tk.Path(str(output / "segments.jsonl"))
    score.out_wer = SimpleNamespace(set=lambda value: None)
    score.run()
    comparison.new = job.out_hypotheses
    comparison.new_score = score.out_summary
    comparison.out_report = tk.Path(str(output / "comparison.json"))
    comparison.out_differences = tk.Path(str(output / "differences.jsonl"))
    comparison.out_markdown = tk.Path(str(output / "comparison.md"))
    comparison.run()


if __name__ == "__main__":
    main()
