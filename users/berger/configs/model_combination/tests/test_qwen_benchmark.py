import json
import gzip

import pytest

from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_benchmark import (
    SelectFixedQwenNbestJob,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob
from i6_experiments.example_setups.seq2seq_rasr_2025.data.loquacious_benchmark import LoquaciousSubsetJob


def test_fixed_selection_and_grouped_scoring(tmp_path):
    rows = []
    for key, text in [("a", "hello"), ("b", "world")]:
        rows.append(
            dict(
                id=key,
                candidates=[
                    dict(
                        text=text,
                        am_cost=0,
                        lm_cost=0,
                        first_pass_cost=0,
                        qwen={"initial-capital-period": dict(nll=1, token_costs=[0.5, 0.5])},
                    )
                ],
            )
        )
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("".join(json.dumps(r) + "\n" for r in rows))
    selected = SelectFixedQwenNbestJob(tk.Path(str(candidates)), code_version="test")
    selected.out_hypotheses = tk.Path(str(tmp_path / "hypotheses.jsonl"))
    selected.run()
    refs = tmp_path / "refs.jsonl"
    refs.write_text(
        "".join(
            json.dumps(dict(id=k, group_id="session", chunk_index=i, reference="hello world")) + "\n"
            for i, k in enumerate(["a", "b"])
        )
    )
    score = ScoreBenchmarkJob(tk.Path(str(refs)), selected.out_hypotheses, code_version="test")
    score.out_summary = tk.Path(str(tmp_path / "summary.json"))
    score.out_segments = tk.Path(str(tmp_path / "segments.jsonl"))
    score.out_wer = tk.Variable(str(tmp_path / "wer"))
    score.run()
    assert json.loads((tmp_path / "summary.json").read_text())["wer_percent"] == 0


def test_loquacious_subset_uses_ids_without_decoding_audio(tmp_path):
    refs = tmp_path / "refs"
    hyp = tmp_path / "hyp"
    refs.write_text(
        "".join(
            json.dumps(dict(id=k, reference="hi", group_id=k, chunk_index=0)) + "\n"
            for k in ["common_voice_001", "123-4-5"]
        )
    )
    hyp.write_text("".join(json.dumps(dict(id=k, text="hi")) + "\n" for k in ["common_voice_001", "123-4-5"]))
    job = LoquaciousSubsetJob(tk.Path(str(refs)), tk.Path(str(hyp)), "commonvoice")
    job.out_references = tk.Path(str(tmp_path / "subset_refs"))
    job.out_hypotheses = tk.Path(str(tmp_path / "subset_hyp"))
    job.run()
    assert json.loads((tmp_path / "subset_hyp").read_text())["id"] == "common_voice_001"


@pytest.mark.parametrize("compressed", [False, True])
def test_loquacious_short_subset_reads_plain_or_gzip_segments(tmp_path, compressed):
    refs = tmp_path / "refs.jsonl"
    hyp = tmp_path / "hyp.jsonl"
    refs.write_text("".join(json.dumps(dict(id=k, reference="hi")) + "\n" for k in ["first", "second"]))
    hyp.write_text("".join(json.dumps(dict(id=k, text="hi")) + "\n" for k in ["first", "second"]))
    segments = tmp_path / ("segments.gz" if compressed else "segments")
    opener = gzip.open if compressed else open
    with opener(segments, "wt", encoding="utf-8") as dst:
        dst.write("loquacious/second/second\n\n")
    job = LoquaciousSubsetJob(tk.Path(str(refs)), tk.Path(str(hyp)), "short", segments=tk.Path(str(segments)))
    job.out_references = tk.Path(str(tmp_path / "subset_refs"))
    job.out_hypotheses = tk.Path(str(tmp_path / "subset_hyp"))
    job.run()
    assert json.loads((tmp_path / "subset_refs").read_text())["id"] == "second"
    assert json.loads((tmp_path / "subset_hyp").read_text())["id"] == "second"
