"""Portability gates, auditable fingerprint reuse and actual CTC API comparison."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
from i6_experiments.users.berger.configs.model_combination import bootstrap
from i6_experiments.users.berger.model_pipelines.common.ctc_api_parity import compare_runs
from i6_experiments.users.berger.model_pipelines.common.segment_resume import SegmentResumeState


def test_config_only_import_needs_no_cmake_and_creates_no_builds():
    script = """
from sisyphus import tk
import i6_core.tools.compile as compile
if hasattr(compile, 'CMakeJob'): del compile.CMakeJob
before={j._sis_id() for j in tk.sis_graph.jobs()}
from i6_experiments.users.berger.configs.model_combination.experiments.open_asr import recognition_config
from i6_experiments.users.berger.configs.model_combination import tools
assert {j._sis_id() for j in tk.sis_graph.jobs()} == before
from types import SimpleNamespace
inventory=SimpleNamespace(out_vocab=tk.Path('labels'),out_blank=0)
cfg=recognition_config('ctc',inventory)
assert cfg.lib_rasr.label_scorer._get('type')=='no-op'
from i6_experiments.users.berger.configs.model_combination.experiments.open_asr.rnnt_qwen import recognition_config as rnnt_config
rnnt_config(inventory,kind='zero')
assert {j._sis_id() for j in tk.sis_graph.jobs()} == before
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy(), timeout=30)


def test_exact_reviewed_fingerprints_and_future_edit_invalidation(tmp_path, monkeypatch):
    manifest = json.loads((bootstrap.CONFIGS / "runtime_fingerprint_compat.json").read_text())["files"]
    for name, entry in manifest.items():
        assert hashlib.sha256(bootstrap.source_bytes(bootstrap.RESEARCH / name)).hexdigest() == entry["original_sha256"]
    path = tmp_path / "model_pipelines/common/rnnt_llm.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        (bootstrap.RESEARCH / "model_pipelines/common/rnnt_llm.py").read_bytes() + b"\n# subsequent edit\n"
    )
    monkeypatch.setattr(bootstrap, "RESEARCH", tmp_path)
    assert bootstrap.source_bytes(path) == path.read_bytes()


def test_accepted_transformers_versions_are_explicit(monkeypatch):
    import transformers
    from i6_experiments.users.berger.model_pipelines.common.hf_benchmark import check_transformers_version

    for version in ["5.16.1", "5.19.0"]:
        monkeypatch.setattr(transformers, "__version__", version)
        check_transformers_version()
    monkeypatch.setattr(transformers, "__version__", "0.0.0")
    with pytest.raises(RuntimeError):
        check_transformers_version()


def test_frozen_prior_and_selected_stages_are_self_contained():
    root = bootstrap.RESEARCH / "data/assets/granite_ctc"
    meta = json.loads((root / "librispeech_train_clean_100_prior.metadata.json").read_text())
    p = root / "librispeech_train_clean_100_prior.json"
    d = json.loads(p.read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["sha256"]
    assert len(d["log_prior"]) == len(d["pieces"]) == meta["num_labels"]
    assert d["blank"] == meta["blank"] and len(d["sample_ids"]) == meta["sample_count"]
    assert d["source"]["split"].startswith("train")
    manifest = json.loads((bootstrap.CONFIGS / "experiments/open_asr/frozen_public_conditions.json").read_text())
    condition = manifest["conditions"]["Granite CTC | None | standalone"]["development"]["librispeech-dev-other"]
    text = condition["stages"][0]["files"]["rasr_config"]["public_text"]
    assert "tree-timesync-beam-search" in text and "unknown-word-fallback" in text
    from i6_experiments.users.berger.configs.model_combination.experiments.open_asr.ctc_api_parity import (
        request_api_config,
    )

    fusion = manifest["conditions"]["Granite CTC | Speech Qwen 1.7B | one-pass"]["development"]["librispeech-dev-other"]
    cfg = request_api_config(fusion["stages"][0]["files"]["rasr_config"]["public_text"])
    assert cfg.lib_rasr.search_algorithm._get("word-piece-marker") == "▁"
    assert cfg.lib_rasr.search_algorithm._get("word-start-marker") is None


def test_ctc_gate_runs_in_own_process(tmp_path, rasr_bpe_lib_dir):
    script = """
import sys
from sisyphus import tk
from i6_experiments.users.berger.model_pipelines.common.ctc_llm import CtcLlmGateJob
j=CtcLlmGateJob(tk.Path(sys.argv[1]),code_version='test')
j.out_report=tk.Path(sys.argv[2]);j.out_log=tk.Path(sys.argv[3])
j.output_path=lambda name: tk.Path(sys.argv[4]+'/'+name)
j.run()
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(rasr_bpe_lib_dir),
            str(tmp_path / "report"),
            str(tmp_path / "log"),
            str(tmp_path),
        ],
        check=True,
        timeout=30,
    )
    assert json.loads((tmp_path / "report").read_text())["passed"]


def test_parity_does_not_accept_matching_rounded_wer_or_missing_ids(tmp_path):
    row = dict(id="x", text="hello", token_ids=[1], traceback=[dict(am_score=2.0, lm_score=3.0)])
    score = dict(ins=0, del_=0, sub=0, total=0, ref_len=1, wer_percent=0, normalizer_revision="same")
    score["del"] = score.pop("del_")
    a, b, s = tmp_path / "old", tmp_path / "new", tmp_path / "score"
    a.write_text(json.dumps(row) + "\n")
    b.write_text(a.read_text())
    s.write_text(json.dumps(score))
    report, _ = compare_runs(a, b, s, s)
    assert report["exact_parity"]
    row["token_ids"] = [2]
    b.write_text(json.dumps(row) + "\n")
    report, _ = compare_runs(a, b, s, s)
    assert not report["exact_parity"] and report["token_differences"] == 1
    row["traceback"][0]["am_score"] = float("nan")
    b.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_runs(a, b, s, s)
    row["traceback"][0]["am_score"] = 2.0
    row["id"] = "different"
    b.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="segment IDs"):
        compare_runs(a, b, s, s)


def test_resume_prefix_timings_torn_write_and_identity(tmp_path):
    p = tmp_path / "hypotheses.jsonl"
    ids = ["a", "b", "c"]
    identity = dict(config="pinned")
    state = SegmentResumeState(p, ids, identity)
    assert state.start_index == 0
    first = dict(id="a", elapsed_seconds=2.0, _resume_audio_seconds=4.0, _resume_validation=dict(passed=True))
    p.write_text(json.dumps(first) + "\n" + '{"id":"b"')
    state = SegmentResumeState(p, ids, identity)
    assert state.start_index == 1 and state.elapsed_seconds == 2 and state.audio_seconds == 4
    assert state.validation["passed"] and p.read_text() == json.dumps(first) + "\n"
    with p.open("a") as f:
        for i in range(state.start_index, len(ids)):
            f.write(json.dumps(dict(id=ids[i], elapsed_seconds=1.0, _resume_audio_seconds=3.0)) + "\n")
    done = SegmentResumeState(p, ids, identity)
    assert done.start_index == 3 and done.elapsed_seconds == 4 and done.audio_seconds == 10
    with pytest.raises(ValueError, match="identity"):
        SegmentResumeState(p, ids, dict(config="changed"))
    with pytest.raises(ValueError, match="identity"):
        SegmentResumeState(p, list(reversed(ids)), identity)


def test_resume_rejects_unidentified_or_corrupt_completed_output(tmp_path):
    p = tmp_path / "hyp"
    p.write_text("{}\n")
    with pytest.raises(ValueError, match="no resume identity"):
        SegmentResumeState(p, ["a"], {})
    p.unlink()
    SegmentResumeState(p, ["a"], {})
    p.write_text(json.dumps(dict(id="wrong", elapsed_seconds=1, _resume_audio_seconds=2)) + "\n")
    with pytest.raises(ValueError, match="ordered"):
        SegmentResumeState(p, ["a"], {})


def test_rnnt_job_resumes_completed_segments_without_reencoding(tmp_path, monkeypatch):
    import numpy as np
    import torch
    import datasets
    from types import SimpleNamespace
    from sisyphus import tk
    from i6_experiments.users.berger.model_pipelines.common import hf_benchmark as hf, rnnt_llm

    class Data:
        _fingerprint = "same-dataset"

        def __len__(self):
            return 3

        def __iter__(self):
            return (self[i] for i in range(3))

        def __getitem__(self, key):
            if key == "id":
                return ["a", "b", "c"]
            return dict(id=["a", "b", "c"][key], audio=dict(array=np.full(160, key + 1), sampling_rate=16000))

    monkeypatch.setattr(datasets, "load_from_disk", lambda path: Data())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(hf, "register_hf_scorer", lambda *args: None)
    monkeypatch.setattr(hf, "traceback_result", lambda *args: dict(text="hello", token_ids=[1]))
    calls = []
    fail = [True]

    class Adapter:
        def __init__(self, *args):
            pass

        def inputs(self, wave):
            return wave

        def encode_inputs(self, wave):
            value = int(wave[0])
            calls.append(value)
            if value == 2 and fail[0]:
                raise RuntimeError("simulated interruption")
            return np.zeros((2, 1), dtype=np.float32)

    monkeypatch.setattr(hf, "HfBenchmarkAdapter", Adapter)
    cfg = SimpleNamespace(set_from_file=lambda path: True, set=lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "librasr",
        SimpleNamespace(
            Configuration=lambda: cfg,
            init_logging=lambda cfg: None,
            register_llm_scorer_type=lambda *args: None,
            LlmScorer=object,
            SearchAlgorithm=lambda **kwargs: SimpleNamespace(recognize_segment=lambda features: []),
        ),
    )
    gate = tmp_path / "gate"
    gate.write_text('{"passed":true}')
    config, vocab = tmp_path / "config", tmp_path / "vocab"
    config.write_text("same-config")
    vocab.write_text("[BLANK]\nhello\n")
    kwargs = dict(
        dataset=tk.Path("dataset"),
        repo=tk.Path("repo"),
        auxiliary_repo=None,
        vocabulary=tk.Path(str(vocab)),
        binary=tk.Path("binary"),
        config=tk.Path(str(config)),
        gate=tk.Path(str(gate)),
        kind="zero",
        code_version="test",
    )
    job = rnnt_llm.RnntQwenRecogJob(**kwargs, resume=True)
    legacy = rnnt_llm.RnntQwenRecogJob(**kwargs)
    assert job._sis_id() != legacy._sis_id()
    assert job.tasks().__next__().resumeable()
    assert not legacy.tasks().__next__().resumeable()
    for name in ["out_hypotheses", "out_runtime", "out_validation", "out_log"]:
        setattr(job, name, tk.Path(str(tmp_path / name)))
    with pytest.raises(RuntimeError, match="interruption"):
        job.run()
    fail[0] = False
    job.run()
    assert calls == [1, 2, 2, 3]
    rows = [json.loads(line) for line in (tmp_path / "out_hypotheses").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    runtime = json.loads((tmp_path / "out_runtime").read_text())
    assert runtime["num_segments"] == 3 and runtime["audio_seconds"] == pytest.approx(0.03)
    assert runtime["elapsed_seconds"] == pytest.approx(sum(r["elapsed_seconds"] for r in rows))
