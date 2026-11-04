"""Dev/test separation, immutable evidence, paired clustering and missing cells."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import controlled_comparison as cc


def record(model, condition, dataset, wer, *, lm="None", rtf=.1, group="test", status="complete"):
    method = "Native" if lm == "None" else "1-pass fusion"
    r = dict(model=model, lm=lm, condition=condition, dataset=dataset, group=group, decoding=method,
             lexicon="None", status=status, normalizer=cc.NORMALIZER, reference_sha256="same",
             summary=dict(wer_percent=wer), timing=dict(rtf=rtf))
    r["family"] = cc.family(r)
    return r


def development():
    rows = []
    for ds, am in zip(cc.DEV, [2, 4]):
        rows += [record("Granite CTC", "native", ds, am),
                 record("Speech Qwen 1.7B", "native", ds, am * 1.2, group="speech"),
                 record("Granite CTC", "good-dev", ds, am * .9, lm="Speech Qwen 1.7B"),
                 record("Granite CTC", "slow-better", ds, am * .8, lm="Speech Qwen 1.7B", rtf=3)]
    return dict(partition="dev", records=rows)


def test_selection_is_joint_dev_only_and_keeps_speed_frontiers():
    snapshot = development()
    p = cc.select_protocol(snapshot)
    family = "Granite CTC | Speech Qwen 1.7B | one-pass"
    assert p["selected"][family + " | fast"]["condition"] == "good-dev"
    assert p["selected"][family + " | unrestricted"]["condition"] == "slow-better"
    assert p["selected"][family + " | fast"]["objective"] == pytest.approx(.9)
    snapshot["records"].append(record("Granite CTC", "tempting-test", "librispeech-test-other", .1))
    with pytest.raises(ValueError, match="only see development"):
        cc.select_protocol(snapshot)


def test_selection_does_not_hide_pending_or_missing_timing():
    snapshot = development()
    for ds in cc.DEV:
        snapshot["records"].append(record("Granite CTC", "pending", ds, 0, lm="Speech Qwen 1.7B", status="pending"))
        snapshot["records"].append(record("Granite CTC", "no-timing", ds, .1, lm="Speech Qwen 1.7B", rtf=None))
    p = cc.select_protocol(snapshot)
    assert p["ready"] is False
    assert "good-dev" in [r["condition"] for r in p["selected"].values()]
    assert "no-timing" not in [r["condition"] for r in p["selected"].values()]


def test_no_test_inputs_on_selection_job():
    from sisyphus import tk
    job = cc.SelectControlledConfigsJob(tk.Path("dev.json"), code_version="test")
    assert not hasattr(job, "evaluation_snapshot")
    with pytest.raises(ValueError, match="separate inputs"):
        cc.SnapshotControlledResultsJob([record("Granite CTC", "native", "librispeech-test-other", 1)],
                                        partition="dev", code_version="test")


def test_paired_bootstrap_identity_ratios_and_best_aggregate_not_oracle():
    # Complementary standalone errors; the oracle would be zero but the
    # better aggregate standalone is 10%. The identical fusion gains zero.
    counts = np.array([[[20, 100], [0, 100], [20, 100]],
                       [[0, 100], [20, 100], [0, 100]]], dtype=float)
    result = cc.paired_bootstrap(counts, [[], [], [0, 1]], samples=500, seed=5)
    assert result[2]["delta"] == 0
    assert result[2]["baselines"][0]["delta_ci"] == [0, 0]
    assert result == cc.paired_bootstrap(counts, [[], [], [0, 1]], samples=500, seed=5)
    assert result[2]["delta_ci"][0] >= 0
    counts[:, 2, 1] *= 2  # System-specific compound-equivalent reference lengths.
    assert cc.paired_bootstrap(counts, [[], [], [0, 1]], samples=500)[2]["wer"] == 5


@pytest.mark.parametrize("dataset,seq_id,cluster", [
    ("librispeech-test-other", "1089-134686-0000", "chapter:1089-134686"),
    ("ami-cleaned", "AMI_EN2002a_H00_MEE073_0000096_0000665", "meeting:EN2002a"),
    ("gigaspeech-cleaned", "POD1000000005_S0000000", "recording:POD1000000005"),
    ("spgispeech", "abc/118.wav", "recording:abc"),
    ("voxpopuli-cleaned", "20090202-0900-PLENARY-9-en_20090202-17:20:18_2", "session:20090202-0900-PLENARY-9-en"),
    ("loquacious-test", "-0BVrzfoOmA-00003-00002016-00003032.wav", "recording:-0BVrzfoOmA"),
    ("earnings22-cleaned", "4468919_trimmed", "scoring-unit:4468919_trimmed"),
])
def test_cluster_policy(dataset, seq_id, cluster):
    assert cc.cluster_id(dataset, seq_id) == cluster


def paired_records(tmp_path):
    rows = []
    for model, condition, lm, errors in [("Granite CTC", "native", "None", [2, 0]),
                                        ("Speech Qwen 1.7B", "native", "None", [0, 2]),
                                        ("Granite CTC", "fusion", "Speech Qwen 1.7B", [1, 0])]:
        path = tmp_path / (model + condition + ".jsonl")
        segments = [dict(id=f"1-1-{i}", counts=dict(ins=0, del_=0, sub=e, total=e, ref_len=10)) for i, e in enumerate(errors)]
        for s in segments:
            s["counts"]["del"] = s["counts"].pop("del_")
        path.write_text("".join(json.dumps(s) + "\n" for s in segments))
        r = record(model, condition, "librispeech-test-other", 0, lm=lm)
        r.update(segments=str(path), score="summary.json", dataset_paths=["same-audio"], stages=[])
        r["summary"] = {key: sum(s["counts"][key] for s in segments) for key in segments[0]["counts"]}
        rows.append(r)
    return rows


def test_compare_guards_provenance_and_coverage_and_few_clusters(tmp_path):
    rows = paired_records(tmp_path)
    r = cc.compare_dataset("librispeech-test-other", rows, samples=200)
    assert r["n_clusters"] == 1 and r["uncertainty_warning"]
    assert r["rows"][2]["delta"] == -5
    assert r["rows"][2]["missing_baseline"] is False
    bad = copy.deepcopy(rows)
    bad[1]["reference_sha256"] = "different"
    with pytest.raises(ValueError, match="different raw references"):
        cc.compare_dataset("librispeech-test-other", bad, samples=200)
    bad = copy.deepcopy(rows)
    bad[1]["dataset_paths"] = ["other-audio"]
    with pytest.raises(ValueError, match="different prepared audio"):
        cc.compare_dataset("librispeech-test-other", bad, samples=200)
    bad = cc.compare_dataset("librispeech-test-other", [rows[0], rows[2]], samples=200)
    assert bad["rows"][1]["missing_baseline"]
    assert "delta" not in bad["rows"][1]


def test_info_parser_never_evaluates_code_and_extracts_paths(tmp_path):
    (tmp_path / "info").write_text("PARAMETER: beam: 4\nPARAMETER: kwargs: {'repo': <Path /a/b>, 'zero': False}\n"
                                  "PARAMETER: malicious: __import__('os').getcwd()\n")
    params = cc.parameters_flat(tmp_path)
    assert params["repo"] == "/a/b" and params["zero"] is False and params["beam"] == 4
    assert params["malicious"].startswith("__import__")


def test_snapshot_rejects_changes_and_copies_complete_evidence(tmp_path):
    from sisyphus import tk
    score = tmp_path / "score.json"
    score.write_text(json.dumps(dict(wer_percent=1)))
    r = record("Granite CTC", "native", cc.DEV[0], 1)
    r.update(score=str(score), score_sha256=cc.sha256(score), runtime=None)
    job = cc.SnapshotControlledResultsJob([r], partition="dev", code_version="test")
    job.out_snapshot = tk.Path(str(tmp_path / "snapshot.json"))
    job.run()
    assert cc.read_json(job.out_snapshot.get_path())["records"][0]["summary"]["wer_percent"] == 1
    score.write_text(json.dumps(dict(wer_percent=2)))
    with pytest.raises(ValueError, match="Evidence changed"):
        job.run()


def test_selected_evaluation_graph_reuses_controls_and_replays_only_missing(tmp_path, monkeypatch):
    from sisyphus import tk
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import controlled_evaluation as ev
    p = cc.select_protocol(development())
    # Restrict to a controlled standalone/auxiliary/fusion toy graph.
    p["selected"] = {k: v for k, v in p["selected"].items() if k.endswith(" | fast")}
    key = "Granite CTC | Speech Qwen 1.7B | one-pass | fast"
    stage = dict(job_type="QwenSpeechRecogJob", files={}, parameters=dict(
        dataset="/dev/dataset", speech_repo="/models/speech", granite_repo="/models/granite", binary="/rasr",
        vocabulary="/vocab", rasr_config="/config", baseline="/dev/baseline", code_version="fixed", batch_size=8))
    for ds in cc.DEV:
        p["selected"][key]["dev"][ds]["stages"] = [stage]
    existing = []
    for model, group in [("Granite CTC", "test"), ("Speech Qwen 1.7B", "speech")]:
        r = record(model, "native", "librispeech-test-other", 2, group=group)
        r.update(score="/completed/summary", segments="/completed/segments", hypotheses="/completed/hyp",
                 runtime="/completed/runtime", references="/completed/ref")
        existing.append(r)
    monkeypatch.setattr(ev.cc, "scan_exports", lambda *args: existing)
    file = tmp_path / "selection.json"
    file.write_text(json.dumps(p))
    prepared = SimpleNamespace(out_dataset=tk.Path("/eval/dataset"), out_references=tk.Path("/completed/ref"))
    result = ev.run(file, datasets={"librispeech-test-other": prepared}, output_root=tmp_path, alias_root=tmp_path)
    assert len(result["jobs"]) == 1 and len(result["entries"]) == 3
    job = result["jobs"][0]
    assert job.dataset == prepared.out_dataset
    assert job.baseline.get_path() == "/completed/summary"
    assert job.rqmt["gpu_mem"] == 48 and job.rasr_config.get_path() == "/config"
    assert result["report"].operating_points == ("fast",)
    # The graph pins the resolved file; a moving output alias is not an input.
    assert result["protocol"] == str(file.resolve())
    p["ready"] = False
    file.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="provisional"):
        ev.run(file, datasets={"librispeech-test-other": prepared}, output_root=tmp_path, alias_root=tmp_path)


def test_unknown_conditions_and_oracles_are_not_candidate_runs():
    r = record("Granite CTC", "speech-nbest-scores", cc.DEV[0], 1, lm="Speech Qwen 1.7B")
    assert cc.family(r) is None


def test_inference_replay_is_allowlisted_and_checks_config_hash(tmp_path):
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.controlled_evaluation import instantiate
    with pytest.raises(ValueError, match="No audited"):
        instantiate(dict(job_type="Unknown", parameters={}, files={}))
    config = tmp_path / "config"
    config.write_text("old")
    stage = dict(job_type="QwenSpeechRecogJob", parameters={}, files={"config":dict(path=str(config),sha256=cc.sha256(config))})
    config.write_text("changed")
    with pytest.raises(ValueError, match="Frozen configuration"):
        instantiate(stage)


def test_comparison_job_exports_counts_intervals_and_exact_missing_cells(tmp_path):
    from sisyphus import tk
    rows = paired_records(tmp_path)
    choices = {}
    for row in rows:
        choices[row["family"] + " | fast"] = dict(group=row["group"], condition=row["condition"],
            family=row["family"], max_dev_rtf=.1,
            dev={ds:dict(summary=dict(wer_percent=1)) for ds in cc.DEV})
    # Explicitly absent configuration must remain missing, not be replaced by
    # the completed fusion row on the same model pair.
    choices["missing | fast"] = dict(group="test", condition="not-run", family="missing", max_dev_rtf=.1,
                                     dev={ds:dict(summary=dict(wer_percent=1)) for ds in cc.DEV})
    protocol = dict(selected=choices, ready=True, objective="test", max_rtf=1.5)
    selection, evaluation = tmp_path / "selection.json", tmp_path / "eval.json"
    selection.write_text(json.dumps(protocol))
    evaluation.write_text(json.dumps(dict(partition="evaluation", records=rows)))
    job = cc.ControlledComparisonJob(tk.Path(str(selection)), tk.Path(str(evaluation)),
                                    bootstrap_samples=100, operating_points=("fast",), code_version="test")
    for name in ("report", "results", "tsv", "missing"):
        setattr(job, "out_" + name, tk.Path(str(tmp_path / name)))
    job.run()
    assert "Few independent clusters" in (tmp_path / "report").read_text()
    assert "not-run" in (tmp_path / "report").read_text()
    assert "not-run" not in (tmp_path / "tsv").read_text()
    missing = json.loads((tmp_path / "missing").read_text())
    assert len(missing) == 1 and missing[0]["condition"] == "not-run"
    assert "fast/librispeech-test-other" in json.loads((tmp_path / "results").read_text())["comparisons"]
