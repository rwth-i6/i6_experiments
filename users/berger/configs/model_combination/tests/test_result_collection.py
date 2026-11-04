"""Live reports remain available without scheduling archived experiments."""

import json

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import result_collection as reports


def export(tmp_path, group, condition, dataset, wer, normalizer="new"):
    directory = tmp_path / "open-asr" / group / condition / dataset
    directory.mkdir(parents=True)
    (directory / "scoring").write_text(json.dumps(dict(wer_percent=wer, normalizer_revision=normalizer)))
    (directory / "runtime").write_text(json.dumps(dict(rtf=0.12)))
    return directory


def capture_tables(monkeypatch):
    rows = []

    def table(headers, values):
        rows.extend(dict(zip(headers, row)) for row in values)
        return "rendered table\n"

    monkeypatch.setattr(reports, "_table", table)
    return rows


def test_report_uses_matching_model_size_and_normalizer_for_delta(tmp_path, monkeypatch):
    export(tmp_path, "qwen-granite", "native", "librispeech-test-other", 4)
    export(tmp_path, "qwen-speech-0.6b", "native", "librispeech-test-other", 1)
    export(tmp_path, "qwen-speech-1.7b", "native", "librispeech-test-other", 3)
    export(tmp_path, "qwen-speech-1.7b", "granite-fusion", "librispeech-test-other", 2.5)
    export(tmp_path, "qwen-speech-1.7b", "granite-fusion-scale-0.1", "librispeech-test-other", 0.1, "old")
    rendered = capture_tables(monkeypatch)
    collection = reports.ResultCollection(tmp_path)
    collection.corpus_report("librispeech")
    combinations = [r for r in rendered if r.get("ASR") == "Granite CTC" and r.get("LM / model", "").endswith("1.7B")]
    correct = next(r for r in combinations if float(r["WER %"]) == 2.5)
    incompatible = next(r for r in combinations if float(r["WER %"]) == 0.1)
    assert float(correct["ΔWER pp"]) == -0.5
    assert incompatible["ΔWER pp"] == "—"
    assert len(collection.rows()) == 5


def test_pending_invalid_excluded_and_subsplit_rows_are_not_pooled(tmp_path, monkeypatch):
    good = export(tmp_path, "qwen-granite", "native", "loquacious-test.yodas", 7)
    bad = export(tmp_path, "qwen-granite", "unreadable", "loquacious-test", 1)
    (bad / "scoring").write_text("{")
    pending = tmp_path / "open-asr/qwen-granite/one-pass/loquacious-test/scoring"
    pending.parent.mkdir(parents=True)
    pending.symlink_to(tmp_path / "not-completed")
    export(tmp_path, "granite-ctc", "tree-open-corpus-voxpopuli-cleaned-ls4gram", "voxpopuli-cleaned", 90)
    collection = reports.ResultCollection(tmp_path)
    rows = collection.rows()
    assert next(r for r in rows if r["condition"] == "unreadable")["status"].startswith("unreadable")
    assert next(r for r in rows if r["condition"] == "one-pass")["wer"] is None
    assert next(r for r in rows if r["dataset"] == "voxpopuli-cleaned")["excluded"]
    rendered = capture_tables(monkeypatch)
    collection.corpus_report("loquacious")
    scored = [r for r in rendered if r.get("WER %") != "—"]
    assert [float(r["WER %"]) for r in scored] == [7.0]
    assert next(r for r in rows if r["dataset"] == "loquacious-test.yodas")["source"] == str(good / "scoring")


def test_report_registration_is_idempotent_and_constructs_no_jobs(tmp_path, monkeypatch):
    from sisyphus import tk

    calls = []
    before = {job._sis_id() for job in tk.sis_graph.jobs()}
    monkeypatch.setattr(reports.tk, "register_report", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = reports.register_result_collection(output_root=tmp_path)
    count = len(calls)
    assert count > 0
    assert reports.register_result_collection(output_root=tmp_path) is result
    assert len(calls) == count
    assert len({args[0] for args, kwargs in calls}) == count
    assert all(callable(args[1]) and kwargs["update_frequency"] > 0 for args, kwargs in calls)
    assert {job._sis_id() for job in tk.sis_graph.jobs()} == before


def test_own_model_rows_read_live_scores_instead_of_stale_summary(tmp_path):
    directory = (
        tmp_path / "librispeech/recognition/ffnn_transducer_bpe__recog_wordwise-ls-word-trafo-lm0.4-ilm0.2/dev-other"
    )
    directory.mkdir(parents=True)
    output = tmp_path / "score-job/output"
    (output / "reports").mkdir(parents=True)
    (output / "wer").write_text("5.1")
    (directory / "scoring_reports").symlink_to(output / "reports")
    stale = tmp_path / "wordwise-lm/dev/summary"
    stale.parent.mkdir(parents=True)
    stale.write_text("outdated result 99.9")
    (row,) = reports.ResultCollection(tmp_path).rows()
    assert row["wer"] == 5.1 and row["condition"].endswith("ilm0.2")
    assert row["group"].startswith("own-")
    assert row["dataset"] == "librispeech-dev-other"
    assert "Trafo" in row["lm"]
