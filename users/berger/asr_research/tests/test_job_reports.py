"""Exercise the actual periodic report writer, including missing dependencies."""

import logging

import pytest
from sisyphus import gs, tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import job_reports


def test_periodic_update_before_dependency_exists_then_publishes(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(gs, "OUTPUT_DIR", str(tmp_path / "published"))
    path = tk.Path(str(tmp_path / "not-started" / "index.md"))
    report = job_reports.register_job_report("reports/analysis.md", path, title="Analysis")
    assert path in report.required_full_list
    assert not path.available()
    with caplog.at_level(logging.WARNING):
        report.update_requirements(force=True)
    published = tmp_path / "published/reports/analysis.md"
    assert "Report not available yet" in published.read_text()
    assert "failed tasks" in published.read_text()
    assert not any("Error while updating" in record.message for record in caplog.records)
    source = tmp_path / "not-started/index.md"
    source.parent.mkdir()
    source.write_text("# Completed\n\nResults.\n", encoding="utf-8")
    report.update_requirements(force=True)
    assert published.read_text() == source.read_text()


def test_dangling_symlink_is_pending_and_other_errors_are_not_swallowed(tmp_path):
    source = tmp_path / "index.md"
    source.symlink_to(tmp_path / "missing.md")
    assert "Report not available yet" in job_reports.read_job_report(tk.Path(str(source)), title="Pending")
    source.unlink()
    source.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        job_reports.read_job_report(tk.Path(str(source)), title="Invalid")
