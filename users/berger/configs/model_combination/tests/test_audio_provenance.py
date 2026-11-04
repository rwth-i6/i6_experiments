"""Pre-decode subset selection and post-decode filtering have the same audio."""

import pytest

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.audio_provenance import (
    canonical_audio_origin,
    audio_origins,
    check_audio_origins,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.error_complementarity import check_pair


def subset(tmp_path, parent, name="PrepareBenchmarkSubsetJob.subset"):
    job = tmp_path / name
    job.mkdir()
    (job / "info").write_text(f"PARAMETER: dataset: {parent}\n")
    return job / "output/dataset"


def test_subset_parent_and_mixed_runtime_origins_are_equivalent(tmp_path):
    parent = tmp_path / "PrepareLoquaciousBenchmarkJob.parent/output/dataset"
    short = subset(tmp_path, parent)
    mixed = dict(reference_sha256="same", dataset_paths=[str(parent), str(short)])
    preselected = dict(reference_sha256="same", dataset_paths=[str(short)])
    assert audio_origins(mixed) == audio_origins(preselected) == (str(parent.resolve()),)
    check_pair(mixed, preselected, {"a": {}}, {"a": {}})
    # Equivalence of origins does not imply equivalence of selected sequences.
    with pytest.raises(ValueError, match="coverage"):
        check_pair(mixed, preselected, {"a": {}}, {"b": {}})
    with pytest.raises(ValueError, match="raw references"):
        check_pair(mixed, dict(preselected, reference_sha256="different"), {}, {})


def test_nested_selection_and_path_aliases(tmp_path):
    parent = tmp_path / "parent/output/dataset"
    first = subset(tmp_path, parent, "PrepareBenchmarkSubsetJob.first")
    second = subset(tmp_path, first, "PrepareBenchmarkSubsetJob.second")
    alias = tmp_path / "alias"
    alias.symlink_to(second)
    assert canonical_audio_origin(str(alias)) == str(parent.resolve())


def test_unrelated_origins_and_nonselection_transforms_remain_distinct(tmp_path):
    parent = tmp_path / "parent/output/dataset"
    short = subset(tmp_path, parent)
    other = subset(tmp_path, tmp_path / "other/output/dataset", "PrepareBenchmarkSubsetJob.other")
    with pytest.raises(ValueError, match="different prepared audio"):
        check_audio_origins(dict(dataset_paths=[str(short)]), dict(dataset_paths=[str(other)]))
    # Sharing one origin is insufficient when another unrelated origin exists.
    with pytest.raises(ValueError, match="different prepared audio"):
        check_audio_origins(dict(dataset_paths=[str(short), str(other)]), dict(dataset_paths=[str(parent)]))
    transformed = subset(tmp_path, parent, "NormalizeAudioJob.changed")
    assert canonical_audio_origin(str(transformed)) == str(transformed.resolve())
    with pytest.raises(ValueError, match="different prepared audio"):
        check_audio_origins(dict(dataset_paths=[str(parent)]), dict(dataset_paths=[str(transformed)]))


def test_missing_metadata_and_cycles_fail_closed(tmp_path):
    missing = tmp_path / "PrepareBenchmarkSubsetJob.missing/output/dataset"
    with pytest.raises(ValueError, match="Cannot verify"):
        canonical_audio_origin(str(missing))
    cycle = tmp_path / "PrepareBenchmarkSubsetJob.cycle/output/dataset"
    subset(tmp_path, cycle, "PrepareBenchmarkSubsetJob.cycle")
    with pytest.raises(ValueError, match="Cycle"):
        canonical_audio_origin(str(cycle))
