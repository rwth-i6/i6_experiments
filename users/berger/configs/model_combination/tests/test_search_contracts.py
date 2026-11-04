"""Search semantics and cache identity, independent of the current tuning grids."""

import pytest
from sisyphus import tk
from i6_core.rasr import RasrCommand

from i6_experiments.users.berger.configs.model_combination.experiments import wordwise_lm
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.label_scorer_config import (
    get_no_op_label_scorer_config,
)


@pytest.mark.parametrize("ctc", [False, True])
@pytest.mark.parametrize("kind", [None, "word", "bpe", "qwen-0.6b"])
def test_decoder_topology_and_lm_changes_reach_serialized_config(tmp_path, ctc, kind):
    artifacts = {
        "word": (tk.Path("word.onnx"), tk.Path("word.vocab"), None),
        "bpe": (tk.Path("bpe.onnx"), tk.Path("bpe.vocab"), tk.Path("bpe.codes")),
        "qwen-0.6b": tk.Path("qwen-repo"),
    }
    options = dict(blank=3, ctc=ctc, kind=kind, scale=0.37, beam=19, artifacts=artifacts, threshold=7.25, pre_beam=43)

    def create(**changes):
        return wordwise_lm.recognition_config(
            tk.Path("acoustic-labels"), [get_no_op_label_scorer_config()], **(options | changes)
        ).creator

    job = create()
    path = tmp_path / "rasr.config"
    RasrCommand.write_config(job.config, job.post_config, str(path))
    text = path.read_text()
    search = job.config.lib_rasr.search_algorithm
    assert search._get("blank-label-index") == 3
    assert search._get("collapse-repeated-labels") == ctc
    assert search._get("type") == ("llm-timesync-beam-search" if kind else "lexiconfree-timesync-beam-search")
    assert search._get("log-stepwise-statistics") is False
    assert "log-stepwise-statistics" in text
    for changes in (dict(blank=2), dict(beam=23), dict(threshold=5.5), dict(ctc=not ctc)):
        assert create(**changes)._sis_id() != job._sis_id()
    if kind:
        assert search._get("word-piece-convention") == "continuation-marked"
        assert search._get("llm-scale") == options["scale"]
        assert create(scale=0.73)._sis_id() != job._sis_id()
        assert create(pre_beam=71)._sis_id() != job._sis_id()
        assert str(artifacts[kind][0] if kind in ("word", "bpe") else artifacts[kind]) in text
    if kind == "word":
        assert create(unknown_penalty=2.5)._sis_id() != job._sis_id()


def test_alias_context_restores_after_nested_failure(monkeypatch):
    from sisyphus import gs
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import (
        ExperimentContext,
    )

    monkeypatch.setattr(gs, "ALIAS_AND_OUTPUT_SUBDIR", "parent")
    with pytest.raises(RuntimeError):
        with ExperimentContext("outer"):
            assert gs.ALIAS_AND_OUTPUT_SUBDIR == "parent/outer"
            with ExperimentContext("inner"):
                assert gs.ALIAS_AND_OUTPUT_SUBDIR == "parent/outer/inner"
                raise RuntimeError("driver failed")
    assert gs.ALIAS_AND_OUTPUT_SUBDIR == "parent"
