"""Accounting, reference separation, candidate reuse and staged dev grids."""

import json
from types import SimpleNamespace

import pytest
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import speech_qwen_round as runtime


def trace(text, am, lm):
    return [SimpleNamespace(lemma=text, am_score=am, lm_score=lm)]


def test_nbest_scaled_costs_deduplication_and_parity():
    first = trace("a", 2, 0.3)
    rows = runtime.pack_nbest([trace("b", 3, 0.4), trace("a", 3, 0.3), first], first, lambda t: t[0].lemma)
    assert [r["text"] for r in rows] == ["a", "b"]
    assert rows[0]["first_pass_cost"] == pytest.approx(2.3)
    assert rows[0]["lm_cost"] == 0.3
    with pytest.raises(ValueError, match="parity"):
        runtime.pack_nbest([trace("a", 2, 1)], first, lambda t: t[0].lemma)
    with pytest.raises(ValueError, match="parity"):
        runtime.pack_nbest([], first, lambda t: t[0].lemma)


def test_nbest_search_restores_native_class_on_failure():
    class Search:
        def recognize_segment_n_best(self, features, n):
            return [features] * n

        def recognize_segment(self, features):
            return features

    lib = SimpleNamespace(SearchAlgorithm=Search)
    seen = []
    with pytest.raises(RuntimeError):
        with runtime.nbest_search(lib, lambda traces, first: seen.append((traces, first)), 2):
            assert lib.SearchAlgorithm().recognize_segment(42) == 42
            raise RuntimeError("test")
    assert lib.SearchAlgorithm is Search
    assert seen == [([42, 42], 42)]


def test_whole_candidate_scoring_prompt_eos_and_selection():
    class Tokenizer:
        all_special_ids = [999]

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [ord(c) for c in text]

    requests = []

    def score(prefixes, continuations):
        assert all(p == [100, 101] for p in prefixes)
        assert all(c[-1] == 999 and c.count(999) == 1 for c in continuations)
        requests.extend(continuations)
        return [[1.0] * len(c) for c in continuations]

    backend = SimpleNamespace(tokenizer=Tokenizer(), initial=[100, 101], end=[999], score_continuations=score)
    candidates = [dict(text="hi", first_pass_cost=2.0), dict(text="Hi", first_pass_cost=3.0)]
    rows = runtime.score_speech_candidates(backend, candidates, ("original", "initial-capital-period"))
    assert len(requests) == 3  # shared formatted surface scored once
    assert rows[0]["speech"]["original"]["nll"] == 3
    assert rows[0]["speech"]["initial-capital-period"]["nll"] == 4
    assert runtime.choose_speech_candidate(rows, n=50, scale=0.3, word_penalty=0, policy="original")["text"] == "hi"
    # Word reward is outside LM scale; zero LM scale is exactly first-pass ranking.
    rows[1]["text"] = "two words"
    assert (
        runtime.choose_speech_candidate(rows, n=50, scale=0, word_penalty=-2, policy="original")["text"] == "two words"
    )
    assert runtime.choose_speech_candidate(rows, n=1, scale=0, word_penalty=-2, policy="original")["text"] == "hi"


def test_cached_selection_runtime_and_no_reference_dependency(tmp_path, monkeypatch):
    row = dict(id="x", candidates=[dict(text="hello", first_pass_cost=1, speech={"original": dict(nll=2)})])
    candidate_file = tmp_path / "candidates.jsonl"
    candidate_file.write_text(json.dumps(row) + "\n")
    generation, scoring = tmp_path / "gen.json", tmp_path / "score.json"
    generation.write_text(json.dumps(dict(audio_seconds=10, elapsed_seconds=2)))
    scoring.write_text(json.dumps(dict(audio_seconds=10, elapsed_seconds=1)))
    job = runtime.SelectSpeechNbestJob(
        tk.Path(str(candidate_file)),
        tk.Path(str(generation)),
        tk.Path(str(scoring)),
        n=50,
        scale=0.1,
        word_penalty=0,
        policy="original",
        code_version="test",
    )
    job.out_hypotheses, job.out_runtime = tk.Path(str(tmp_path / "hyp.jsonl")), tk.Path(str(tmp_path / "runtime.json"))
    job.run()
    assert not hasattr(job, "references")
    assert json.loads((tmp_path / "hyp.jsonl").read_text()) == dict(id="x", text="hello")
    assert json.loads((tmp_path / "runtime.json").read_text())["rtf"] == 0.3
