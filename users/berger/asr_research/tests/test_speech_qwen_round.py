"""Accounting, reference separation, candidate reuse and staged dev grids."""

import json
from types import SimpleNamespace

import pytest
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import speech_qwen_round as runtime
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import speech_qwen_round as driver
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.result_collection import describe


def trace(text, am, lm):
    return [SimpleNamespace(lemma=text, am_score=am, lm_score=lm)]


def test_nbest_scaled_costs_deduplication_and_parity():
    first = trace("a", 2, .3)
    rows = runtime.pack_nbest([trace("b", 3, .4), trace("a", 3, .3), first], first, lambda t: t[0].lemma)
    assert [r["text"] for r in rows] == ["a", "b"]
    assert rows[0]["first_pass_cost"] == pytest.approx(2.3)
    assert rows[0]["lm_cost"] == .3
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
    candidates = [dict(text="hi", first_pass_cost=2.), dict(text="Hi", first_pass_cost=3.)]
    rows = runtime.score_speech_candidates(backend, candidates, ("original", "initial-capital-period"))
    assert len(requests) == 3  # shared formatted surface scored once
    assert rows[0]["speech"]["original"]["nll"] == 3
    assert rows[0]["speech"]["initial-capital-period"]["nll"] == 4
    assert runtime.choose_speech_candidate(rows, n=50, scale=.3, word_penalty=0, policy="original")["text"] == "hi"
    # Word reward is outside LM scale; zero LM scale is exactly first-pass ranking.
    rows[1]["text"] = "two words"
    assert runtime.choose_speech_candidate(rows, n=50, scale=0, word_penalty=-2, policy="original")["text"] == "two words"
    assert runtime.choose_speech_candidate(rows, n=1, scale=0, word_penalty=-2, policy="original")["text"] == "hi"


def test_cached_selection_runtime_and_no_reference_dependency(tmp_path, monkeypatch):
    row = dict(id="x", candidates=[dict(text="hello", first_pass_cost=1, speech={"original": dict(nll=2)})])
    candidate_file = tmp_path / "candidates.jsonl"
    candidate_file.write_text(json.dumps(row) + "\n")
    generation, scoring = tmp_path / "gen.json", tmp_path / "score.json"
    generation.write_text(json.dumps(dict(audio_seconds=10, elapsed_seconds=2)))
    scoring.write_text(json.dumps(dict(audio_seconds=10, elapsed_seconds=1)))
    job = runtime.SelectSpeechNbestJob(tk.Path(str(candidate_file)), tk.Path(str(generation)), tk.Path(str(scoring)),
        n=50, scale=.1, word_penalty=0, policy="original", code_version="test")
    job.out_hypotheses, job.out_runtime = tk.Path(str(tmp_path / "hyp.jsonl")), tk.Path(str(tmp_path / "runtime.json"))
    job.run()
    assert not hasattr(job, "references")
    assert json.loads((tmp_path / "hyp.jsonl").read_text()) == dict(id="x", text="hello")
    assert json.loads((tmp_path / "runtime.json").read_text())["rtf"] == .3


@pytest.fixture(scope="module")
def batch():
    return driver.run()


def test_dev_grids_and_unchanged_inference_settings(batch):
    assert len(batch["fusion"]) == 60
    assert len(batch["pruning"]) == 16
    priors, names, weights = set(), set(), set()
    for job, score in batch["fusion"]:
        names.add(score.references.creator)
        assert job.rqmt["gpu_mem"] == 48
        if hasattr(job, "kind"):
            cfg = job.config.creator.config.lib_rasr.search_algorithm
            assert job.kind == "speech" and cfg.max_beam_size == [8]
            assert cfg.pre_llm_max_beam_size == 32 and cfg.score_threshold == [3]
            assert cfg.word_penalty in (0, -.5, -1)
            assert cfg.llm_scale in (.05, .1, .15, .2, .25, .3)
        elif hasattr(job, "prior"):
            priors.add(job.prior.creator)
            weights.add(job.prior_scale)
            cfg = job.rasr_config.creator.config.lib_rasr.search_algorithm
            assert cfg.llm_scale in (.2, .3, .4)
    assert len(names) == 2 and len(priors) == 2
    assert weights == {.05, .1, .2}
    for prior in priors:
        assert prior.max_segments == 512
        assert prior.training_source["split"].startswith("train")
    for job, _ in batch["pruning"]:
        cfg = job.config.creator.config.lib_rasr.search_algorithm
        assert cfg.max_beam_size == [32] and cfg.pre_llm_max_beam_size == 128
        assert cfg.score_threshold[0] in (1.5, 2, 2.5, 3)


def test_shared_candidates_sizes_oracles_and_reference_separation(batch):
    result = batch["rescoring"]
    assert len(result["generation"]) == 4
    assert len(result["scoring"]) == 8
    assert len(result["selections"]) == 104
    assert len(set(job.candidates for job in result["scoring"])) == 4
    assert {job.speech_repo.creator.repo_id for job in result["scoring"]} == {
        "Qwen/Qwen3-ASR-0.6B-hf", "Qwen/Qwen3-ASR-1.7B-hf"}
    for job in result["scoring"] + result["selections"]:
        assert not hasattr(job, "references")
    for gen in result["generation"]:
        if isinstance(gen, runtime.RnntSpeechNbestJob):
            cfg = gen.config.creator.config.lib_rasr.search_algorithm
            assert gen.kind == "zero" and gen.auxiliary_repo is None
            assert cfg.recombination_mode == "viterbi" and cfg.llm_scale == 0
            assert isinstance(gen.gate.creator, runtime.RnntSpeechNbestGateJob)
    assert {job.n for job in result["oracles"]} == {10, 50}


def test_dev_boundary_and_staged_beam_tuning():
    for func in (driver.run_fusion, driver.run_pruning, driver.run_rescoring):
        with pytest.raises(ValueError, match="dev-other/dev.short"):
            func({"librispeech-test-other": None})
    with pytest.raises(ValueError, match="selected development"):
        driver.run_pruning(stage="beam")
    rows = driver.run_pruning(stage="beam", scale=.1, threshold=2, word_penalty=-.5)
    assert len(rows) == 8
    assert {job.config.creator.config.lib_rasr.search_algorithm.max_beam_size[0] for job, _ in rows} == {4, 8, 16, 32}


@pytest.mark.parametrize("group,condition,expected", [
    ("qwen-speech-1.7b", "two-pass-speech-n50-scale-0.3-word-penalty0-original", ("Granite CTC", "Speech Qwen 1.7B", "2-pass (50-best)", "LS (open)")),
    ("qwen-parakeet-rnnt", "two-pass-speech-n10-scale-0.3-original", ("Parakeet RNN-T", "Speech Qwen 0.6B", "2-pass (10-best)", "None")),
    ("qwen-parakeet-rnnt-speech-1.7b", "oracle-speech-n50", ("Parakeet RNN-T", "Oracle", "Oracle (50-best)", "None")),
])
def test_report_methods_and_model_sizes(group, condition, expected):
    assert describe(dict(group=group, condition=condition)) == expected
