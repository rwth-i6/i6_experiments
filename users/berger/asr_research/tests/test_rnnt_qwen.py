"""Offline checks for the pinned RNN-T dynamic LLM experiment setup."""

from types import SimpleNamespace

import pytest
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.rnnt_qwen import recognition_config, run
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.rnnt_llm import RnntRawPiecesJob


def test_request_based_llm_adapter():
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.rnnt_llm import (
        register_rnnt_llm_scorer,
        ZeroRnntLlmBackend,
    )

    registered = {}

    class Base:
        def __init__(self, config):
            pass

    librasr = SimpleNamespace(LlmScorer=Base, register_llm_scorer_type=lambda name, cls: registered.update({name: cls}))
    cls = register_rnnt_llm_scorer(librasr, "zero", lambda config: ZeroRnntLlmBackend())
    scorer = cls({})
    assert registered["zero"] is cls
    assert scorer.initial_tokens() == [0]
    assert scorer.tokenize(["hello", " world"]) == [[1], [1]]
    assert scorer.spelling_variants(["hello", "WORLD", "123"]) == [["hello", "Hello"], ["world", "World"], ["123"]]
    requests = [SimpleNamespace(prefix=[0], tokens=[1, 2]), SimpleNamespace(prefix=[0, 1], tokens=[3])]
    assert scorer.score(requests) == [[0, 0], [0]]
    received = []
    scorer.backend.score_continuations = lambda p, c: received.append((p, c)) or [[2, 3], [4]]
    assert scorer.score(requests) == [[2, 3], [4]]
    assert received == [([[0], [0, 1]], [[1, 2], [3]])]
    scorer.backend.tokenize = lambda texts: [[[1], [2]] for _ in texts]
    with pytest.raises(ValueError, match="exactly one"):
        scorer.tokenize(["hello"])


def test_search_config_preserves_rnnt_topology():
    raw = SimpleNamespace(out_vocab=tk.Path("pieces.txt"), out_blank=1024)
    for kind in ("zero", "text", "speech"):
        s = recognition_config(raw, kind=kind, auxiliary_repo=tk.Path("qwen")).lib_rasr.search_algorithm
        assert s.type == "llm-rnnt-timesync-beam-search"
        assert s.blank_label_index == 1024
        assert s.collapse_repeated_labels is False
        assert s.max_labels_per_timeframe == 10
        assert s.recombination_mode == "sum"
        assert s.word_piece_convention == "word-start-marked"
        assert s.word_piece_marker == "▁"
        assert s.max_beam_size == [32]
        assert s.pre_llm_max_beam_size == 128
        assert s.log_stepwise_statistics is False
        assert s.llm_scale == (0 if kind == "zero" else 0.3)
    greedy = recognition_config(raw, kind="zero", beam=1).lib_rasr.search_algorithm
    assert greedy.recombination_mode == "viterbi"
    assert greedy.pre_llm_max_beam_size == 1
    with pytest.raises(ValueError):
        recognition_config(raw, kind="invalid")


def test_granite_text_and_speech_preserve_hypothesized_casing():
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.dynamic_qwen_tuned import (
        TunedQwenScorer,
    )
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.qwen_speech import QwenSpeechScorer

    tokenizer = SimpleNamespace(
        encode=lambda s, **kw: list(s.encode()),
        decode=lambda ids, **kw: bytes(ids).decode(),
        all_special_ids=[],
    )
    text, speech = TunedQwenScorer.__new__(TunedQwenScorer), QwenSpeechScorer.__new__(QwenSpeechScorer)
    for scorer in (text, speech):
        scorer.tokenizer, scorer.formatting = tokenizer, "original"
    inputs = ["HELLO", " WORLD", " Commission's", " Europe"]
    assert text.tokenize(inputs) == speech.tokenize(inputs)
    assert [tokenizer.decode(row[0]) for row in speech.tokenize(inputs)] == [
        "HELLO",
        " WORLD",
        " Commission's",
        " Europe",
    ]
    assert all(len(row) == 1 for row in speech.tokenize(inputs))


def test_raw_inventory_keeps_final_blank_and_model_order(monkeypatch, tmp_path):
    import transformers

    labels = ["▁hello", "▁world", "▁thank", "▁you", "▁commission's", "▁proposal", "▁European", "▁Union", "<blank>"]
    tokenizer = SimpleNamespace(
        convert_ids_to_tokens=lambda ids: [labels[i] for i in ids],
        encode=lambda text, **kw: [labels.index("▁" + word) for word in text.split()],
        decode=lambda ids, **kw: " ".join(labels[i][1:] for i in ids),
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a: SimpleNamespace(vocab_size=9, blank_token_id=8)
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a: tokenizer)
    job = RnntRawPiecesJob(tk.Path("repo"), code_version="test")
    job.out_vocab, job.out_report = tk.Path(str(tmp_path / "pieces.txt")), tk.Path(str(tmp_path / "report.json"))
    job.out_blank = SimpleNamespace(set=lambda value: setattr(job, "blank", value))
    job.run()
    assert job.blank == 8
    assert (tmp_path / "pieces.txt").read_text().splitlines() == labels[:-1] + ["[BLANK]"]


def test_frozen_evaluation_matrix():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.qwen import run as run_qwen
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.qwen_speech import run as run_speech

    granite = run_qwen(loquacious=False, two_pass=False)
    data = granite["data"]
    speech = run_speech(data=data)
    rnnt = run(
        data=data, speech_baselines=speech["scores"]["qwen-speech/native"], tuning_only=False, evaluation_only=True
    )
    scores = {**granite["scores"], **speech["scores"], **rnnt["scores"]}
    assert len(data) == 8
    assert set(scores) == {
        "granite-ctc/native",
        "granite-ctc/one-pass",
        "qwen-speech/native",
        "qwen-speech/granite-fusion",
        "parakeet-rnnt/native",
        "parakeet-rnnt/text-qwen-beam8-thr2",
        "parakeet-rnnt/speech-qwen-fast",
    }
    for condition, rows in scores.items():
        assert set(rows) == set(data)
        if condition.endswith("/native"):
            continue
        for name, summary in rows.items():
            job = summary.creator.hypotheses.creator
            config_path = getattr(job, "config", None) or job.rasr_config
            config = config_path.creator.config.lib_rasr.search_algorithm
            threshold = (
                2.0
                if condition == "parakeet-rnnt/text-qwen-beam8-thr2"
                else 3.0
                if condition == "parakeet-rnnt/speech-qwen-fast"
                else 6.0
            )
            assert config.score_threshold == [threshold]
            assert config.pre_llm_score_threshold == threshold
            fast = condition in {"parakeet-rnnt/speech-qwen-fast", "parakeet-rnnt/text-qwen-beam8-thr2"}
            assert config.max_beam_size == ([8] if fast else [32])
            assert config.pre_llm_max_beam_size == (32 if fast else 128)
            assert config.llm_scale == 0.3
            assert job.dataset == data[name].out_dataset


def test_suite_reuses_baselines_and_gates_new_recognitions():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.qwen import run as run_qwen
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.qwen_speech import run as run_speech

    data = run_qwen()["data"]
    speech = run_speech(data=data)["scores"]["qwen-speech/native"]
    result = run(data=data, speech_baselines=speech, tuning_only=False)
    scores = result["scores"]
    for name, prepared in data.items():
        for condition in ("zero-greedy", "zero-beam32", "text-qwen", "speech-qwen"):
            job = scores[f"parakeet-rnnt/{condition}"][name].creator.hypotheses.creator
            scope = f"open-asr/qwen-parakeet-rnnt/{condition}/{name}"
            assert scope in job._sis_alias_prefixes
            assert "recognition" in job.get_aliases()
            score = scores[f"parakeet-rnnt/{condition}"][name].creator
            assert scope in score._sis_alias_prefixes
            assert "scoring" in score.get_aliases()
            assert scope not in job.get_aliases()  # Never alias the scope directory itself.
            assert job.dataset == prepared.out_dataset
            assert job.gate.creator.__class__.__name__ == "RnntLlmGateJob"
            assert job.binary.creator.source_folder.creator.commit == "d000386c08a9b9213238e3cafdeb4291dc0b1775"
            assert job.repo.creator.repo_id == "nvidia/parakeet-rnnt-1.1b"
            if condition == "speech-qwen":
                assert job.speech_baseline == speech[name]
        if name.startswith("loquacious"):
            assert f"{name}.yodas" in scores["parakeet-rnnt/text-qwen"]
    for kind in ("text", "speech"):
        for threshold in (2.0, 3.0, 4.0):
            rows = scores[f"parakeet-rnnt/{kind}-qwen-thr-{threshold:g}"]
            assert all(name.startswith("loquacious-dev") for name in rows)
            assert "loquacious-dev.short" in rows
            job = rows["loquacious-dev"].creator.hypotheses.creator
            config = job.config.creator.config.lib_rasr.search_algorithm
            assert config.score_threshold == [threshold]
            assert config.pre_llm_score_threshold == threshold
            assert config.max_beam_size == [32]
            assert config.pre_llm_max_beam_size == 128
            assert config.llm_scale == 0.3
        for scale in (0.05, 0.1, 0.5):
            assert all(
                name.startswith("loquacious-dev") for name in scores[f"parakeet-rnnt/{kind}-qwen-scale-{scale:g}"]
            )


def test_default_run_decodes_only_development_subset():
    result = run()
    assert set(result["data"]) == {"loquacious-dev.short"}
    subset = result["data"]["loquacious-dev.short"]
    assert subset.__class__.__name__ == "PrepareBenchmarkSubsetJob"
    assert subset.dataset.creator.split == "dev"
    assert len(result["scores"]) == 24  # native, existing conditions and seven speed pilots
    for rows in result["scores"].values():
        assert set(rows) == {"loquacious-dev.short"}
        job = rows["loquacious-dev.short"].creator.hypotheses.creator
        assert job.dataset == subset.out_dataset
        if getattr(job, "kind", None) == "speech":
            assert job.speech_baseline.creator.hypotheses.creator.dataset == subset.out_dataset
    with pytest.raises(ValueError, match="Tuning mode"):
        run(data={"librispeech-test-other": subset})


def test_speech_speed_sweeps_are_development_only():
    result = run()
    expected = [(f"speech-qwen-thr-{t:g}", 32, 128, t) for t in (1.0, 1.5)]
    expected += [(f"speech-qwen-beam32-pre{pre}-thr3", 32, pre, 3.0) for pre in (64, 32, 16)]
    expected += [(f"speech-qwen-beam{beam}-pre32-thr3", beam, 32, 3.0) for beam in (16, 8)]
    for condition, beam, pre, threshold in expected:
        rows = result["scores"][f"parakeet-rnnt/{condition}"]
        assert set(rows) == {"loquacious-dev.short"}
        job = rows["loquacious-dev.short"].creator.hypotheses.creator
        config = job.config.creator.config.lib_rasr.search_algorithm
        assert config.max_beam_size == [beam]
        assert config.pre_llm_max_beam_size == pre
        assert config.score_threshold == [threshold]
        assert config.pre_llm_score_threshold == threshold
        assert config.llm_scale == 0.3
        assert job.rqmt["gpu"] == 1


def test_subset_selects_audio_before_recognition(tmp_path):
    import gzip
    import json
    from datasets import Dataset
    from i6_experiments.example_setups.seq2seq_rasr_2025.data.benchmark_subset import PrepareBenchmarkSubsetJob

    Dataset.from_dict({"id": ["a", "b", "c"], "text": ["one", "two", "three"]}).save_to_disk(str(tmp_path / "source"))
    references = tmp_path / "refs.jsonl"
    references.write_text("".join(json.dumps(dict(id=seq_id, reference=seq_id)) + "\n" for seq_id in "abc"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(excluded_empty_reference_ids=["empty"])))
    segments = tmp_path / "segments.gz"
    with gzip.open(segments, "wt") as dst:
        dst.write("dev/b/b\ndev/empty/empty\n")
    job = PrepareBenchmarkSubsetJob(
        tk.Path(str(tmp_path / "source")),
        tk.Path(str(references)),
        tk.Path(str(manifest)),
        tk.Path(str(segments)),
        code_version="test",
    )
    job.out_dataset = tk.Path(str(tmp_path / "selected"))
    job.out_references, job.out_manifest = (
        tk.Path(str(tmp_path / "out.jsonl")),
        tk.Path(str(tmp_path / "out-manifest.json")),
    )
    job.run()
    from datasets import load_from_disk

    assert list(load_from_disk(job.out_dataset.get_path())["id"]) == ["b"]
    assert json.loads((tmp_path / "out.jsonl").read_text())["id"] == "b"
