"""Offline contract tests; no checkpoint/dataset downloads and no GPU required."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from i6_experiments.example_setups.seq2seq_rasr_2025.data.open_asr import reference_rows, DATASETS
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.benchmark_scoring import scoring_pairs
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.hf_benchmark import (
    audio_array,
    collapse_ctc,
    HfBenchmarkAdapter,
    traceback_result,
)


def test_reference_identity_and_empty_filter():
    rows = reference_rows([dict(id="x", text="hello"), dict(id="empty", text="[noise]"), dict(id="y", text="world")])
    assert [r["source_index"] for r in rows] == [0, 2]
    assert [r["group_id"] for r in rows] == ["x", "y"]
    with pytest.raises(ValueError, match="duplicate"):
        reference_rows([dict(id="x", text="hello"), dict(id="x", text="world")])


def test_chunk_order_and_utterance_isolation():
    refs = reference_rows(
        [
            dict(id="a1", parent_id="call", chunk_index=1, text="hello world"),
            dict(id="a0", parent_id="call", chunk_index=0, text="hello world"),
        ]
    )
    pairs = list(scoring_pairs(refs, [dict(id="a1", text="world"), dict(id="a0", text="hello")]))
    assert pairs == [dict(id="call", reference="hello world", hypothesis="hello world")]
    refs = reference_rows([dict(id="a", text="hello"), dict(id="b", text="world")])
    assert len(list(scoring_pairs(refs, [dict(id="a", text="hello"), dict(id="b", text="world")]))) == 2
    with pytest.raises(ValueError, match="coverage"):
        list(scoring_pairs(refs, [dict(id="a", text="hello")]))
    with pytest.raises(ValueError, match="Missing chunk"):
        reference_rows(
            [
                dict(id="a", parent_id="call", chunk_index=0, text="hello"),
                dict(id="b", parent_id="call", chunk_index=2, text="hello"),
            ]
        )


def test_audio_and_ctc_collapse():
    assert collapse_ctc([0, 1, 1, 0, 1, 2, 2, 0], 0) == [1, 1, 2]
    expected = np.array([0.25, -0.25], dtype=np.float32)
    np.testing.assert_array_equal(audio_array(dict(array=expected, sampling_rate=16000)), expected)
    decoder = SimpleNamespace(
        get_all_samples=lambda: SimpleNamespace(data=torch.from_numpy(expected[None]), sample_rate=16000)
    )
    np.testing.assert_array_equal(audio_array(decoder), expected)
    with pytest.raises(ValueError):
        audio_array(dict(array=expected, sampling_rate=8000))


@pytest.mark.parametrize("family", ["aed", "rnnt"])
def test_adapter_matches_full_model(family):
    from transformers import (
        CohereAsrConfig,
        CohereAsrForConditionalGeneration,
        ParakeetEncoderConfig,
        ParakeetRNNTConfig,
        ParakeetForRNNT,
    )

    torch.manual_seed(7)
    enc = ParakeetEncoderConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        subsampling_conv_channels=8,
        num_mel_bins=8,
    )
    adapter = HfBenchmarkAdapter.__new__(HfBenchmarkAdapter)
    adapter.family, adapter.device, adapter.max_new_tokens, adapter.blank = family, "cpu", 20, 10
    if family == "aed":
        adapter.model = CohereAsrForConditionalGeneration(
            CohereAsrConfig(
                encoder_config=enc,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                vocab_size=11,
                decoder_start_token_id=4,
            )
        ).eval()
    else:
        adapter.model = ParakeetForRNNT(
            ParakeetRNNTConfig(
                encoder_config=enc, vocab_size=11, blank_token_id=10, decoder_hidden_size=16, num_decoder_layers=1
            )
        ).eval()
    inputs = dict(input_features=torch.randn(1, 64, 8), attention_mask=torch.ones(1, 64, dtype=torch.long))
    with torch.inference_mode():
        if family == "aed":
            inputs["decoder_input_ids"] = torch.tensor([[5, 6]])
            prepared = adapter.prepare_aed_policy(inputs)
            adapter.prompt = prepared.pop("decoder_input_ids")[0].tolist()
            assert adapter.prompt == [4, 5, 6]  # native generate prepends BOS
            adapter.encoder = adapter.model.get_encoder()(**prepared)
        else:
            adapter.encoder = adapter.model.get_audio_features(**inputs).pooler_output
        adapter.num_frames = 8
        contexts = [(0, ()), (1, (7,)), (2, (8,)), (3, (7, 7))]
        actual = adapter.score_contexts(contexts)
        for (t, prefix), (costs, _) in zip(contexts, actual):
            if family == "aed":
                args = dict(prepared, decoder_input_ids=torch.tensor([adapter.prompt + list(prefix)]), use_cache=False)
                logits = adapter.model(**args).logits[0, -1]
            else:
                output = adapter.model(
                    **inputs, decoder_input_ids=torch.tensor([[10] + list(prefix)]), use_decoder_cache=False
                ).logits
                logits = output[0, t, -1] if output.ndim == 4 else output[0, t]
            np.testing.assert_allclose(costs, -logits.float().log_softmax(-1).numpy(), atol=2e-6)


def test_traceback_preserves_repeated_labels_and_whole_words():
    def item(word):
        return SimpleNamespace(lemma=word, am_score=0, lm_score=0, start_time=0, end_time=1)

    adapter = SimpleNamespace(blank=0, eos=3, detokenize=lambda ids: str(ids), postprocess=lambda x: x)
    result = traceback_result([item("a"), item("a"), item("<blank>")], adapter, {"a": 1, "<blank>": 0}, True)
    assert result["token_ids"] == [1, 1]
    result = traceback_result([item("[BLANK]"), item("qwen-v1:2:3:HELLO"), item("WORLD")], adapter, {}, False)
    assert result["text"] == "HELLO WORLD"
    with pytest.raises(ValueError, match="missing from the acoustic vocabulary"):
        traceback_result([item("unmapped")], adapter, {"a": 1}, True)


@pytest.mark.parametrize("family", ["ctc", "aed"])
def test_inventory_restores_aed_special_ids(tmp_path, monkeypatch, family):
    from sisyphus import tk
    from i6_core.lib.lexicon import Lexicon, Lemma
    from i6_core.util import write_xml
    import transformers
    from i6_experiments.example_setups.seq2seq_rasr_2025.data import benchmark_lexicon as module

    config = SimpleNamespace(vocab_size=4, blank_token_id=0, eos_token_id=3)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda _: config)
    tokenizer = SimpleNamespace(all_special_ids=[0, 3], convert_ids_to_tokens=lambda _: "<unk>")
    monkeypatch.setattr(module, "_load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(module, "_get_labels", lambda *a, **kw: ["<blank>", "a", "b", "<eos>"])
    source = Lexicon()
    for phone in ("<blank>", "a", "b", "<eos>"):
        source.add_phoneme(phone, variation="none")
    source.add_lemma(Lemma(orth=["AB"], phon=["a b"]))
    source.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], synt=[], special="blank"))
    source.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=[""], synt=["<UNK>"], special="unknown"))
    source.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], phon=[""], synt=["<s>"], special="sentence-begin"))
    source.add_lemma(Lemma(orth=["[SENTENCE-END]"], phon=[""], synt=["</s>"], special="sentence-end"))
    source_file = tmp_path / "source.xml"
    write_xml(str(source_file), source.to_xml())
    job = module.BenchmarkInventoryJob(
        tk.Path(str(tmp_path)), family, code_version="test", lexicon=tk.Path(str(source_file))
    )
    for name in ("out_vocab", "out_lexicon", "out_metadata"):
        setattr(job, name, tk.Path(str(tmp_path / name)))
    job.out_eos = job.out_blank = SimpleNamespace(set=lambda _: None)
    job.run()
    lex = Lexicon()
    lex.load(job.out_lexicon.get_path())
    assert list(lex.phonemes) == (["<unk>"] if family == "aed" else ["<blank>"]) + ["a", "b", "<eos>"]
    assert next(lemma for lemma in lex.lemmata if lemma.special == "unknown").phon == []
    if family == "aed":
        assert not any(lemma.special == "blank" for lemma in lex.lemmata)
        assert next(lemma for lemma in lex.lemmata if lemma.special == "sentence-end").phon == ["<eos>"]
        assert next(lemma for lemma in lex.lemmata if lemma.special == "sentence-begin").phon == []


def test_matrix_and_pins():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import run_all, CONDITIONS, RNNT_COMMIT

    results = run_all()
    assert len(DATASETS) == 8
    assert sum(map(len, results.values())) == 118
    for model in ("granite-ctc", "cohere-aed"):
        for condition in CONDITIONS:
            assert len(results[f"{model}/{condition}"]) == 8
    assert {k for k in results if k.startswith("parakeet-rnnt/")} == {
        "parakeet-rnnt/native",
        "parakeet-rnnt/rasr-lexfree",
    }
    for key, rows in results.items():
        for score_path in rows.values():
            recog = score_path.creator.hypotheses.creator
            assert recog.dtype == "float32"
            assert recog.repo.creator.revision
            if recog.native:
                continue
            cfg = recog.rasr_config.creator.config.lib_rasr
            if key.startswith("parakeet-rnnt"):
                assert recog.rasr_binary.creator.source_folder.creator.commit == RNNT_COMMIT
            if "/tree" in key:
                assert cfg.acoustic_model.state_tying.type == "monophone"
                assert cfg.search_algorithm.unknown_word_fallback == "disabled"
            if key.endswith(("tree-trafo", "tree-qwen")):
                assert cfg.lm.max_batch_size == 1


def test_matrix_without_neural_lms_keeps_remaining_jobs(monkeypatch):
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments import open_asr
    from i6_experiments.example_setups.seq2seq_rasr_2025.data import oov_neural_lm

    full = open_asr.run_all()

    def unexpected_neural_setup(*args, **kwargs):
        raise AssertionError("Disabled neural LMs must not register exports or validation jobs")

    monkeypatch.setattr(open_asr, "_qwen", unexpected_neural_setup)
    monkeypatch.setattr(oov_neural_lm, "get_gpu_word_transformer_params", unexpected_neural_setup)
    reduced = open_asr.run_all(neural_lms=False)
    expected = {key: rows for key, rows in full.items() if not key.endswith(("/tree-trafo", "/tree-qwen"))}
    assert reduced == expected
    assert sum(map(len, reduced.values())) == 86

    active = open_asr.run_all(neural_lms=False, aed_tree=False)
    assert active == {
        key: rows
        for key, rows in reduced.items()
        if not key.startswith("cohere-aed/") or key.endswith(("/native", "/rasr-lexfree"))
    }
    assert sum(map(len, active.values())) == 67


def test_granite_open_tree_matrix_and_matched_controls():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import run_all

    old = run_all(neural_lms=False, aed_tree=False)
    expanded = run_all(neural_lms=False, aed_tree=False, granite_open_tree=True)
    assert {key: expanded[key] for key in old} == old
    assert sum(map(len, expanded.values())) == 99
    for lm in (False, True):
        suffix = "-4gram" if lm else ""
        opened = expanded[f"granite-ctc/tree-open{suffix}"]
        controls = expanded[f"granite-ctc/tree-open-control{suffix}"]
        assert len(opened) == len(controls) == 8
        for ds in opened:
            a = opened[ds].creator.hypotheses.creator.rasr_config.creator.config.lib_rasr
            b = controls[ds].creator.hypotheses.creator.rasr_config.creator.config.lib_rasr
            assert a.lexicon.file == b.lexicon.file
            validation = a.lexicon.file.creator
            assert validation.pruning_report is not None
            assert validation.decoder_smoke_report is not None
            assert a.acoustic_model.state_tying.type == "monophone"
            assert a.search_algorithm.unknown_word_fallback == "known-excluding"
            assert b.search_algorithm.unknown_word_fallback == "disabled"
            assert a.search_algorithm.unknown_word_tokenization == "word-start-marked"
            assert a.search_algorithm.unknown_word_penalty == (6 if lm else 1)
            assert a.search_algorithm.unknown_piece_penalty == 2
            assert a.search_algorithm.max_beam_size == b.search_algorithm.max_beam_size == [64 if lm else 256]
            assert a.search_algorithm.score_threshold == b.search_algorithm.score_threshold == [16 if lm else 8]
            assert a.search_algorithm.sentence_end_fall_back is False
            if lm:
                assert a.lm.scale == b.lm.scale == 0.3


def test_granite_fallback_traceback_with_real_tokenizer():
    from pathlib import Path
    from sisyphus import tk
    from i6_experiments.example_setups.seq2seq_rasr_2025.data.huggingface import _load_tokenizer

    repo = Path("../../alias/open-asr/granite-ctc/download/output/repo")
    if not repo.is_dir():
        pytest.skip("Completed Granite tokenizer download is unavailable")
    tokenizer = _load_tokenizer(tk.Path(str(repo.resolve())))
    adapter = SimpleNamespace(family="ctc", tokenizer=tokenizer, postprocess=lambda x: x)

    def item(name):
        return SimpleNamespace(lemma=name, am_score=0, lm_score=0, start_time=0, end_time=1)

    # UTF-8 bytes must be decoded together, and blanks must not split a run.
    tokens = ["▁", "<0xC2>", "<0xA2>", "3", "7", "▁and", "▁i"]
    trace = [item("HELLO")]
    for token in tokens:
        trace.extend([item(token + "&&"), item("[BLANK]")])
    trace += [item("WORLD"), item("[SENTENCE-END]")]
    result = traceback_result(trace, adapter, {}, False)
    assert result["text"] == "HELLO ¢37 and i WORLD"
    assert result["fallback_piece_count"] == 6
    assert result["max_fallback_word_pieces"] == 4
    assert result["traceback"][1]["lemma"] == "▁&&"


def test_tuned_granite_transformer_only_adds_matched_pair(monkeypatch):
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments import open_asr

    def unexpected_qwen(*args, **kwargs):
        raise AssertionError("Tuned Transformer conditions must not enable Qwen")

    monkeypatch.setattr(open_asr, "_qwen", unexpected_qwen)
    base = open_asr.run_all(neural_lms=False, aed_tree=False, granite_open_tree=True)
    expanded = open_asr.run_all(neural_lms=False, aed_tree=False, granite_open_tree=True, granite_open_transformer=True)
    assert {key: expanded[key] for key in base} == base
    assert sum(map(len, expanded.values())) == 115
    for condition in ("tree-open-trafo", "tree-open-control-trafo"):
        rows = expanded[f"granite-ctc/{condition}"]
        assert len(rows) == 8
        for score in rows.values():
            job = score.creator.hypotheses.creator
            cfg = job.rasr_config.creator.config.lib_rasr
            assert cfg.lm.type == "onnx-stateless"
            assert cfg.lm.scale == 0.3
            assert cfg.lm.max_batch_size == 1
            assert cfg.search_algorithm.max_beam_size == [64]
            assert cfg.search_algorithm.max_word_end_beam_size == 16
            assert cfg.search_algorithm.score_threshold == [16]
            assert cfg.search_algorithm.word_end_score_threshold == 16
            if condition == "tree-open-trafo":
                assert cfg.search_algorithm.unknown_word_fallback == "known-excluding"
                assert cfg.search_algorithm.unknown_word_penalty == 4
                assert cfg.search_algorithm.unknown_piece_penalty == 2
            else:
                assert cfg.search_algorithm.unknown_word_fallback == "disabled"
    with pytest.raises(ValueError, match="requires granite_open_tree"):
        open_asr.run_all(granite_open_transformer=True)


def test_corpus_open_tree_resource_selection_and_identity():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import run_all

    settings = dict(neural_lms=False, aed_tree=False, granite_open_tree=True, granite_open_transformer=True)
    base = run_all(**settings)
    expanded = run_all(**settings, corpus_open_tree=True)
    assert {key: expanded[key] for key in base} == base
    assert sum(map(len, expanded.values())) == 121
    for ds, suffixes in (("gigaspeech-cleaned", ("", "-corpus4gram")), ("voxpopuli-cleaned", ("",))):
        for suffix in suffixes:
            opened = expanded[f"granite-ctc/tree-open-corpus-{ds}{suffix}"]
            closed = expanded[f"granite-ctc/tree-open-control-corpus-{ds}{suffix}"]
            assert set(opened) == set(closed) == {ds}
            cfg = opened[ds].creator.hypotheses.creator.rasr_config.creator.config.lib_rasr
            control = closed[ds].creator.hypotheses.creator.rasr_config.creator.config.lib_rasr
            assert cfg.lexicon.file == control.lexicon.file
            assert cfg.search_algorithm.unknown_word_fallback == "known-excluding"
            assert control.search_algorithm.unknown_word_fallback == "disabled"
            if suffix == "-corpus4gram":
                assert cfg.lm.file == cfg.lexicon.file.creator.arpa_lm
                assert cfg.lm.file.get_path().endswith("/4gram.arpa.gz")
    assert not any(key.endswith(("-ls4gram", "-ls-trafo")) for key in expanded)
    with pytest.raises(ValueError, match="corpus_open_tree requires"):
        run_all(corpus_open_tree=True)


def test_resource_overview_includes_corpus_rows(tmp_path):
    from sisyphus import tk
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.benchmark_overview import (
        BenchmarkResourceOverviewJob,
    )

    score = tmp_path / "score.json"
    score.write_text(json.dumps(dict(wer_percent=9.53)))
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(dict(rtf=0.014)))
    keys = ("granite-ctc/tree-open-4gram", "granite-ctc/tree-open-corpus-gigaspeech-cleaned-corpus4gram")
    scores = {key: {"gigaspeech-cleaned": tk.Path(str(score))} for key in keys}
    runtimes = {key: {"gigaspeech-cleaned": tk.Path(str(runtime))} for key in keys}
    resources = {
        keys[0]: dict(lexicon="LibriSpeech", lm="LibriSpeech 4gram"),
        keys[1]: dict(lexicon="gigaspeech-cleaned", lm="gigaspeech-cleaned 4gram"),
    }
    job = BenchmarkResourceOverviewJob(scores, runtimes, resources)
    job.out_report = tk.Path(str(tmp_path / "overview.md"))
    job.run()
    text = (tmp_path / "overview.md").read_text()
    assert "| LibriSpeech | LibriSpeech 4gram | 9.530 |" in text
    assert "| gigaspeech-cleaned | gigaspeech-cleaned 4gram | 9.530 |" in text


def test_score_counts_and_partial_macro(tmp_path):
    from sisyphus import tk
    from importlib.metadata import version
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import benchmark_scoring as scoring

    refs = tmp_path / "refs.jsonl"
    hyps = tmp_path / "hyps.jsonl"
    refs.write_text(json.dumps(dict(id="x", group_id="x", chunk_index=0, reference="a b")) + "\n")
    hyps.write_text(json.dumps(dict(id="x", text="a c")) + "\n")
    job = scoring.ScoreBenchmarkJob(tk.Path(str(refs)), tk.Path(str(hyps)), code_version="test")
    job.out_summary, job.out_segments = tk.Path(str(tmp_path / "summary")), tk.Path(str(tmp_path / "segments"))
    job.out_wer = SimpleNamespace(set=lambda value: None)
    job.run()
    assert json.loads((tmp_path / "summary").read_text())["wer_percent"] == 50
    assert json.loads((tmp_path / "summary").read_text())["kaldialign_version"] == version("kaldialign")
    summary = scoring.BenchmarkSummaryJob(
        {"partial": {"test": job.out_summary}}, ["test", "missing"], code_version="test"
    )
    summary.out_report = tk.Path(str(tmp_path / "report"))
    summary.run()
    assert json.loads((tmp_path / "report").read_text())["partial"]["macro_wer_percent"] is None


def test_score_real_edit_counts(tmp_path):
    from sisyphus import tk
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.benchmark_scoring import (
        ScoreBenchmarkJob,
    )

    cases = [
        ("same", "hello world", "hello world"),
        ("sub", "hello world", "hello friend"),
        ("ins", "hello world", "hello dear world"),
        ("del", "hello dear world", "hello world"),
        ("empty", "hello world", ""),
    ]
    refs, hyps = tmp_path / "references", tmp_path / "hypotheses"
    refs.write_text(
        "\n".join(json.dumps(dict(id=key, group_id=key, chunk_index=0, reference=ref)) for key, ref, _ in cases)
    )
    hyps.write_text("\n".join(json.dumps(dict(id=key, text=hyp)) for key, _, hyp in cases))
    job = ScoreBenchmarkJob(tk.Path(str(refs)), tk.Path(str(hyps)), code_version="real-library-test")
    job.out_summary = tk.Path(str(tmp_path / "summary"))
    job.out_segments = tk.Path(str(tmp_path / "segments"))
    recorded = []
    job.out_wer = SimpleNamespace(set=recorded.append)
    job.run()
    summary = json.loads((tmp_path / "summary").read_text())
    assert {key: summary[key] for key in ("ins", "del", "sub", "total", "ref_len")} == {
        "ins": 1,
        "del": 3,
        "sub": 1,
        "total": 5,
        "ref_len": 11,
    }
    assert summary["num_scoring_units"] == 5
    assert summary["wer_percent"] == pytest.approx(100 * 5 / 11)
    assert recorded == [summary["wer_percent"]]
    segments = {row["id"]: row["counts"] for row in map(json.loads, (tmp_path / "segments").read_text().splitlines())}
    assert segments["empty"]["del"] == 2
    assert segments["same"]["total"] == 0


def test_scoring_changes_do_not_invalidate_recognition(monkeypatch):
    from pathlib import Path
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import source_version, scoring_version

    assert source_version() == "c37b1c7c749309ef10c6e704b895bdb7f689ffa5be54364a73afb0f017d73cb7"
    original_recognition, original_scoring = source_version(), scoring_version()
    original_read = Path.read_bytes

    def read_changed(path):
        content = original_read(path)
        if path.name == "benchmark_scoring.py":
            content += b"\n# scoring-only change\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", read_changed)
    assert source_version() == original_recognition
    assert scoring_version() != original_scoring


@pytest.mark.parametrize("mode", ["native", "rasr-free", "rasr-tree"])
@pytest.mark.parametrize("empty", [False, True])
def test_cohere_chunk_decoding_and_reassembly(mode, empty):
    from transformers import CohereAsrProcessor

    seen, processor_calls = [], []
    batch = dict(
        input_features=torch.tensor([[[4.0], [0.0]], [[5.0], [6.0]]]),
        attention_mask=torch.tensor([[1, 0], [1, 1]]),
        decoder_input_ids=torch.tensor([[1, 2], [1, 2]]),
        audio_chunk_index=[(0, 0), (0, 1)],
    )
    texts = {4: "" if empty else "hello", 5: "" if empty else "world"}

    class Processor:
        _reassemble_chunk_texts = staticmethod(CohereAsrProcessor._reassemble_chunk_texts)

        def __call__(self, waveforms, **kwargs):
            processor_calls.append(kwargs)
            assert len(waveforms) == 1
            return batch

        def batch_decode(self, raw, **kwargs):
            return [texts[int(raw[0, -2])]]

    adapter = HfBenchmarkAdapter.__new__(HfBenchmarkAdapter)
    adapter.family, adapter.device, adapter.eos, adapter.blank = "aed", "cpu", 9, None
    adapter.max_new_tokens = 500
    adapter.processor = Processor()
    adapter.tokenizer = SimpleNamespace(decode=lambda ids, **kw: texts[ids[0]] if ids else "")
    adapter.prepare_aed_policy = lambda inputs: inputs

    def inspect_inputs(inputs):
        i = len(seen)
        for key in ("input_features", "attention_mask", "decoder_input_ids"):
            torch.testing.assert_close(inputs[key], batch[key][i : i + 1])
        # Each chunk starts from the original prompt, not the previous output.
        seen.append(int(inputs["input_features"][0, 0, 0]))

    def generate(**inputs):
        assert inputs.pop("max_new_tokens") == 500
        assert inputs.pop("num_beams") == 1
        assert inputs.pop("do_sample") is False
        inspect_inputs(inputs)
        return torch.tensor([[1, 2, seen[-1], 9]])

    def encode(inputs):
        inspect_inputs(inputs)
        return np.zeros((2, 1), dtype=np.float32)

    def recognize(features):
        assert features.shape == (2, 1)
        word = f"piece-{seen[-1]}" if mode == "rasr-free" else texts[seen[-1]]
        eos = "<eos>" if mode == "rasr-free" else "[SENTENCE-END]"
        return [SimpleNamespace(lemma=w, am_score=0, lm_score=0, start_time=0, end_time=1) for w in [word, eos]]

    adapter.model = SimpleNamespace(generate=generate)
    adapter.encode_inputs = encode
    waveform = np.zeros(16, dtype=np.float32)
    if mode == "native":
        result = adapter.native(waveform)
    else:
        result = adapter.rasr(
            waveform,
            SimpleNamespace(recognize_segment=recognize),
            {"piece-4": 4, "piece-5": 5, "<eos>": 9},
            mode == "rasr-free",
        )
    assert result["text"] == ("" if empty else "hello world")
    assert seen == [4, 5]
    assert len(processor_calls) == 1
    assert processor_calls[0]["punctuation"] is False
    assert [chunk["audio_chunk_index"] for chunk in result["chunks"]] == [(0, 0), (0, 1)]
    if mode == "native":
        assert [r["raw_token_ids"] for r in result["chunks"]] == [[1, 2, 4, 9], [1, 2, 5, 9]]
    else:
        assert all(r["traceback"][0]["start_time"] == 0 for r in result["chunks"])
        assert "traceback" not in result  # no misleading utterance-level timestamps
    if mode != "rasr-tree":
        assert result["token_ids"] == [4, 5]
        assert result["token_chunk_lengths"] == [1, 1]


@pytest.mark.parametrize("indices", [None, [(0, 0)], [(0, 1), (0, 0)], [(0, 0), (1, 1)]])
def test_cohere_rejects_inconsistent_chunk_metadata(indices):
    adapter = HfBenchmarkAdapter.__new__(HfBenchmarkAdapter)
    adapter.family, adapter.device = "aed", "cpu"
    adapter.processor = lambda *a, **kw: dict(input_features=torch.zeros(2, 10, 8), audio_chunk_index=indices)
    with pytest.raises(ValueError, match="chunk"):
        list(adapter.input_chunks(np.zeros(160, dtype=np.float32)))


def test_chunk_fix_keeps_data_and_other_model_versions(monkeypatch):
    from pathlib import Path
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import (
        source_version,
        recognition_version,
    )

    original = "c37b1c7c749309ef10c6e704b895bdb7f689ffa5be54364a73afb0f017d73cb7"
    assert source_version() == original
    assert recognition_version("ctc") == recognition_version("rnnt") == original
    assert recognition_version("aed") != original
    read = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: read(p) + (b"\n# future edit" if p.name == "hf_benchmark.py" else b"")
    )
    assert source_version() == original
    assert recognition_version("ctc") != original


def test_aed_native_cap_returns_partial_without_appending_eos():
    adapter = HfBenchmarkAdapter.__new__(HfBenchmarkAdapter)
    adapter.family, adapter.device, adapter.eos = "aed", "cpu", 3
    adapter.max_new_tokens = 3
    raw = torch.tensor([[9, 4, 5, 6]])  # one prompt token, three generated labels
    adapter.model = SimpleNamespace(generate=lambda **kwargs: raw, config=SimpleNamespace(vocab_size=10))
    adapter.processor = SimpleNamespace(batch_decode=lambda *args, **kwargs: ["hello hello hello"])
    result = adapter.native_inputs(dict(decoder_input_ids=torch.tensor([[9]])))
    assert result["token_ids"] == [4, 5, 6]
    assert result["raw_token_ids"] == [9, 4, 5, 6]
    assert result["truncated"] is True
    assert result["text"] == "hello hello hello"
    adapter.model.generate = lambda **kwargs: torch.tensor([[9, 4, 3]])
    result = adapter.native_inputs(dict(decoder_input_ids=torch.tensor([[9]])))
    assert result["raw_token_ids"] == [9, 4, 3]
    assert result["token_ids"] == [4]
    assert "truncated" not in result
    adapter.model.generate = lambda **kwargs: torch.tensor([[9, 4]])
    with pytest.raises(ValueError, match="before its token limit"):
        adapter.native_inputs(dict(decoder_input_ids=torch.tensor([[9]])))


def test_aed_rasr_partial_and_single_placeholder():
    adapter = HfBenchmarkAdapter.__new__(HfBenchmarkAdapter)
    adapter.family, adapter.eos, adapter.blank = "aed", 3, None
    adapter.model = SimpleNamespace(
        get_encoder=lambda: lambda **kwargs: SimpleNamespace(last_hidden_state=torch.zeros(1, 17, 4))
    )
    features = adapter.encode_inputs(dict(decoder_input_ids=torch.tensor([[9]])))
    assert features.shape == (1, 1)
    assert adapter.encoder.last_hidden_state.shape[1] == adapter.num_frames == 17
    adapter.decode_chunks = lambda waveform, decode: decode(dict(decoder_input_ids=torch.tensor([[9]])))
    adapter.detokenize = lambda ids: "partial text"
    trace = [SimpleNamespace(lemma="piece", am_score=2, lm_score=0, start_time=0, end_time=1)]
    result = adapter.rasr(None, SimpleNamespace(recognize_segment=lambda features: trace), {"piece": 4}, True)
    assert result["token_ids"] == [4]
    assert result["text"] == "partial text"
    assert result["truncated"] is True
    assert result["traceback"][0]["am_score"] == 2


def test_failed_suite_job_fix_is_targeted():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import (
        recognition_version,
        run_all,
    )

    unchanged = recognition_version("aed", "voxpopuli-cleaned")
    assert recognition_version("aed", "earnings22-cleaned") != unchanged
    assert recognition_version("aed", "gigaspeech-cleaned") != unchanged
    results = run_all(neural_lms=False, aed_tree=False, granite_open_tree=True, granite_open_transformer=True)
    for condition in ("native", "rasr-lexfree"):
        for dataset in ("earnings22-cleaned", "gigaspeech-cleaned"):
            job = results[f"cohere-aed/{condition}"][dataset].creator.hypotheses.creator
            assert job.max_new_tokens == 500
            if condition == "rasr-lexfree":
                assert job.rasr_config.creator.config.lib_rasr.search_algorithm.max_labels_per_timestep == 500
    for dataset, path in results["granite-ctc/tree-open-control-trafo"].items():
        assert path.creator.hypotheses.creator.rqmt["time"] == (96 if dataset == "spgispeech" else 48)


@pytest.mark.parametrize("family,tree", [("ctc", False), ("ctc", True), ("aed", False), ("aed", True)])
def test_native_rasr_smoke(tmp_path, family, tree):
    """Exercise the actual pinned Python ABI/tree, if its completed build exists."""
    import os
    import subprocess
    from pathlib import Path
    from sisyphus import tk
    from i6_core.rasr.command import RasrCommand
    from i6_core.rasr.config import RasrConfig
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import recognition_config

    binary = Path(
        "/work/asr4/berger/sisyphus_work_dirs/librispeech/20241106_seq2seq_template_setups/"
        "i6_core/tools/compile/CMakeJob.AhnICBf8VYpg/output/install"
    )
    if not list(binary.glob("librasr*.so")):
        pytest.skip("Completed multi-token RASR build not present")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("<blank>\na\nb\n<eos>\n")
    lexicon = tmp_path / "lexicon.xml"
    lexicon.write_text(
        """<lexicon><phoneme-inventory>
        <phoneme><symbol>&lt;blank&gt;</symbol><variation>none</variation></phoneme>
        <phoneme><symbol>a</symbol><variation>none</variation></phoneme>
        <phoneme><symbol>b</symbol><variation>none</variation></phoneme>
        <phoneme><symbol>&lt;eos&gt;</symbol><variation>none</variation></phoneme>
        </phoneme-inventory><lemma><orth>WORD</orth><phon>a a b</phon></lemma>
        <lemma special="sentence-begin"><orth>[SENTENCE-BEGIN]</orth><synt><tok>&lt;s&gt;</tok></synt></lemma>
        <lemma special="sentence-end"><orth>[SENTENCE-END]</orth>"""
        + ("<phon>&lt;eos&gt;</phon>" if family == "aed" else "")
        + """<synt><tok>&lt;/s&gt;</tok></synt></lemma>"""
        + (
            """<lemma special="blank"><orth>[BLANK]</orth><phon>&lt;blank&gt;</phon><synt/></lemma>"""
            if family == "ctc"
            else ""
        )
        + "</lexicon>"
    )
    cfg = recognition_config(
        family,
        SimpleNamespace(out_vocab=tk.Path(str(vocab)), out_blank=0, out_eos=3),
        lexicon=tk.Path(str(lexicon)) if tree else None,
        beam=32 if tree else 1,
    )
    if family == "aed" and not tree:
        cfg.lib_rasr.search_algorithm.max_labels_per_timestep = 3
    for channel in ("log", "warning", "error", "statistics"):
        cfg[f"*.{channel}.channel"] = str(tmp_path / "rasr.log")
    path = tmp_path / "rasr.config"
    RasrCommand.write_config(cfg, RasrConfig(), str(path))
    script = r"""
import sys, resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import numpy as np
sys.path.insert(0, sys.argv[1])
import librasr
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.hf_benchmark import register_hf_scorer
class Adapter:
    family = "aed"
    def score_contexts(self, contexts):
        out = []
        for t, prefix in contexts:
            # This scorer NEVER prefers EOS: RASR must return the partial path
            # after three labels by its own feature-relative stopping rule.
            if sys.argv[4] != 'tree': assert len(prefix) < 3, prefix
            next_id = [1, 1, 2, 3][min(len(prefix), 3)]
            scores = [1000.] * 4
            scores[next_id] = 0.
            out.append((scores, len(prefix)))
        return out
if sys.argv[3] == 'aed': register_hf_scorer(librasr, Adapter())
cfg = librasr.Configuration()
assert cfg.set_from_file(sys.argv[2])
librasr.init_logging(cfg)
search = librasr.SearchAlgorithm(config=cfg)
path = [0, 1, 1, 0, 1, 1, 0, 2, 2, 0]
scores = np.full((len(path), 4), 1000., dtype=np.float32)
scores[np.arange(len(path)), path] = 0.
if sys.argv[3] == 'aed': scores = np.zeros((1, 1), dtype=np.float32)
trace = search.recognize_segment(scores)
if sys.argv[3] == 'aed' and sys.argv[4] != 'tree':
    assert all(x.lemma not in ('<eos>', '[SENTENCE-END]') for x in trace)
words = [x.lemma for x in trace if x.lemma not in ('<blank>', '[BLANK]', '[SENTENCE-END]', '<eos>')]
assert words == (['WORD'] if sys.argv[4]=='tree' else ['a','a','b']), words
"""
    result = subprocess.run(
        ["/usr/bin/python3", "-c", script, str(binary), str(path), family, "tree" if tree else "free"],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ,
    )
    logs = (tmp_path / "rasr.log").read_text() if (tmp_path / "rasr.log").exists() else ""
    assert result.returncode == 0, result.stdout + result.stderr + logs[-5000:]
