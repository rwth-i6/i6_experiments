"""Live reports remain available without scheduling archived experiments."""

import json

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import result_collection as reports


def export(tmp_path, group, condition, dataset, wer, normalizer="new"):
    directory = tmp_path / "open-asr" / group / condition / dataset
    directory.mkdir(parents=True)
    (directory / "scoring").write_text(json.dumps(dict(wer_percent=wer, normalizer_revision=normalizer)))
    (directory / "runtime").write_text(json.dumps(dict(rtf=0.12)))
    return directory


def test_comparisons_inventory_and_provenance(tmp_path):
    export(tmp_path, "granite-ctc", "native", "dev", 3.0)
    export(tmp_path, "granite-ctc", "tree-open-trafo", "dev", 2.8)
    export(tmp_path, "qwen-granite", "one-pass", "dev", 2.7)
    export(tmp_path, "qwen-granite", "old-normalizer", "dev", 2.6, "old")
    invalid = "tree-open-corpus-voxpopuli-cleaned-ls4gram"
    export(tmp_path, "granite-ctc", invalid, "voxpopuli", 98.0)
    pending = tmp_path / "open-asr/qwen-parakeet-rnnt/speech-qwen/dev"
    pending.mkdir(parents=True)
    (pending / "scoring").symlink_to(tmp_path / "missing-score")
    broken = export(tmp_path, "qwen-granite", "bad-json", "dev", 1.0)
    (broken / "scoring").write_text("{")
    collection = reports.ResultCollection(tmp_path)
    comparison = collection.comparisons()
    assert "Granite CTC | Text Qwen 0.6B | 1-pass fusion" in comparison
    assert "2.70 | -0.30 | 0.120" in comparison
    assert "2.60 | — | 0.120" in comparison
    assert "no completed export" in comparison
    assert "unreadable: JSONDecodeError" in comparison
    assert invalid not in collection.comparisons(word_lms=True)
    assert invalid in collection.inventory()
    assert "normalizer\tsource\tscore_job" in collection.tsv()
    assert "scoring" in collection.tsv()


def test_own_models_missing_then_readable(tmp_path):
    collection = reports.ResultCollection(tmp_path)
    assert "No own-model summary" in collection.own_models()
    path = tmp_path / "wordwise-lm/dev/summary"
    path.parent.mkdir(parents=True)
    path.write_text("header\nseparator\n1__model | dev | 5.2 | 0.5 | 0.5 | 4.2 | 0.01 | 0.1 | 0.11\n")
    assert "| model | dev | 5.2 | 0.5 | 0.5 | 4.2 | 0.11 |" in collection.own_models()


def test_speech_size_specific_standalone_comparison(tmp_path):
    export(tmp_path, "qwen-granite", "native", "librispeech-test-other", 4.0)
    export(tmp_path, "qwen-speech-0.6b", "native-beam8", "librispeech-test-other", 1.0)
    export(tmp_path, "qwen-speech-1.7b", "native-beam8", "librispeech-test-other", 3.0)
    export(tmp_path, "qwen-speech-1.7b", "granite-fusion", "librispeech-test-other", 2.5)
    export(tmp_path, "qwen-parakeet-rnnt-speech-1.7b", "speech-qwen-fast", "librispeech-test-other", 2.7)
    report = reports.ResultCollection(tmp_path).corpus_report("librispeech")
    assert "Granite CTC | Speech Qwen 1.7B | 1-pass fusion | None | 2.50 | -0.50" in report
    assert "Parakeet RNN-T | Speech Qwen 1.7B | 1-pass fusion | None | 2.70 | -0.30" in report


def test_registration_adds_reports_only_and_is_idempotent(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(reports.tk, "register_report", lambda *args, **kwargs: calls.append((args, kwargs)))
    collection = reports.register_result_collection(output_root=tmp_path)
    assert reports.register_result_collection(output_root=tmp_path) is collection
    assert len(calls) == 10
    assert all(call[1] == dict(required=[], update_frequency=300) for call in calls)
    assert all(call[0][0].startswith("reports/asr-lm/") for call in calls)
    assert all(callable(call[0][1]) for call in calls)


def test_corpus_breakdown_models_and_subsplits(tmp_path):
    export(tmp_path, "qwen-granite", "native", "librispeech-test-other", 2.7)
    export(tmp_path, "qwen-granite", "two-pass-librispeech-lexicon", "librispeech-test-other", 2.5)
    export(tmp_path, "qwen-speech-0.6b", "native", "loquacious-test.yodas", 7.7)
    export(tmp_path, "qwen-parakeet-rnnt", "speech-native", "loquacious-test.yodas", 7.7)
    export(tmp_path, "qwen-parakeet-rnnt", "speech-qwen", "loquacious-test.yodas", 7.0)
    export(tmp_path, "granite-ctc", "tree-open-corpus-gigaspeech-cleaned-corpus4gram", "gigaspeech-cleaned", 8.8)
    collection = reports.ResultCollection(tmp_path)
    ls = collection.corpus_report("librispeech")
    assert "2-pass (50-best) | LS (open)" in ls
    assert "Text Qwen 0.6B" in ls
    assert "loquacious" not in ls
    loq = collection.corpus_report("loquacious")
    assert "## loquacious-test.yodas" in loq
    assert loq.count("Speech Qwen 0.6B | None | Native") == 1
    assert "Parakeet RNN-T | Speech Qwen 0.6B" in loq
    suite = collection.corpus_report("open_asr")
    assert "Corpus 4-gram" in suite
    assert "librispeech-test-other" not in suite
    assert "Qwen/Qwen3-ASR-0.6B-hf" in collection.terminology()


def test_larger_qwen_and_fast_fusion_labels(tmp_path):
    export(tmp_path, "qwen-granite-1.7b", "one-pass-scale-0.1", "librispeech-dev-other", 2.5)
    export(tmp_path, "qwen-parakeet-rnnt", "speech-qwen-fast", "librispeech-test-other", 2.0)
    report = reports.ResultCollection(tmp_path).corpus_report("librispeech")
    assert "Granite CTC | Text Qwen 1.7B | 1-pass fusion" in report
    assert "LM 0.1" in report
    assert "LM 0.3; beam 8; pre 32; thr 3" in report


def test_own_models_reads_individual_scores_instead_of_stale_summary(tmp_path):
    directory = tmp_path / "librispeech/recognition/ffnn_transducer_bpe__recog_wordwise-ls-word-trafo-lm0.4-ilm0.2/dev-other"
    directory.mkdir(parents=True)
    score = tmp_path / "scoring-job/output"
    (score / "reports").mkdir(parents=True)
    (score / "wer").write_text("5.1")
    (directory / "scoring_reports").symlink_to(score / "reports")
    collection = reports.ResultCollection(tmp_path)
    report = collection.corpus_report("librispeech")
    assert "LibriSpeech Context-1 mRNN-T" in report
    assert "LS word Trafo" in report
    assert "5.10" in report
    assert "LM 0.4; ILM 0.2" in report
    assert "5.10" in collection.own_models()
    assert "own-librispeech" in collection.tsv()
