"""Coverage vs ranking, no fabricated union score, independent slicing inputs."""

import gzip
import json

import pytest
from sisyphus import tk

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import research_diagnostics as rd
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import controlled_comparison as cc
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import research_diagnostics as driver


def candidate(text, first, nll):
    return dict(
        text=text,
        am_cost=first,
        lm_cost=0,
        first_pass_cost=first,
        speech={"original": dict(nll=nll, token_costs=[nll / 2, nll / 2])},
    )


def qunit(ref, hyp):
    counts, normalized = rd.official_counts(ref, hyp)
    return dict(counts=counts, hypothesis=normalized, reference=ref)


SETTING = dict(n=50, scale=0.3, word_penalty=0, policy="original")


def test_coverage_ranking_and_retained_am_cost_are_distinct():
    q = qunit("the cat", "the cat")
    missing = rd.diagnose_pool("the cat", [candidate("the dog", 0, 1)], q, SETTING)
    assert missing["gap"] == "coverage_gap_vs_qwen"
    assert missing["augmented_oracle_errors"] == 0
    assert missing["augmented_candidate_am_score"] is None
    # Correct candidate exists and Speech-Qwen prefers it, but its AM cost wins out.
    ranked = rd.diagnose_pool("the cat", [candidate("the dog", 0, 10), candidate("the cat", 10, 1)], q, SETTING)
    assert ranked["gap"] == "ranking_gap_vs_qwen" and ranked["ranking_missed_correct"]
    assert ranked["compound_equivalent_qwen_in_pool"] and ranked["normalized_qwen_in_pool"]
    assert ranked["first_pass_overturns_better_speech_choice"]
    assert not ranked["word_reward_overturns_better_speech_choice"]


def test_word_reward_is_not_misattributed_to_first_pass():
    rows = [candidate("cat", 0, 1), candidate("a very big dog", 0, 2)]
    r = rd.diagnose_pool("cat", rows, qunit("cat", "cat"), dict(SETTING, word_penalty=-1))
    assert r["word_reward_overturns_better_speech_choice"]
    assert not r["first_pass_overturns_better_speech_choice"]
    with pytest.raises(ValueError, match="cumulative"):
        rd.diagnose_pool("cat", [dict(rows[0], first_pass_cost=5)], qunit("cat", "cat"), SETTING)


def test_fixed_bins_and_source_labels():
    assert [rd.frequency_bin(x) for x in [0, 1, 5, 6, 20, 21, 100, 101]] == [
        "unseen",
        "1-5",
        "1-5",
        "6-20",
        "6-20",
        "21-100",
        "21-100",
        ">100",
    ]
    assert rd.duration_bin(4.9) == "<5s" and rd.duration_bin(5) == "<10s"
    assert rd.duration_bin(300) == ">=300s"
    assert rd.source_bin("loquacious-test", "common_voice_en_123") == "commonvoice"
    assert rd.source_bin("loquacious-test", "123-456-789") == "librispeech"
    assert rd.source_bin("loquacious-test", "movie.wav") == "yodas"


def test_duration_metadata_groups_chunks_and_rejects_missing_audio_length(tmp_path):
    refs = tmp_path / "refs.jsonl"
    hyp = tmp_path / "hyp.jsonl"
    refs.write_text("".join(json.dumps(dict(id=str(i), group_id="recording", chunk_index=i)) + "\n" for i in range(2)))
    hyp.write_text("".join(json.dumps(dict(id=str(i), audio_seconds=2.5)) + "\n" for i in range(2)))
    job = rd.ScoringUnitDurationsJob(tk.Path(str(hyp)), tk.Path(str(refs)), code_version="test")
    job.out_durations = tk.Path(str(tmp_path / "durations.json"))
    job.run()
    value = json.loads((tmp_path / "durations.json").read_text())
    assert value["units"]["recording"] == dict(seconds=5, chunks=2, max_chunk_seconds=2.5)
    hyp.write_text('{"id":"0","elapsed_seconds":5}\n')
    with pytest.raises(ValueError, match="recorded audio duration"):
        job.run()


def test_training_frequencies_use_normalized_training_text_only(tmp_path):
    source = tmp_path / "train.xml.gz"
    with gzip.open(source, "wt") as f:
        f.write(
            '<corpus name="train-other-960"><recording><segment><orth>HELLO WORLD</orth></segment><segment><orth>HELLO</orth></segment></recording></corpus>'
        )
    job = rd.TrainingWordFrequenciesJob(tk.Path(str(source)), code_version="test")
    job.out_frequencies = tk.Path(str(tmp_path / "frequencies.json"))
    job.run()
    counts = json.loads((tmp_path / "frequencies.json").read_text())
    assert counts["counts"] == dict(hello=2, world=1) and counts["normalizer"] == cc.NORMALIZER
    with gzip.open(source, "wt") as f:
        f.write('<corpus name="dev-other"/>')
    with pytest.raises(ValueError, match="not evaluation text"):
        job.run()


def test_no_training_audio_download_is_scheduled(tmp_path):
    with pytest.raises(ValueError, match="ASR_RARITY_TRAINING_CORPUS"):
        driver.training_corpus_path(tmp_path)
    with pytest.raises(ValueError, match="development-only"):
        rd.CandidateRankingDiagnosticsJob(
            "librispeech-test-other", tk.Path("snapshot"), tk.Path("protocol"), code_version="test"
        )


def test_ranking_job_uses_frozen_settings_and_checks_actual_selection(tmp_path):

    rows = [candidate("the dog", 0, 10), candidate("the cat", 10, 1)]
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps(dict(id="x", candidates=rows)) + "\n")
    refs = tmp_path / "refs.jsonl"
    refs.write_text(json.dumps(dict(id="x", group_id="x", chunk_index=0, reference="the cat")) + "\n")
    records = []
    for group, condition, model, lm, hyp in [
        ("qwen-speech-1.7b", "native", "Speech Qwen 1.7B", "None", "the cat"),
        ("qwen-speech-1.7b", "two-pass-speech-n50", "Granite CTC", "Speech Qwen 1.7B", "the dog"),
    ]:
        folder = tmp_path / condition
        folder.mkdir()
        u = dict(id="x", **qunit("the cat", hyp))
        (folder / "segments.jsonl").write_text(json.dumps(u) + "\n")
        summary = dict(u["counts"], wer_percent=100 * u["counts"]["total"] / u["counts"]["ref_len"])
        (folder / "summary.json").write_text(json.dumps(summary))
        records.append(
            dict(
                group=group,
                condition=condition,
                model=model,
                lm=lm,
                status="complete",
                dataset=cc.DEV[0],
                score=str(folder / "summary.json"),
                score_sha256=cc.sha256(folder / "summary.json"),
                summary=summary,
                segments=str(folder / "segments.jsonl"),
                references=str(refs),
                normalizer=cc.NORMALIZER,
                reference_sha256=cc.sha256(refs),
                dataset_paths=["same"],
                stages=[dict(job_type="SelectSpeechNbestJob", parameters=dict(SETTING, candidates=str(pool)))],
            )
        )
    protocol = dict(
        ready=True,
        selected={
            "Speech Qwen 1.7B | None | standalone | fast": dict(
                family="Speech Qwen 1.7B | None | standalone",
                group=records[0]["group"],
                condition=records[0]["condition"],
            ),
            "Granite CTC | Speech Qwen 1.7B | two-pass | fast": dict(
                family="Granite CTC | Speech Qwen 1.7B | two-pass",
                group=records[1]["group"],
                condition=records[1]["condition"],
            ),
        },
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(dict(records=records)))
    proto = tmp_path / "protocol.json"
    proto.write_text(json.dumps(protocol))
    job = rd.CandidateRankingDiagnosticsJob(cc.DEV[0], tk.Path(str(snapshot)), tk.Path(str(proto)), code_version="test")
    for field, suffix in [("summary", "json"), ("units", "jsonl.gz"), ("examples", "jsonl"), ("report", "md")]:
        setattr(job, "out_" + field, tk.Path(str(tmp_path / (field + "." + suffix))))
    job.run()
    result = json.loads((tmp_path / "summary.json").read_text())["results"][0]
    assert result["counters"]["ranking_gap_vs_qwen"] == 1
    assert result["counters"]["augmented_oracle_errors"] == 0
    assert result["scores"]["selected"]["wer"] == 50


def test_stratification_common_grid_neighbors_numbers_and_no_pooling(tmp_path):
    from kaldialign import edit_distance

    records = []
    specs = [
        ("Granite CTC", "qwen-granite", "native", "None", ["rare broken 7", "cannot"]),
        ("Speech Qwen 1.7B", "qwen-speech-1.7b", "native", "None", ["rare common 7", "can not"]),
        ("Granite CTC", "qwen-speech-1.7b", "granite-fusion", "Speech Qwen 1.7B", ["rare common 8", "cannot"]),
    ]
    for i, (model, group, condition, lm, hyps) in enumerate(specs):
        folder = tmp_path / str(i)
        folder.mkdir()
        refs = ["rare common 7", "can not" if i == 1 else "cannot"]
        units = [
            dict(id=str(j), reference=r, hypothesis=h, counts=edit_distance(r.split(), h.split()))
            for j, (r, h) in enumerate(zip(refs, hyps))
        ]
        summary = {k: sum(r["counts"][k] for r in units) for k in ("ins", "del", "sub", "total", "ref_len")}
        summary["wer_percent"] = 100 * summary["total"] / summary["ref_len"]
        (folder / "summary.json").write_text(json.dumps(summary))
        (folder / "segments.jsonl").write_text("".join(json.dumps(r) + "\n" for r in units))
        records.append(
            dict(
                dataset="loquacious-dev.short",
                model=model,
                lm=lm,
                group=group,
                condition=condition,
                status="complete",
                decoding="Native" if lm == "None" else "1-pass fusion",
                score=str(folder / "summary.json"),
                score_sha256=cc.sha256(folder / "summary.json"),
                summary=summary,
                segments=str(folder / "segments.jsonl"),
                normalizer=cc.NORMALIZER,
                reference_sha256="same",
                dataset_paths=["same"],
            )
        )
    files = {}
    for name, value in dict(
        snapshot=dict(records=records),
        protocol=dict(ready=True, selected={}),
        durations=dict(
            reference_sha256="same", units={"0": dict(seconds=12, chunks=1), "1": dict(seconds=600, chunks=2)}
        ),
        frequencies=dict(
            normalizer=cc.NORMALIZER,
            source="training only",
            source_sha256="hash",
            counts={"rare": 1, "common": 1000, "7": 1},
        ),
    ).items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        files[name] = tk.Path(str(path))
    job = rd.StratifiedErrorsJob(
        "loquacious-dev.short",
        [files["snapshot"]],
        files["protocol"],
        files["durations"],
        files["frequencies"],
        code_version="test",
    )
    job.out_summary = tk.Path(str(tmp_path / "out.json"))
    job.out_report = tk.Path(str(tmp_path / "out.md"))
    job.run()
    p = json.loads((tmp_path / "out.json").read_text())["panels"]["native"]
    assert p["word_units_included"] == 1 and p["word_units_excluded_reference_mismatch"] == 1
    fusion = p["systems"][2]
    assert fusion["words"]["category/contains-digit"]["accuracy"] == 0
    assert fusion["paired_word_changes"]["Granite CTC"]["context/common-neighbor-of-rare"]["corrected"] == 1
    assert fusion["paired_word_changes"]["Granite CTC"]["frequency/1-5"]["introduced"] == 1
    assert fusion["slices"]["duration/>=300s"]["units"] == 1
    assert fusion["slices"]["input/multi-chunk"]["units"] == 1
