"""Step 2 has no inference/test selection and never conflates word/unit oracles."""

import json
from collections import Counter

import pytest

from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import error_complementarity as ec
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common import controlled_comparison as cc


def unit(seq_id, ref, hyp):
    from kaldialign import edit_distance

    return dict(id=seq_id, reference=ref, hypothesis=hyp, counts=edit_distance(ref.split(), hyp.split()))


def evidence(tmp_path, model, group, condition, rows, lm="None", decoding="Native"):
    directory = tmp_path / group / condition
    directory.mkdir(parents=True, exist_ok=True)
    segments, score = directory / "segments.jsonl", directory / "summary.json"
    segments.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = {key: sum(r["counts"][key] for r in rows) for key in ("ins", "del", "sub", "total", "ref_len")}
    summary = dict(counts, wer_percent=100 * counts["total"] / counts["ref_len"])
    score.write_text(json.dumps(summary))
    return dict(
        model=model,
        group=group,
        condition=condition,
        lm=lm,
        decoding=decoding,
        lexicon="None",
        dataset="librispeech-test-other",
        status="complete",
        summary=summary,
        timing=dict(rtf=0.1),
        normalizer=cc.NORMALIZER,
        reference_sha256="same-reference",
        dataset_paths=["same-audio"],
        score=str(score),
        score_sha256=cc.sha256(score),
        segments=str(segments),
    )


@pytest.fixture
def pair(tmp_path):
    refs = ["the cat"] * 5 + ["we go", "a", "cannot"]
    aa = ["the cat", "the cat", "the dog", "the dog", "the dog", "we go", "a x x", "cannot"]
    bb = ["the cat", "the dog", "the cat", "the dog", "a cat", "we go", "a x", "can not"]
    ff = ["the dog", "the cat", "the cat", "the dog", "the cat", "we", "a x y", "cannot"]
    ua = [unit(str(i), r, h) for i, (r, h) in enumerate(zip(refs, aa))]
    ub = [unit(str(i), "can not" if i == 7 else r, h) for i, (r, h) in enumerate(zip(refs, bb))]
    uf = [unit(str(i), r, h) for i, (r, h) in enumerate(zip(refs, ff))]
    a = evidence(tmp_path, "Granite CTC", "qwen-granite", "native", ua)
    b = evidence(tmp_path, "Speech Qwen 1.7B", "qwen-speech-1.7b", "native", ub)
    f = evidence(
        tmp_path,
        "Granite CTC",
        "qwen-speech-1.7b",
        "granite-fusion",
        uf,
        lm="Speech Qwen 1.7B",
        decoding="1-pass fusion",
    )
    return a, b, f, {r["id"]: r for r in ua}, {r["id"]: r for r in ub}, {r["id"]: r for r in uf}


def test_unit_categories_and_word_normalization_exclusions(pair):
    a, b, f, ua, ub, uf = pair
    assert Counter(ec.unit_category(ua[k], ub[k]) for k in ua) == dict(
        both_correct=3,
        only_a_correct=1,
        only_b_correct=1,
        both_wrong_same_hypothesis=1,
        both_wrong_different_hypotheses=2,
    )
    assert ec.word_complementarity(ua["7"], ub["7"]) == dict(word_units_excluded_reference_mismatch=1)
    inserts = ec.word_complementarity(ua["6"], ub["6"])
    assert inserts["shared_insertions"] == 1 and inserts["only_a_insertions"] == 1
    changes = ec.error_changes(ua["6"], uf["6"])
    assert changes["removed_insertions"] == changes["added_insertions"] == 1


def test_selection_oracle_and_fusion_can_beat_it_within_a_unit(pair):
    a, b, f, ua, ub, uf = pair
    units, examples = [], []
    result = ec.analyze_pair(
        a, b, ua, ub, {"fusion": (f, uf)}, unit_sink=units.append, example_sink=examples.append, examples_per_category=1
    )
    expected_errors = sum(min(ua[k]["counts"]["total"], ub[k]["counts"]["total"]) for k in ua)
    assert result["selection_oracle"]["counts"]["total"] == expected_errors
    assert result["selection_oracle"]["diagnostic_only"] is True
    stats = result["fusions"]["fusion"]
    assert stats["units"]["beats_selection_oracle"] >= 1
    assert stats["units"]["both_correct_units_broken"] == 2
    assert stats["units"]["both_wrong_units_corrected"] == 1
    assert stats["units"]["only_b_correct_recovered"] == 1
    assert stats["words_vs_b"]["word_units_excluded_reference_mismatch"] == 1
    assert stats["both_correct_words_broken"] == 2
    assert stats["both_wrong_words_corrected"] == 0  # Different wrong positions in unit 4.
    assert len(units) == len(ua)
    assert all(sum(x["example_kind"] == e["example_kind"] for x in examples) <= 1 for e in examples)


def test_pairing_requires_same_audio_references_and_coverage(pair):
    a, b, _, ua, ub, _ = pair
    with pytest.raises(ValueError, match="raw references"):
        ec.check_pair(a, dict(b, reference_sha256="other"), ua, ub)
    with pytest.raises(ValueError, match="prepared audio"):
        ec.check_pair(a, dict(b, dataset_paths=["different"]), ua, ub)
    with pytest.raises(ValueError, match="coverage"):
        ec.check_pair(a, b, ua, {})


def test_panels_do_not_pick_best_test_or_substitute_missing_selected(pair):
    a, b, f, *_ = pair
    protocol = dict(
        ready=True,
        selected={
            "Granite CTC | None | standalone | fast": dict(
                group="qwen-granite", condition="not-yet-run", family="Granite CTC | None | standalone"
            ),
            "Granite CTC | Speech Qwen 1.7B | one-pass | fast": dict(
                group=f["group"], condition=f["condition"], family="Granite CTC | Speech Qwen 1.7B | one-pass"
            ),
        },
    )
    controls, fs, missing = ec.panel_records([a, b, f], protocol, "native")
    assert controls["Granite CTC"] is a and f in fs
    controls, fs, missing = ec.panel_records([a, b, f], protocol, "dev-selected")
    assert "Granite CTC" not in controls and missing[0]["condition"] == "not-yet-run"
    controls, fs, missing = ec.panel_records([a, b, f], dict(protocol, ready=False), "dev-selected")
    assert not controls and not fs and missing


def test_score_checksum_guard_and_alignment_validation(pair):
    a, _, _, ua, _, _ = pair
    assert len(ec.load_units(a)) == len(ua)
    with pytest.raises(ValueError, match="changed after snapshot"):
        ec.load_units(dict(a, score_sha256="not-the-hash"))
    with pytest.raises(ValueError, match="Alignment disagrees"):
        ec.aligned_errors(dict(ua["0"], counts=dict(ua["0"]["counts"], total=1)))
