import ast
import json
import math

from i6_experiments.users.sheremeta.recognition.scale_race import (
    ScaleRace,
    ScaleRaceForwardCallback,
    active_race,
    student_t_quantile,
    word_edit_distance,
)


N_SEQS = 512
BATCH = 16
REF = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"


def _refs():
    return {f"u{i}": REF for i in range(N_SEQS)}


def _decode(tokens):
    return " ".join(f"w{t}" for t in tokens)


def _tokens(index: int, scale: float):
    """scale 0.3 and 0.5 are right, 0.0 substitutes one word on every 16th utterance, 1.0 deletes three on every second"""
    words = list(range(10))
    if scale == 0.0 and index % 16 == 0:
        words[3] = 99
    if scale == 1.0 and index % 2 == 0:
        words = words[:7]
    return words


class _Search:
    def __init__(self, tags):
        self.tags = tags
        self.calls = {}

    def __call__(self, scale):
        self.calls[scale] = self.calls.get(scale, 0) + 1
        return [_tokens(int(tag[1:]), scale) for tag in self.tags]


def _run(race):
    calls = {}
    for start in range(0, N_SEQS, BATCH):
        tags = [f"u{i}" for i in range(start, start + BATCH)]
        search = _Search(tags)
        race.run_batch(seq_tags=tags, search=search)
        for scale, count in search.calls.items():
            calls[scale] = calls.get(scale, 0) + count
    return calls


def test_word_edit_distance():
    for hyp, ref, expected in (
        ("", "", 0),
        ("a b c", "a b c", 0),
        ("a b c", "a x c", 1),
        ("a b", "a b c", 1),
        ("", "a b", 2),
        ("b a", "a b", 2),
        ("a a a a", "a", 3),
    ):
        assert word_edit_distance(hyp.split(), ref.split()) == expected, (hyp, ref)


def test_student_t_quantile_matches_the_tables():
    for p, dof, expected in (
        (0.975, 1, 12.706),
        (0.975, 10, 2.228),
        (0.995, 30, 2.750),
        (0.99, 5, 3.365),
        (0.9995, 100, 3.390),
        (0.975, 1_000_000, 1.960),
    ):
        assert math.isclose(student_t_quantile(p, dof), expected, abs_tol=2e-3), (p, dof)


def test_race_drops_the_worse_scales_at_the_expected_looks_and_keeps_the_equal_ones():
    race = ScaleRace(candidates=(1.0, 0.0, 0.5, 0.3), refs=_refs(), decode=_decode)
    assert race.level == 0.05 / (32 * 4 * 3)
    calls = _run(race)
    result = race.finish()

    assert [look["seqs"] for look in result["looks"]] == [128, 256, 512]
    assert result["per_candidate"]["1.0"]["eliminated"]["seqs"] == 128
    assert result["per_candidate"]["0.0"]["eliminated"]["seqs"] == 256
    assert result["per_candidate"]["0.0"]["eliminated"]["against"] in (0.3, 0.5)
    assert result["survivors"] == [0.3, 0.5]
    assert result["band"] == [0.3, 0.5]
    assert result["winner"] == 0.3 and result["chosen"] == 0.3
    assert calls == {1.0: 8, 0.0: 16, 0.3: 32, 0.5: 32}
    assert result["per_candidate"]["1.0"]["seqs"] == 128 and result["per_candidate"]["0.0"]["seqs"] == 256
    assert result["per_candidate"]["0.3"]["seqs"] == N_SEQS and result["per_candidate"]["0.3"]["errors"] == 0
    assert math.isclose(result["per_candidate"]["1.0"]["wer"], 15.0)
    assert result["ref_words"] == 10 * N_SEQS
    assert [tag for tag, _text in race.decoded(0.3)] == [f"u{i}" for i in range(N_SEQS)]
    assert all(text == REF for _tag, text in race.decoded(0.3))


def test_pick_selects_within_the_band():
    for pick, expected in (("smallest", 0.3), ("largest", 0.5)):
        race = ScaleRace(candidates=(0.0, 0.3, 0.5, 1.0), refs=_refs(), decode=_decode, pick=pick)
        _run(race)
        assert race.finish()["chosen"] == expected, pick


def test_callback_attaches_the_race_and_writes_the_result_and_the_chosen_decode(tmp_path, monkeypatch):
    ref_file = tmp_path / "ref.py"
    ref_file.write_text("{\n" + "".join(f"{tag!r}: {text!r},\n" for tag, text in _refs().items()) + "}\n")
    monkeypatch.chdir(tmp_path)

    class _Model:
        pass

    model = _Model()
    race_opts = {"candidates": [0.0, 0.3, 0.5, 1.0], "ref": str(ref_file), "first_look": 64}
    callback = ScaleRaceForwardCallback(lambda _model: _decode, race_opts)
    assert active_race(model) is None
    callback.init(model=model)
    race = active_race(model)
    assert race is not None and race.first_look == 64 and race.candidates == [0.0, 0.3, 0.5, 1.0]
    for start in range(0, N_SEQS, BATCH):
        tags = [f"u{i}" for i in range(start, start + BATCH)]
        race.run_batch(seq_tags=tags, search=_Search(tags))
        for tag in tags:
            callback.process_seq(seq_tag=tag, outputs=None)
    callback.finish()

    assert active_race(model) is None
    result = json.loads((tmp_path / "race_result.json").read_text())
    assert result["chosen"] == 0.3 and result["survivors"] == [0.3, 0.5]
    assert [look["seqs"] for look in result["looks"]] == [64, 128, 256, 512]
    hyps = ast.literal_eval((tmp_path / "search_out.py").read_text())
    assert len(hyps) == N_SEQS and all(text == REF for text in hyps.values())
