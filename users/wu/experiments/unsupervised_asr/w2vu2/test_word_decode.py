from .word_decode import packed_decode_agreement


def test_packed_decode_agreement_checks_order_and_hypotheses():
    banked = {"u0": "A", "u1": "B", "u2": "C", "u3": "D", "u4": "E"}
    assert packed_decode_agreement(
        packed={"u0": "A", "u2": "C", "u4": "E"},
        banked=banked,
        total_shards=2,
        shard=0,
    )["exact_match"]
    wrong_order = packed_decode_agreement(
        packed={"u2": "C", "u0": "A", "u4": "E"},
        banked=banked,
        total_shards=2,
        shard=0,
    )
    assert not wrong_order["ordered_coverage"]
    wrong_hyp = packed_decode_agreement(
        packed={"u0": "A", "u2": "X", "u4": "E"},
        banked=banked,
        total_shards=2,
        shard=0,
    )
    assert wrong_hyp["hypothesis_mismatches"] == 1
    assert not wrong_hyp["exact_match"]
