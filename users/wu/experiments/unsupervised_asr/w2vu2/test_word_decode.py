import numpy as np
import pytest

from .packed_audio import reconstruct_pcm16
from .word_decode import packed_decode_agreement


def test_reconstruct_pcm16_projects_to_source_lattice():
    half_step = 0.5 / 32768.0
    wave = np.array([-1.1, -0.5 - half_step / 2, half_step / 2, 0.5 + half_step / 2, 1.1])
    got = reconstruct_pcm16(wave)
    np.testing.assert_array_equal(
        got,
        np.array([-1.0, -0.5, 0.0, 0.5, 32767 / 32768], dtype=np.float32),
    )
    assert got.dtype == np.float32


def test_reconstruct_pcm16_rejects_non_mono_or_integer_input():
    with pytest.raises(ValueError):
        reconstruct_pcm16(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        reconstruct_pcm16(np.zeros(3, dtype=np.int16))


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
