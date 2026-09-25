"""Guard: the MTR driver's ``--batch_size`` path writes the same submission as the serial path.

WHY. MTR-DuplexBench inference ran one dialogue at a time (B=1): ~20-24 GPU-h per PersonaPlex model for
~58 h of audio (2026-09-25). ``--batch_size B`` streams B dialogues per forward through the engine's
``reply_batch`` (longest first) and hands each model channel to a per-dialogue ``finish`` callback, and the
round timing is dumped up front. What must hold, so that only the MODEL's random draws can differ:

  1. Every submission file (CF / IF / Safety wavs, DQ per-turn mp3s) and ``round_start.json`` is
     byte-identical to the B=1 path, given a deterministic stand-in model -- i.e. each dialogue's output
     lands in ITS files (no late-binding closure, no row mix-up across a group, no lost partial group).
  2. Groups are at most B, longest first; B=1 never calls ``reply_batch``.
  3. An engine without ``reply_batch`` refuses ``--batch_size`` > 1 instead of silently running B=1.
  Can fail: a mutant ``_run_batched`` that hands a group's outputs back in reverse order is caught.

moshi_family venv (numpy, soundfile, sphn, torch), CPU, a few seconds, from the setup root:
    PYTHONPATH=recipe/speech_llm/full_duplex <moshi_family venv python> \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_mtr_batched_driver.py
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "recipe/speech_llm/full_duplex")

from mtr import driver  # noqa: E402

SR = 24000
RNG = np.random.default_rng(0)


def tone(sec: float) -> np.ndarray:
    n = int(sec * SR)
    return (0.3 * np.sin(np.arange(n) * RNG.uniform(0.01, 0.2)) * RNG.uniform(0.2, 1.0)).astype(np.float32)


class StubModel:
    """Deterministic content-only 'model': output = a shifted, scaled copy of the input."""

    sample_rate = SR
    device = "cpu"

    def __init__(self):
        self.groups = []

    def reply(self, pcm):
        return (0.5 * np.roll(pcm, 1234)).astype(np.float32)

    def reply_batch(self, pcms, batch_size):
        assert 1 <= len(pcms) <= batch_size, (len(pcms), batch_size)
        self.groups.append([len(p) for p in pcms])
        return [self.reply(p) for p in pcms]


class SerialOnly(StubModel):
    reply_batch = None


def build_data(root: Path) -> None:
    d = root / "data"
    # Conversational features (smooth): 5 dialogues x 2-3 rounds of different lengths.
    enc, cf = {}, d / "Conversational_Features/original"
    cf.mkdir(parents=True)
    (d / "Conversational_Features/background").mkdir()
    for did in range(5):
        enc[str(did)] = {}
        for r in range(1, 2 + did % 2 + 1):
            sf.write(
                cf / f"original_dialogue_data_dialogue_{did:03d}_round_{r:02d}.wav", tone(0.5 + 0.3 * did + 0.1 * r), SR
            )
            enc[str(did)][str(r)] = "smooth-turntaking"
    (d / "Scenarios_encoding/single_scenario").mkdir(parents=True)
    (d / "Scenarios_encoding/single_scenario/scenario_encoding_smooth.json").write_text(json.dumps(enc))
    # Instruction following: 4 dialogues x 2-3 rounds.
    ifd = d / "Instrcution_Following/audios"
    ifd.mkdir(parents=True)
    rows = []
    for k in range(4):
        rounds = []
        for r in range(1, 2 + k % 2 + 1):
            name = f"q_{k}_{r}.wav"
            sf.write(ifd / name, tone(0.4 + 0.25 * ((k * 7 + r) % 4)), SR)
            rounds.append({"round": r, "audio_file": f"/your/path/to/{name}", "scenario": "smooth"})
        rows.append({"dialogue_id": f"instruction_following_{k:03d}", "rounds": rounds})
    (d / "Instrcution_Following/instruction_following_data.json").write_text(json.dumps(rows))
    # Dialogue quality: 3 stereo mp3s with 2-3 turns each.
    dq = d / "Dialogue_Quality"
    dq.mkdir(parents=True)
    tj = []
    for k in range(3):
        sid = f"uuid{k}_0.0_{3 + k}.0"
        sf.write(dq / f"{sid}.mp3", np.stack([tone(3 + k), tone(3 + k)], axis=-1), SR)
        tj.append({"id": sid, "filtered_turn": [{"start": 0.5 * t, "end": 0.5 * t + 0.4} for t in range(2 + k % 2)]})
    (d / "Scenarios_encoding/dialogue_quality").mkdir(parents=True)
    (d / "Scenarios_encoding/dialogue_quality/semantic_turned_time_v3.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tj) + "\n"
    )


def run_all(model, root: Path, out: Path, B: int) -> dict:
    driver.run_conv_features(model, root, "smooth", str(out / "cf"), None, None, B)
    driver.run_if_safety(model, root, "instruction_following", str(out / "if"), None, None, B)
    driver.run_dialogue_quality(model, root, str(out / "dq"), None, None, B)
    return {
        str(p.relative_to(out)): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_data(tmp / "mtr")
        serial = StubModel()
        ref = run_all(serial, tmp / "mtr", tmp / "b1", 1)
        assert not serial.groups, "B=1 must not call reply_batch"
        n_files = len(ref)
        assert n_files >= 5 + 10 + 7 + 2, ref
        assert {"cf/round_start.json", "if/round_start.json"} <= set(ref), sorted(ref)

        for B in (2, 3, 16):
            m = StubModel()
            got = run_all(m, tmp / "mtr", tmp / f"b{B}", B)
            assert got == ref, (B, {k for k in ref if got.get(k) != ref[k]})
            assert all(len(g) <= B for g in m.groups), m.groups
            for g in m.groups:
                assert g == sorted(g, reverse=True), m.groups  # longest first within a group
            print(f"[ok] B={B}: {n_files} files identical to B=1; groups {m.groups}")

        try:
            run_all(SerialOnly(), tmp / "mtr", tmp / "serial_only", 4)
        except (ValueError, TypeError) as e:
            print(f"[ok] engine without reply_batch refuses --batch_size 4: {e}")
        else:
            raise AssertionError("an engine without reply_batch accepted --batch_size 4")

        # Mutant: a group's outputs handed back in reverse order must be caught.
        orig = driver._run_batched

        def reversed_group(model, items, batch_size, label):
            order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
            for g in range(0, len(order), batch_size):
                idx = order[g : g + batch_size]
                outs = model.reply_batch([items[i][0] for i in idx], batch_size)
                for i, out in zip(idx, outs[::-1]):
                    items[i][1](out)

        driver._run_batched = reversed_group
        try:
            got = run_all(StubModel(), tmp / "mtr", tmp / "mutant", 3)
        finally:
            driver._run_batched = orig
        assert got != ref, "the reversed-group mutant was NOT caught"
        print("[ok] mutant (outputs reversed within a group) caught")
    print("check_mtr_batched_driver: all passed")


if __name__ == "__main__":
    main()
