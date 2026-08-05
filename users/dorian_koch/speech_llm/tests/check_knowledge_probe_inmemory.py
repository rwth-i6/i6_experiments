"""Guard: the in-loop knowledge probe must not touch the filesystem.

The bug this exists for (2026-08-05): ``run_knowledge_probe`` generated 64 reply wavs plus 64
``.txt`` sidecars into a scratch directory and read the sidecars straight back, purely to move a
string between two lines of the same function. The audio was never looked at. A single transient
Lustre ``EIO`` on one of those writes raised, the probe caught it, disabled itself permanently, and
a8-long ran 5,200 more steps producing no readout while still reporting success.

The retry/fail-loud policy in ``moshi_finetune_launcher`` is the net for that class of failure. This
removes the exposure instead: work that never opens a file cannot fail that way.

Both modes are driven through the REAL ``run_pairs`` with a fake model, rather than asserting on a
reimplementation -- a guard that builds its own idealised caller is how ``check_mixed_loader`` missed
a live DDP bug for months (see CLAUDE.md). The assertion that matters is ``os.listdir(tmp) == []``.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_knowledge_probe_inmemory.py
"""

import os
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from speech_llm.full_duplex.moshi_family.moshi_engine import (  # noqa: E402
    RunPairsResult,
    run_pairs,
)

SR = 24000
N_CLIPS = 5
BATCH = 2
#: Token ids the decoder must drop (pad / epad), mirroring run_pairs' own filter.
DROPPED = (0, 3)


class _FakeTokenizer:
    """Maps a token id to a piece, with the sentencepiece word-boundary marker on word starts."""

    def id_to_piece(self, t: int) -> str:
        return {1: "▁the", 2: "▁answer", 4: "▁is", 5: "▁paris"}.get(t, "?")


class _FakeState:
    def __init__(self, batch_size):
        self.text_tokenizer = _FakeTokenizer()
        self.batch_size = batch_size
        self.mimi = self
        self.lm_gen = self

    def reset_streaming(self):
        pass

    def run(self, in_pcms):
        """Return ``[(text_tokens, audio)]`` per batch row, the shape run_pairs unpacks.

        Row b gets a distinguishable monologue so a positional mix-up between clips is visible, and
        audible tail energy so the truncation branch is exercised on the disk path too.
        """
        bs = in_pcms.shape[0]
        out = []
        for b in range(bs):
            toks = torch.tensor([1, 2, 0, 4, 3, 5 if b % 2 == 0 else 1])
            audio = torch.zeros(1, SR // 2)
            audio[0, -100:] = 0.5  # non-silent tail -> "possibly truncated"
            out.append((toks, audio))
        return out


class _FakeModel:
    def __init__(self, batch_size=BATCH):
        self.sample_rate = SR
        self.batch_size = batch_size
        self.device = "cpu"
        self.state = _FakeState(batch_size)


def _inputs(tmp: Path) -> list:
    """Real wav files on disk -- run_pairs reads its INPUTS with sphn either way."""
    import sphn

    paths = []
    for i in range(N_CLIPS):
        p = tmp / f"in{i}.wav"
        sphn.write_wav(str(p), np.zeros(SR, dtype=np.float32), SR)
        paths.append(str(p))
    return paths


def check_disk_mode_still_writes():
    """The benchmark path is unchanged: one wav + one txt per clip, plus truncation.json."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ins = _inputs(tmp)
        out = tmp / "out"
        pairs = [(p, str(out / f"{i}.wav")) for i, p in enumerate(ins)]
        res = run_pairs(_FakeModel(), pairs, capture_s=0.1, progress_every=0)
        assert isinstance(res, RunPairsResult), type(res)
        assert res.count == N_CLIPS, res
        wavs = sorted(out.glob("*.wav"))
        txts = sorted(out.glob("*.txt"))
        assert len(wavs) == N_CLIPS, wavs
        assert len(txts) == N_CLIPS, txts
        assert (out / "truncation.json").exists(), "the truncation rate must stay queryable"
        # The sidecar content must equal what the caller now gets in memory -- otherwise the
        # in-memory path is scoring something different from what the benchmark path records.
        for i, mono in enumerate(res.monologues):
            assert (out / f"{i}.txt").read_text() == mono, i
    print("PASS  disk mode writes one wav + one txt per clip, and truncation.json")


def check_memory_mode_writes_nothing():
    """out_wav=None: not one byte on disk, and the monologues come back identical."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ins = _inputs(tmp)
        before = sorted(os.listdir(tmp))

        res = run_pairs(
            _FakeModel(), [(p, None) for p in ins], capture_s=0.1, progress_every=0
        )
        assert sorted(os.listdir(tmp)) == before, (
            f"the in-memory path created files: {set(os.listdir(tmp)) - set(before)}. This is the "
            f"exact exposure that disabled a8-long's probe."
        )
        assert res.count == N_CLIPS, res
        assert len(res.monologues) == N_CLIPS, res.monologues

        # Byte-identical to what the disk path produces for the same inputs.
        out = tmp / "out"
        disk = run_pairs(
            _FakeModel(),
            [(p, str(out / f"{i}.wav")) for i, p in enumerate(ins)],
            capture_s=0.1,
            progress_every=0,
        )
        assert res.monologues == disk.monologues, (res.monologues, disk.monologues)
        # ...and non-trivial: dropped ids gone, word markers decoded, clips distinguishable.
        assert res.monologues[0] == "the answer is paris", res.monologues
        assert res.monologues[1] != res.monologues[0], "clips must not be collapsed together"
        assert "?" not in "".join(res.monologues), "an unmapped token leaked into the monologue"
    print("PASS  in-memory mode writes ZERO files and returns the same monologues as disk mode")


def check_truncation_accounting_survives_without_a_directory():
    """The truncation counters are still right with nowhere to write truncation.json."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ins = _inputs(tmp)
        # tail_check_s must be shorter than the fake reply (0.5 s) or the check never fires --
        # run_pairs skips clips shorter than the tail window, which is why the default 4 s would
        # silently make this assertion vacuous.
        res = run_pairs(
            _FakeModel(),
            [(p, None) for p in ins],
            capture_s=0.1,
            tail_check_s=0.01,
            max_truncated_frac=1.0,  # deliberately 100% here; the ceiling is checked below
            progress_every=0,
        )
        # Every fake clip has a non-silent tail, so every one counts as truncated.
        assert res.truncated == N_CLIPS, res
        assert abs(res.truncated_fraction - 1.0) < 1e-9, res

        # ...and the pathological-rate ceiling still fires without a directory to report into.
        # It is the only thing standing between "verbose model" and "capture_s is simply wrong",
        # and it must not be quietly skipped just because nothing is being written.
        try:
            run_pairs(
                _FakeModel(),
                [(p, None) for p in ins],
                capture_s=0.1,
                tail_check_s=0.01,
                progress_every=0,
            )
        except AssertionError as e:
            assert "capture window" in str(e), e
        else:
            raise SystemExit("FAIL: a 100% truncation rate did not trip max_truncated_frac")
    print("PASS  truncation counters and the pathological-rate ceiling survive in-memory mode")


def check_probe_scores_positionally_against_its_own_entries():
    """run_knowledge_probe zips monologues to entries by POSITION, so lengths must be asserted."""
    import inspect

    from speech_llm.full_duplex.moshi_family import knowledge_probe

    src = inspect.getsource(knowledge_probe.run_knowledge_probe)
    assert "out_dir" not in src, "out_dir is gone; the probe must not name a scratch directory"
    assert "len(monologues) == len(entries)" in src, (
        "the positional zip between monologues and entries must be asserted -- a mismatch would "
        "score every clip against the wrong gold answer and still report a plausible accuracy"
    )
    sig = inspect.signature(knowledge_probe.run_knowledge_probe)
    assert "out_dir" not in sig.parameters, sig
    print("PASS  run_knowledge_probe takes no out_dir and asserts its positional join")


if __name__ == "__main__":
    check_disk_mode_still_writes()
    check_memory_mode_writes_nothing()
    check_truncation_accounting_survives_without_a_directory()
    check_probe_scores_positionally_against_its_own_entries()
    print("ALL PASS")
