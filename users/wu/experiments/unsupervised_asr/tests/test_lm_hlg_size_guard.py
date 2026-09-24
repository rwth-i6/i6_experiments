"""The HLG size guard's rung selection, replayed on the banked official 4-gram build.

Banked ``LexlatOfficialHLGBuildJob.a3vpyag6WCBy`` (official 4-gram, ladder 0 / 0.5 / 2 / 5 / ...,
200 GB) skipped rungs 0, 0.5 and 2 on the size guard and built at theta = 5.0.  Its ``build.json``
records, per rung, the G arc count and the predicted peak below; the budget was 0.8 x 200 = 160 GiB.

The test drives the ported job's REAL ladder loop (``LexlatOfficialHLGBuildJob.run`` as wired by
``get_hlg("official_4gram")``) with the k2 child replaced by a stand-in that knows only the banked
G arc count of each rung.  The stand-in prices a rung with the real
``lexlat_k2_official.predict_hlg_cost`` and applies the child's own rule (skip, with exit code
``EXIT_SIZE_GUARD``, iff the predicted peak is over ``--mem-guard-gib``).  No k2, no ARPA, CPU only.
"""

import json
import subprocess
import types

import pytest

from i6_experiments.users.wu.experiments.unsupervised_asr.lm import hlg
from i6_experiments.users.wu.experiments.unsupervised_asr.lm import lexlat_k2_official as O

# a3vpyag6WCBy/output/build.json, attempts[*].build.size_guard: theta -> (g_arcs, predicted_peak_gib)
BANKED_OFF4 = {
    0.0: (308819148, 390.7875056001188),
    0.5: (295758819, 374.26063728485843),
    2.0: (151342762, 191.512931874977),
    5.0: (11294986, 14.29295894802567),
}
BANKED_BUDGET_GIB = 160.0
BANKED_CHOSEN_THETA = 5.0


def test_predictor_reproduces_the_banked_peaks():
    for theta, (g_arcs, peak) in BANKED_OFF4.items():
        assert O.predict_hlg_cost(g_arcs)["predicted_peak_gib"] == pytest.approx(peak, rel=1e-12), theta


def _fake_child(calls):
    def run(cmd, env=None, timeout=None, **kw):
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        args = dict(zip(cmd[4::2], cmd[5::2]))
        sub = cmd[3]
        calls.append((sub, args.get("--prune-theta"), args.get("--mem-guard-gib")))
        if sub == "arpa-to-npz":
            open(args["--out-npz"], "wb").close()
            with open(args["--out-json"], "w") as fh:
                json.dump({"order": 4, "n_arcs": 0, "n_states": 0}, fh)
            return types.SimpleNamespace(returncode=0)
        assert sub == "build-hlg"
        g_arcs = BANKED_OFF4[float(args["--prune-theta"])][0]
        guard = dict(O.predict_hlg_cost(g_arcs), g_arcs=g_arcs,
                     mem_guard_gib=float(args["--mem-guard-gib"]))
        skipped = guard["predicted_peak_gib"] > guard["mem_guard_gib"]
        with open(args["--out-json"], "w") as fh:
            json.dump({"size_guard": guard, "skipped_by_size_guard": skipped, "order": 4}, fh)
        if skipped:
            return types.SimpleNamespace(returncode=O.EXIT_SIZE_GUARD)
        open(args["--out-hlg"], "wb").close()
        return types.SimpleNamespace(returncode=0)

    return run


def test_off4_ladder_skips_three_rungs_and_builds_at_theta_5(tmp_path, monkeypatch):
    from sisyphus import tk

    job = hlg.get_hlg("official_4gram")["hlg"].creator
    # sisyphus caches the job instance: redirect its outputs through monkeypatch, restored after
    for attr, fname in (("out_hlg", "out_HLG.pt"), ("out_stats", "build.json"), ("out_lm", "out_lm.json")):
        monkeypatch.setattr(job, attr, tk.Path(str(tmp_path / fname)))
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_child(calls))
    monkeypatch.setattr(job, "_write_summary", lambda *a, **k: None)  # rendering only
    job.run()

    stats = json.load(open(tmp_path / "build.json"))
    assert stats["mem_guard_gib"] == BANKED_BUDGET_GIB
    assert [c[0] for c in calls] == ["arpa-to-npz"] + ["build-hlg"] * 4
    assert [float(c[2]) for c in calls[1:]] == [BANKED_BUDGET_GIB] * 4
    assert [a["theta_nats"] for a in stats["attempts"]] == [0.0, 0.5, 2.0, 5.0]
    assert [bool(a.get("skipped_by_size_guard")) for a in stats["attempts"]] == [True, True, True, False]
    for a in stats["attempts"]:
        peak = a["build"]["size_guard"]["predicted_peak_gib"]
        assert peak == pytest.approx(BANKED_OFF4[a["theta_nats"]][1], rel=1e-12)
    assert stats["chosen_theta_nats"] == BANKED_CHOSEN_THETA
    assert (tmp_path / "out_HLG.pt").exists()
