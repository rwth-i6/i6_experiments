"""Tests of ``training/schedules.py``: the budget schedules at the phase's N = 20, the E60 extension
and the three phone-trigram modes, against the lists the banked configs state
(``PackedBlankfreeTrainJob.2j2bZo1TrkC0`` k2lat_20_ma3000, ``PackedBlankfreeContinueJob.xsWAIaMMsWOw``
E60, ``PackedBlankfreeTrainJob.SZO7xTfO9x7Y`` k2lat_20_ma3000_rp)."""

import pytest

from i6_experiments.users.wu.experiments.unsupervised_asr.training import schedules as S

#: the banked N = 20 lists (written returnn.config of k2lat_20_ma3000 / ctrl_20)
BANKED_TAU_20 = [8.0, 5.039684199579493, 3.174802103936399] + [2.0] * 17
BANKED_LR_20 = [1e-05] + [0.0001] * 11 + [
    8.875e-05,
    7.75e-05,
    6.625000000000001e-05,
    5.5e-05,
    4.375e-05,
    3.2500000000000004e-05,
    2.125e-05,
    1e-05,
]
#: D15's banked prior_weight_schedule (k2lat_20_ma3000_rp)
BANKED_RAMPOUT_20 = [1.0] * 7 + [0.6666666666666667, 0.33333333333333337] + [0.0] * 11


def test_budget_schedules_at_20_match_banked():
    assert S.budget_temperature_schedule(20) == BANKED_TAU_20
    assert S.budget_learning_rates(20, peak=1e-4, warm_frac=S.LR_WARM_FRAC) == BANKED_LR_20


def test_phase_schedules_20():
    tau, lr = S.phase_schedules(20)
    assert tau == BANKED_TAU_20 and lr == BANKED_LR_20


def test_phase_schedules_60_is_20_plus_40_held():
    tau, lr = S.phase_schedules(60)
    assert len(tau) == len(lr) == 60
    assert tau[:20] == BANKED_TAU_20 and lr[:20] == BANKED_LR_20
    assert tau[20:] == [2.0] * 40 and lr[20:] == [0.0001] * 40


def test_phase_schedules_refuses_other_lengths():
    with pytest.raises(ValueError):
        S.phase_schedules(30)


def test_extend_schedule_guards():
    with pytest.raises(AssertionError):
        S.extend_schedule([8.0, 2.0], [1e-4, 1e-4], 2)  # adds nothing
    with pytest.raises(AssertionError):
        S.extend_schedule([8.0, 3.0], [1e-4, 1e-4], 4)  # does not end at the anneal's 2.0
    with pytest.raises(AssertionError):
        S.extend_schedule([8.0, 2.0], [1e-5, 5e-5], 4)  # peak is not the bed's 1e-4


def test_budget_guards():
    with pytest.raises(ValueError):
        S.budget_learning_rates(20, warm_frac=0.05)  # one warmup entry
    with pytest.raises(ValueError):
        S.budget_temperature_schedule(1)


def test_phone_trigram_full_and_off():
    assert S.phone_trigram_weight_schedule("full", 20, onset=8, ramp=3, full_lam=1.0) == [1.0] * 20
    assert S.phone_trigram_weight_schedule("off", 60, onset=8, ramp=3, full_lam=1.0) == [0.0] * 60


def test_phone_trigram_rampout_matches_d15():
    assert S.phone_trigram_weight_schedule("rampout", 20, onset=8, ramp=3, full_lam=1.0) == BANKED_RAMPOUT_20
    s60 = S.phone_trigram_weight_schedule("rampout", 60, onset=8, ramp=3, full_lam=1.0)
    assert s60 == BANKED_RAMPOUT_20 + [0.0] * 40


def test_phone_trigram_unknown_mode():
    with pytest.raises(ValueError):
        S.phone_trigram_weight_schedule("half", 20, onset=8, ramp=3, full_lam=1.0)
