"""Tests of ``phones.py``, ``model/recognizer.py``, ``model/recognizer_only.py``,
``model/lexlat_train.py`` and ``model/param_groups.py`` (CPU, seconds).

The numerical equivalence with the speech-llm c49559ce source is not tested here (committed tests
never import it); it was checked once by the port's old-vs-new comparison scripts.
"""

import pytest
import torch

from i6_experiments.users.wu.experiments.unsupervised_asr import phones as P
from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat_train as LT
from i6_experiments.users.wu.experiments.unsupervised_asr.model import param_groups as PG
from i6_experiments.users.wu.experiments.unsupervised_asr.model import recognizer as R
from i6_experiments.users.wu.experiments.unsupervised_asr.model import recognizer_only as RO


def test_phone_inventory():
    assert len(P.ARPABET_39) == 39 and P.PHONES[-1] == P.SIL == "SIL"
    assert P.N_TYPES == 40 and P.SIL_ID == 39 and P.BOS_ID == 40 and P.N_CTX == 41
    assert P.PHONE2ID["AA"] == 0 and P.PHONE2ID["SIL"] == 39


@pytest.mark.parametrize(
    "label, expected", [("AA1", "AA"), ("ZH0", "ZH"), ("sil", "SIL"), ("spn", "SIL"), ("", "SIL"),
                        ("<unk>", "SIL"), ("XX", "SIL"), ("T", "T")])
def test_canonical_phone(label, expected):
    assert P.canonical_phone(label) == expected


def test_emission_space():
    assert R.N_OUT == 41 and R.BLANK_ID == 0 and R.SIL_OUT_ID == 40
    assert R.OUT_SYMBOLS[1:40] == list(P.ARPABET_39) and R.OUT_SYMBOLS[40] == "SIL"
    assert R.emission_index(P.SIL_ID) == R.SIL_OUT_ID


@pytest.mark.parametrize("stride, n_out, n_layers", [(1, 41, 1), (3, 40, 1), (2, 41, 2)])
def test_recognizer_forward_shape(stride, n_out, n_layers):
    torch.manual_seed(0)
    net = R.ConvRecognizer(in_dim=16, n_out=n_out, stride=stride, n_layers=n_layers, hidden=8)
    lens = torch.tensor([10, 7])
    feats = torch.randn(2, 10, 16).half()
    out = net(feats, lens)
    t_out = -(-10 // stride)
    assert out.shape == (2, t_out, n_out) and out.dtype == torch.float32
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, t_out), atol=1e-5)
    assert net.output_lengths(lens).tolist() == [-(-10 // stride), -(-7 // stride)]


def test_recognizer_padding_does_not_leak():
    torch.manual_seed(0)
    net = R.ConvRecognizer(in_dim=16).eval()
    lens = torch.tensor([6, 6])
    feats = torch.randn(2, 9, 16)
    other = feats.clone()
    other[:, 6:] = 100.0
    assert torch.equal(net(feats, lens), net(other, lens))


@pytest.mark.parametrize("kw", [{"content_k": 64}, {"content_layer": "conv"}])
def test_removed_content_head_is_refused(kw):
    with pytest.raises(ValueError, match="content"):
        R.ConvRecognizer(in_dim=16, **kw)


def test_recognizer_only_get_model_drops_runtime_kwargs(capsys):
    net = RO.get_model(epoch=1, step=0, device="cpu", __fwd_compat_sentinel=1,
                       **dict(RO.RECOGNIZER_NET_ARGS, in_dim=16))
    assert isinstance(net, R.ConvRecognizer) and net.n_out == 41 and net.stride == 1
    assert "ignoring kwargs not declared by ConvRecognizer" in capsys.readouterr().out
    assert RO.RECOGNIZER_NET_ARGS == {
        "in_dim": 1024, "n_out": 41, "kernel": 9, "stride": 1, "hidden": 512, "n_layers": 1,
        "dropout": 0.1, "batch_norm": 30.0, "residual": True, "bias": False}


def test_lexlat_lambda_curriculum():
    assert LT.lambda_schedule(12, onset=8, ramp=3) == [0.0] * 7 + [1 / 3, 2 / 3, 1.0, 1.0, 1.0]
    LT.assert_ramp(20, onset=8, ramp=3)
    LT.assert_ramp(20, onset=1, ramp=3)
    with pytest.raises(AssertionError):
        LT.assert_ramp(9, onset=8, ramp=3)  # the ramp does not finish inside the run


class _Two(torch.nn.Module):
    def __init__(self, extra=False, freeze_reverse=False):
        super().__init__()
        self.recognizer = torch.nn.Linear(2, 2)
        self.reverse = torch.nn.Linear(2, 2)
        if extra:
            self.other = torch.nn.Linear(2, 2)
        if freeze_reverse:
            self.reverse.requires_grad_(False)


def test_param_groups():
    groups = PG.emc_param_groups(model=_Two(), recognizer_lr_multiplier=1.0, reverse_lr_multiplier=30.0,
                                 optimizer_class=None)
    assert [g["learning_rate_multiplier"] for g in groups] == [1.0, 30.0]
    assert [len(g["params"]) for g in groups] == [2, 2]
    (only,) = PG.emc_param_groups(model=_Two(freeze_reverse=True), reverse_lr_multiplier=30.0)
    assert only["learning_rate_multiplier"] == 1.0
    with pytest.raises(AssertionError, match="outside recognizer/reverse"):
        PG.emc_param_groups(model=_Two(extra=True))
