import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model_pipelines.common.memristor_layers import LSTMQuant, WeightQuantizer, ActivationQuantizer
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings


def warmup_observers(model, num_batches=5, batch_size=4, seq_len=10, input_size=16):
    model.train()
    with torch.no_grad():
        for _ in range(num_batches):
            model(torch.randn(batch_size, seq_len, input_size))


def test_numerical_parity():
    B, T, D, H, L = 4, 20, 64, 128, 2
    torch.manual_seed(42)

    qlstm = LSTMQuant(D, H, L, 8, torch.qint8, "per_tensor", 8, torch.quint8, "per_tensor")
    for m in qlstm.modules():
        for n, c in list(m.named_children()):
            if isinstance(c, (WeightQuantizer, ActivationQuantizer)):
                setattr(m, n, nn.Identity())
    qlstm.eval()

    ref = nn.LSTM(D, H, L, batch_first=True)
    ref.eval()

    x = torch.randn(B, T, D)
    with torch.no_grad():
        r_out, _ = ref(x)
        q_out, _ = qlstm(x)
    print(f"Random-init max diff: {(r_out - q_out).abs().max().item():.3e}")

    sd = {}
    for l in range(L):
        sd[f"w_ih.{l}.weight"] = ref.state_dict()[f"weight_ih_l{l}"].clone()
        sd[f"w_hh.{l}.weight"] = ref.state_dict()[f"weight_hh_l{l}"].clone()
        sd[f"w_ih.{l}.bias"] = ref.state_dict()[f"bias_ih_l{l}"].clone()
        sd[f"w_hh.{l}.bias"] = ref.state_dict()[f"bias_hh_l{l}"].clone()
    qlstm.load_state_dict(sd, strict=False)

    with torch.no_grad():
        q_out2, _ = qlstm(x)
    d = (r_out - q_out2).abs().max().item()
    print(f"Same-weight max diff: {d:.3e}")
    ok = d < 1e-4
    print(f"{'PASSED' if ok else 'FAILED'}")
    return ok


def test_gradient_flow():
    B, T, D, H, L = 4, 20, 64, 128, 2
    torch.manual_seed(42)

    qlstm = LSTMQuant(D, H, L, 8, torch.qint8, "per_tensor", 8, torch.quint8, "per_tensor")
    for m in qlstm.modules():
        for n, c in list(m.named_children()):
            if isinstance(c, (WeightQuantizer, ActivationQuantizer)):
                setattr(m, n, nn.Identity())
    qlstm.train()

    x = torch.randn(B, T, D, requires_grad=True)
    qlstm(x)[0].sum().backward()

    ok = True
    for p in qlstm.parameters():
        if p.grad is None or p.grad.abs().sum().item() == 0:
            ok = False
    print(f"{'PASSED' if ok else 'FAILED'}")
    return ok


def test_prep_quant_roundtrip():
    B, T, D, H, L = 4, 20, 64, 128, 2
    torch.manual_seed(42)

    hw = DacAdcHardwareSettings(8, 8, 0, 1.0, 1.0)
    qlstm = LSTMQuant(D, H, L, 8, torch.qint8, "per_tensor", 8, torch.quint8, "per_tensor",
                       converter_hardware_settings=hw)

    x = torch.randn(B, T, D)
    qlstm(x)
    qlstm.eval()
    with torch.no_grad():
        before, _ = qlstm(x)

    qlstm.prep_quant()

    for l in range(L):
        assert "TiledMemristorLinear" in type(qlstm.w_ih[l]).__name__
        assert "TiledMemristorLinear" in type(qlstm.w_hh[l]).__name__
        assert isinstance(qlstm.x_in_q[l], nn.Identity)
        assert isinstance(qlstm.h_in_q[l], nn.Identity)
        assert isinstance(qlstm.ih_out_q[l], nn.Identity)
        assert isinstance(qlstm.hh_out_q[l], nn.Identity)
    print("Gates replaced by TiledMemristorLinear")

    with torch.no_grad():
        after, _ = qlstm(x)
    assert before.shape == after.shape
    print(f"Shapes match, max diff: {(before - after).abs().max().item():.3e}")
    print("PASSED")
    return True


def test_8bit_quantized_parity_margin():
    B, T, D, H = 2, 5, 8, 16
    torch.manual_seed(42)

    ref = nn.LSTM(D, H, batch_first=True)

    qlstm = LSTMQuant(
        D, H, 1, 8, torch.qint8, "per_tensor_symmetric",
        8, torch.quint8, "per_tensor",
        batch_first=True, quant_cell_state=False,
    )

    with torch.no_grad():
        qlstm.w_ih[0].weight.copy_(ref.weight_ih_l0)
        qlstm.w_ih[0].bias.copy_(ref.bias_ih_l0)
        qlstm.w_hh[0].weight.copy_(ref.weight_hh_l0)
        qlstm.w_hh[0].bias.copy_(ref.bias_hh_l0)

    warmup_observers(qlstm, num_batches=3, batch_size=B, seq_len=T, input_size=D)

    x = torch.randn(B, T, D)
    ref_out, _ = ref(x)
    q_out, _ = qlstm(x)

    cos = nn.functional.cosine_similarity(ref_out.flatten(), q_out.flatten(), dim=0)
    print(f"Cosine similarity FP32 → 8-bit: {cos.item():.4f}")
    ok = cos.item() > 0.95
    print(f"{'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    tests = [test_numerical_parity, test_gradient_flow, test_prep_quant_roundtrip, test_8bit_quantized_parity_margin]
    n = sum(1 for t in tests if t())
    print(f"\n{n}/{len(tests)} passed")
