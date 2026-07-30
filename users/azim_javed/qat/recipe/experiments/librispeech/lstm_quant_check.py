import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        batch_first=True,
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


def test_vanilla_training_parity(num_epochs=300, batches_per_epoch=4, batch_size=8, seq_len=16, input_size=32, hidden_size=64, num_layers=2, device="cpu"):
    device = torch.device(device)
    print(f"Starting training test on {device}...")
    torch.manual_seed(42)
    ref = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
    ref.train()

    qlstm = LSTMQuant(
        input_size, hidden_size, num_layers,
        weight_bit_prec=8, weight_quant_dtype=torch.qint8, weight_quant_method="per_tensor",
        activation_bit_prec=8, activation_quant_dtype=torch.quint8, activation_quant_method="per_tensor",
        batch_first=True, dropout=0.0,
        quantization_flags=[False, False, False, False, False],
    ).to(device)
    qlstm.train()

    # Match weights
    sd = {}
    for l in range(num_layers):
        sd[f"w_ih.{l}.weight"] = ref.state_dict()[f"weight_ih_l{l}"].clone()
        sd[f"w_hh.{l}.weight"] = ref.state_dict()[f"weight_hh_l{l}"].clone()
        sd[f"w_ih.{l}.bias"] = ref.state_dict()[f"bias_ih_l{l}"].clone()
        sd[f"w_hh.{l}.bias"] = ref.state_dict()[f"bias_hh_l{l}"].clone()
    qlstm.load_state_dict(sd, strict=False)

    torch.manual_seed(123)
    W_t = torch.randn(input_size, hidden_size, device=device)
    b_t = torch.randn(hidden_size, device=device)

    rng = torch.Generator(device=device).manual_seed(0)
    def make_batch():
        x = torch.randn(batch_size, seq_len, input_size, generator=rng, device=device)
        y = x @ W_t + b_t
        return x, y

    eval_x = torch.randn(batch_size, seq_len, input_size, generator=torch.Generator(device=device).manual_seed(999), device=device)

    # Strictly align q_params order with ref.parameters()
    # PyTorch parameter order for LSTM: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, weight_ih_l1...
    q_params = []
    for l in range(num_layers):
        q_params.append(qlstm.w_ih[l].weight)
        q_params.append(qlstm.w_hh[l].weight)
        q_params.append(qlstm.w_ih[l].bias)
        q_params.append(qlstm.w_hh[l].bias)

    opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-3)
    opt_q = torch.optim.Adam(q_params, lr=1e-3)

    max_out_diff = 0.0
    max_param_diff = 0.0
    for epoch in range(num_epochs):
        for _ in range(batches_per_epoch):
            x, y = make_batch()
            
            opt_ref.zero_grad()
            opt_q.zero_grad()

            r_out, _ = ref(x)
            q_out, _ = qlstm(x)

            loss_ref = nn.functional.mse_loss(r_out, y)
            loss_q = nn.functional.mse_loss(q_out, y)

            loss_ref.backward()
            loss_q.backward()

            opt_ref.step()
            opt_q.step()

        with torch.no_grad():
            r_eval, _ = ref(eval_x)
            q_eval, _ = qlstm(eval_x)
            out_diff = (r_eval - q_eval).abs().max().item()

        ref_sd = ref.state_dict()
        param_diff = 0.0
        for l in range(num_layers):
            param_diff = max(param_diff, (qlstm.w_ih[l].weight - ref_sd[f"weight_ih_l{l}"]).abs().max().item())
            param_diff = max(param_diff, (qlstm.w_hh[l].weight - ref_sd[f"weight_hh_l{l}"]).abs().max().item())
            param_diff = max(param_diff, (qlstm.w_ih[l].bias - ref_sd[f"bias_ih_l{l}"]).abs().max().item())
            param_diff = max(param_diff, (qlstm.w_hh[l].bias - ref_sd[f"bias_hh_l{l}"]).abs().max().item())
        print(f"epoch {epoch+1}/{num_epochs}: out_diff={out_diff:.3e} param_diff={param_diff:.3e}") 
        max_out_diff = max(max_out_diff, out_diff)
        max_param_diff = max(max_param_diff, param_diff)

    with torch.no_grad():
        r_final, _ = ref(eval_x)
        q_final, _ = qlstm(eval_x)
        
    cos = nn.functional.cosine_similarity(r_final.flatten(), q_final.flatten(), dim=0).item()
    print(f"final cosine={cos:.6f} max_out_diff={max_out_diff:.3e} max_param_diff={max_param_diff:.3e} device={device}")
    ok = cos > 0.999 and max_param_diff < 1e-3
    print(f"{'PASSED' if ok else 'FAILED'}")
    return ok

class ManualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.w_ih = nn.ModuleList()
        self.w_hh = nn.ModuleList()
        for l in range(num_layers):
            in_f = input_size if l == 0 else hidden_size
            self.w_ih.append(nn.Linear(in_f, 4 * hidden_size, bias=True))
            self.w_hh.append(nn.Linear(hidden_size, 4 * hidden_size, bias=True))

    def forward(self, input, state=None):
        if not self.batch_first:
            input = input.transpose(0, 1)
        B, T, _D = input.shape
        device, dtype = input.device, input.dtype
        if state is None:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device, dtype=dtype)
            c_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device, dtype=dtype)
        else:
            h_0, c_0 = state
        h_lst = [h_0[l] for l in range(self.num_layers)]
        c_lst = [c_0[l] for l in range(self.num_layers)]
        outputs = []
        for t in range(T):
            x = input[:, t, :]
            for l in range(self.num_layers):
                ih = self.w_ih[l](x)
                hh = self.w_hh[l](h_lst[l])
                gates = ih + hh
                i, f, g, o = gates.chunk(4, dim=-1)
                i = torch.sigmoid(i); f = torch.sigmoid(f)
                g = torch.tanh(g); o = torch.sigmoid(o)
                c = f * c_lst[l] + i * g
                h = o * torch.tanh(c)
                h_lst[l], c_lst[l] = h, c
                x = h
            outputs.append(h_lst[-1])
        output = torch.stack(outputs, dim=1)
        h_n = torch.stack(h_lst, dim=0)
        c_n = torch.stack(c_lst, dim=0)
        return output, (h_n, c_n)


def test_vanilla_training_parity_manual(num_epochs=300, batches_per_epoch=4, batch_size=8, seq_len=16, input_size=32, hidden_size=64, num_layers=2, device="cpu"):
    """Bit-parity between LSTMQuant[F,F,F,F,F] and a manual-loop reference.

    nn.LSTM uses fused cuDNN/MKLDNN kernels with a different reduction order than
    the time-step loop in LSTMQuant.forward, so it is not bit-identical even on
    CPU. This test uses a manual-loop reference with the same structure as
    LSTMQuant to verify numerical parity of the vanilla (all-flags-off) path.
    """
    device = torch.device(device)
    print(f"Starting manual-loop training parity test on {device}...")
    torch.manual_seed(42)
    ref = ManualLSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
    ref.train()

    qlstm = LSTMQuant(
        input_size, hidden_size, num_layers,
        weight_bit_prec=8, weight_quant_dtype=torch.qint8, weight_quant_method="per_tensor",
        activation_bit_prec=8, activation_quant_dtype=torch.quint8, activation_quant_method="per_tensor",
        batch_first=True, dropout=0.0,
        quantization_flags=[False, False, False, False, False],
    ).to(device)
    qlstm.train()

    sd = {}
    for l in range(num_layers):
        sd[f"w_ih.{l}.weight"] = ref.state_dict()[f"w_ih.{l}.weight"].clone()
        sd[f"w_hh.{l}.weight"] = ref.state_dict()[f"w_hh.{l}.weight"].clone()
        sd[f"w_ih.{l}.bias"] = ref.state_dict()[f"w_ih.{l}.bias"].clone()
        sd[f"w_hh.{l}.bias"] = ref.state_dict()[f"w_hh.{l}.bias"].clone()
    qlstm.load_state_dict(sd, strict=False)

    torch.manual_seed(123)
    W_t = torch.randn(input_size, hidden_size, device=device)
    b_t = torch.randn(hidden_size, device=device)

    rng = torch.Generator().manual_seed(0)
    def make_batch():
        x = torch.randn(batch_size, seq_len, input_size, generator=rng).to(device)
        y = x @ W_t + b_t
        return x, y

    eval_x = torch.randn(batch_size, seq_len, input_size, generator=rng).to(device)

    q_params = [p for l in range(num_layers) for w in (qlstm.w_ih[l], qlstm.w_hh[l]) for p in (w.weight, w.bias)]
    opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-3)
    opt_q = torch.optim.Adam(q_params, lr=1e-3)

    max_out_diff = 0.0
    max_param_diff = 0.0
    for epoch in range(num_epochs):
        for _ in range(batches_per_epoch):
            x, y = make_batch()
            # opt_ref.zero_grad(); opt_q.zero_grad()
            r_out, _ = ref(x); q_out, _ = qlstm(x)
            nn.functional.mse_loss(r_out, y).backward()
            nn.functional.mse_loss(q_out, y).backward()
            opt_ref.step(); opt_q.step()

        with torch.no_grad():
            r_eval, _ = ref(eval_x); q_eval, _ = qlstm(eval_x)
            out_diff = (r_eval - q_eval).abs().max().item()
        ref_sd = ref.state_dict()
        param_diff = 0.0
        for l in range(num_layers):
            param_diff = max(param_diff, (qlstm.w_ih[l].weight - ref_sd[f"w_ih.{l}.weight"]).abs().max().item())
            param_diff = max(param_diff, (qlstm.w_hh[l].weight - ref_sd[f"w_hh.{l}.weight"]).abs().max().item())
            param_diff = max(param_diff, (qlstm.w_ih[l].bias - ref_sd[f"w_ih.{l}.bias"]).abs().max().item())
            param_diff = max(param_diff, (qlstm.w_hh[l].bias - ref_sd[f"w_hh.{l}.bias"]).abs().max().item())
        max_out_diff = max(max_out_diff, out_diff)
        max_param_diff = max(max_param_diff, param_diff)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{num_epochs}: out_diff={out_diff:.3e} param_diff={param_diff:.3e}")

    with torch.no_grad():
        r_final, _ = ref(eval_x); q_final, _ = qlstm(eval_x)
    cos = nn.functional.cosine_similarity(r_final.flatten(), q_final.flatten(), dim=0).item()
    print(f"final cosine={cos:.6f} max_out_diff={max_out_diff:.3e} max_param_diff={max_param_diff:.3e} device={device}")
    ok = cos > 0.999 and max_param_diff < 1e-3
    print(f"{'PASSED' if ok else 'FAILED'}")
    return ok


def test_vanilla_training_parity_behavioral(num_epochs=300, batches_per_epoch=4, batch_size=8, seq_len=16, input_size=32, hidden_size=64, num_layers=2, device="cpu", plot_path=None):
    """Behavioral parity: ref (nn.LSTM) and qlstm loss curves track together.

    Bit-parity is impossible across 300 epochs because nn.LSTM uses fused
    kernels with a different reduction order than LSTMQuant's time-step loop,
    and the LSTM recurrence is chaotic. Instead we assert both models converge
    to similar final loss and their per-epoch losses stay close.
    """
    device = torch.device(device)
    print(f"Starting behavioral parity test on {device}...")
    torch.manual_seed(42)
    ref = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
    ref.train()

    qlstm = LSTMQuant(
        input_size, hidden_size, num_layers,
        weight_bit_prec=8, weight_quant_dtype=torch.qint8, weight_quant_method="per_tensor",
        activation_bit_prec=8, activation_quant_dtype=torch.quint8, activation_quant_method="per_tensor",
        batch_first=True, dropout=0.0,
        quantization_flags=[False, False, False, False, False],
    ).to(device)
    qlstm.train()

    sd = {}
    for l in range(num_layers):
        sd[f"w_ih.{l}.weight"] = ref.state_dict()[f"weight_ih_l{l}"].clone()
        sd[f"w_hh.{l}.weight"] = ref.state_dict()[f"weight_hh_l{l}"].clone()
        sd[f"w_ih.{l}.bias"] = ref.state_dict()[f"bias_ih_l{l}"].clone()
        sd[f"w_hh.{l}.bias"] = ref.state_dict()[f"bias_hh_l{l}"].clone()
    qlstm.load_state_dict(sd, strict=False)

    torch.manual_seed(123)
    W_t = torch.randn(input_size, hidden_size, device=device)
    b_t = torch.randn(hidden_size, device=device)

    rng = torch.Generator().manual_seed(0)
    def make_batch():
        x = torch.randn(batch_size, seq_len, input_size, generator=rng).to(device)
        y = x @ W_t + b_t
        return x, y

    eval_x = torch.randn(batch_size, seq_len, input_size, generator=rng).to(device)

    q_params = [p for l in range(num_layers) for w in (qlstm.w_ih[l], qlstm.w_hh[l]) for p in (w.weight, w.bias)]
    opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-3)
    opt_q = torch.optim.Adam(q_params, lr=1e-3)

    ref_losses, q_losses = [], []
    for epoch in range(num_epochs):
        ep_ref_loss = 0.0; ep_q_loss = 0.0
        for _ in range(batches_per_epoch):
            x, y = make_batch()
            opt_ref.zero_grad(); opt_q.zero_grad()
            r_out, _ = ref(x); q_out, _ = qlstm(x)
            r_loss = nn.functional.mse_loss(r_out, y)
            q_loss = nn.functional.mse_loss(q_out, y)
            r_loss.backward(); q_loss.backward()
            opt_ref.step(); opt_q.step()
            ep_ref_loss += r_loss.item(); ep_q_loss += q_loss.item()
        ref_losses.append(ep_ref_loss / batches_per_epoch)
        q_losses.append(ep_q_loss / batches_per_epoch)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{num_epochs}: ref_loss={ref_losses[-1]:.4e} q_loss={q_losses[-1]:.4e}")

    if plot_path is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ref_losses, label="nn.LSTM (ref)")
        ax.plot(q_losses, label="LSTMQuant[F,F,F,F,F]", linestyle="--")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
        ax.set_title(f"Behavioral parity (vanilla) on {device}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(plot_path, dpi=100)
        print(f"Loss-curve plot saved to {plot_path}")

    final_ref, final_q = ref_losses[-1], q_losses[-1]
    rel = abs(final_ref - final_q) / max(abs(final_ref), abs(final_q), 1e-12)
    max_epoch_gap = max(abs(r - q) for r, q in zip(ref_losses, q_losses))
    print(f"final ref_loss={final_ref:.4e} q_loss={final_q:.4e} rel_diff={rel:.4f} max_epoch_gap={max_epoch_gap:.4e}")
    ok = rel < 0.05 and max_epoch_gap < max(ref_losses[0], q_losses[0]) * 0.1
    print(f"{'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tests = [
        test_numerical_parity,
        test_gradient_flow,
        # test_prep_quant_roundtrip,
        test_8bit_quantized_parity_margin,
        lambda: test_vanilla_training_parity(device=dev),
        # lambda: test_vanilla_training_parity_manual(device=dev),
        # lambda: test_vanilla_training_parity_behavioral(device=dev, plot_path="vanilla_training_parity.png"),
    ]
    n = sum(1 for t in tests if t())
    print(f"\n{n}/{len(tests)} passed")
