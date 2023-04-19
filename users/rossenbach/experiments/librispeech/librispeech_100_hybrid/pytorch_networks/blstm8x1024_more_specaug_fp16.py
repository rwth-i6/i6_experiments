import time
import torch
from torch import nn, autocast
from torch.onnx import export as onnx_export
from torch.cuda.amp import GradScaler
from torchaudio.functional import mask_along_axis

from returnn.log import log
from returnn.torch.engine import Engine as TorchEngine
from returnn.util.basic import NumbersDict
from returnn.torch.context import get_run_ctx, init_train_step_run_ctx


class CustomEngine(TorchEngine):
    
    def train_epoch(self):
        """
        train one (sub)epoch
        """
        print("start", self.get_epoch_str(), "with learning rate", self.learning_rate, "...", file=log.v4)

        self._model.train()
        init_train_step_run_ctx(device=self._device)

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        accumulated_losses_dict = NumbersDict()
        step_idx = 0
        for data in self._train_dataloader:
            step_time_start = time.time()
            run_ctx = get_run_ctx()
            run_ctx.init_step()

            self._updater.get_optimizer().zero_grad()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                self._run_step(data)

            losses_dict = run_ctx.losses
            total_loss = run_ctx.total_loss()

            scaler.scale(total_loss).backward()
            scaler.step(self._updater.get_optimizer())
            scaler.update()

            losses_dict = {
                "train_loss_" + name: float(loss.loss.detach().cpu().numpy())
                for name, loss in losses_dict.items()
            }
            accumulated_losses_dict += NumbersDict(losses_dict)
            print("step %i, loss: %f, took: %.3fs" % (
                step_idx, total_loss.detach().cpu().numpy(), time.time() - step_time_start
            ), file=log.v4)

            step_idx += 1

        print("Trained %i steps" % step_idx)

        accumulated_losses_dict = accumulated_losses_dict / step_idx
        self.learning_rate_control.set_epoch_error(self.epoch, dict(accumulated_losses_dict))
        self.learning_rate_control.save()

        if self.epoch % self._save_model_epoch_interval == 0 or self.epoch == self._final_epoch:
            self._save_model()
            self._save_optimizer()

        self.eval_model()

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        lstm_size = 1024
        target_size=12001
        self.blstm1 = nn.LSTM(input_size=50, hidden_size=lstm_size, bidirectional=True, batch_first=False)
        self.blstm_stack = nn.LSTM(input_size=2*lstm_size, hidden_size=lstm_size, bidirectional=True, num_layers=7, batch_first=False)
        self.final_linear = nn.Linear(2*lstm_size, target_size)
        self.lstm_size = lstm_size

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: torch.Tensor,
    ):
        if self.training:
            audio_features_time_masked = mask_along_axis(audio_features, mask_param=20, mask_value=0.0, axis=1)
            audio_features_time_masked_2 = mask_along_axis(audio_features_time_masked, mask_param=20, mask_value=0.0, axis=1)
            audio_features_masked = mask_along_axis(audio_features_time_masked_2, mask_param=10, mask_value=0.0, axis=2)
            audio_features_masked_2 = mask_along_axis(audio_features_masked, mask_param=10, mask_value=0.0, axis=2)
        else:
            audio_features_masked_2 = audio_features
        blstm_in = torch.swapaxes(audio_features_masked_2, 0, 1)  # [B, T, F] -> [T, B, F]

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(blstm_in, audio_features_len)
        blstm_first, _ = self.blstm1(blstm_packed_in)
        blstm_packed_out, _ = self.blstm_stack(blstm_first)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(blstm_packed_out, padding_value=0.0, batch_first=False)  # [T, B, F]
        
        logits = self.final_linear(blstm_out)  # [T, B, F]
        logits_rasr_order = torch.permute(logits, dims=(1, 0, 2))  # RASR expects [B, T, F]
        logits_ce_order  = torch.permute(logits, dims=(1, 2, 0))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits_rasr_order, dim=2)

        return log_probs, logits_ce_order


scripted_model = None

def train_step(*, model: Model, data, run_ctx, **_kwargs):
    global scripted_model
    audio_features = data["data"]
    audio_features_len = data["data:seq_len"]

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = data["classes"][indices, :]
    phonemes_len = data["classes:seq_len"][indices]

    if scripted_model is None:
        scripted_model = torch.jit.script(model)

    # distributed_model = DataParallel(model)
    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(phonemes, phonemes_len, batch_first=True, enforce_sorted=False)
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    run_ctx.mark_as_loss(name="CE", loss=loss)


def export(*, model: Model, model_filename: str):
    scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    dummy_data = torch.randn(1, 30, 50, device="cpu")
    dummy_data_len, _ = torch.sort(torch.randint(low=10, high=30, size=(1,), device="cpu", dtype=torch.int32), descending=True)
    onnx_export(
        scripted_model,
        (dummy_data, dummy_data_len),
        f=model_filename,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "classes": {0: "batch", 1: "time"}
        }
    )


def export_trace(*, model: Model, model_filename: str):
    dummy_data = torch.randn(1, 30, 50, device="cpu")
    dummy_data_len, _ = torch.sort(torch.randint(low=10, high=30, size=(1,), device="cpu", dtype=torch.int32), descending=True)
    scripted_model = torch.jit.optimize_for_inference(torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len)))
    onnx_export(
        scripted_model,
        (dummy_data, dummy_data_len),
        f=model_filename,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "classes": {0: "batch", 1: "time"}
        }
    )


