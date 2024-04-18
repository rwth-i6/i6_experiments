import torch
from torch import nn
import whisper
import returnn.frontend as rf
from torch.onnx import export as onnx_export
from whisper.audio import N_FRAMES

class Model(nn.Module):

    def __init__(self, epoch, step, **kwargs):
        super().__init__()

        self.whisper_name = kwargs.pop("whisper_model", "base")
        self.whisper = whisper.load_model(self.whisper_name, "cuda" if torch.cuda.is_available() else 'cpu', "/work/asr4/hilmes/debug/whisper")
        self.just_encoder = kwargs.pop("just_encoder", False)
        self.finetune_whisper = kwargs.pop("finetune_whisper", 0)
        self.split_seq = kwargs.pop("split_seq", True)


        self.upsampling_layer = torch.nn.ConvTranspose1d(
            in_channels=self.whisper.dims.n_audio_state, out_channels=512, kernel_size=5, stride=2, padding=1
        )
        self.final_layer = nn.Linear(512, 9001)


    def forward(self, audio_mel_features: torch.Tensor):
        for param in self.whisper.parameters():
              param.requires_grad_(False)
        if self.finetune_whisper is not False:  # legacy compatibility for now
            for layer_num in range(self.finetune_whisper):
                for param in self.whisper.encoder.blocks[-layer_num].parameters():
                    param.requires_grad_(True)
        trans_audio_mel_features = torch.transpose(audio_mel_features, 1, 2)
        if self.split_seq:
            pad_widths = [(0, 0)] * trans_audio_mel_features.ndim
            pad_widths[-1] = (0, 2 * N_FRAMES - trans_audio_mel_features.shape[-1])
            padded_features = nn.functional.pad(trans_audio_mel_features, [pad for sizes in pad_widths[::-1] for pad in sizes])
            trans_audio_mel_features_1 = padded_features.index_select(
                dim=-1, index=torch.arange(end=N_FRAMES, device=trans_audio_mel_features.device)
            )
            trans_audio_mel_features_2 = padded_features.index_select(
                dim=-1, index=torch.arange(
                    start=N_FRAMES,
                    end=2*N_FRAMES,
                    device=trans_audio_mel_features.device
                )
            )
            if self.just_encoder:
                x_1: torch.Tensor = self.whisper.encoder(trans_audio_mel_features_1)
                if torch.is_nonzero(torch.count_nonzero(trans_audio_mel_features_2)):
                    x_2: torch.Tensor = self.whisper.encoder(trans_audio_mel_features_2)
            else:
                x_1: torch.Tensor = self.whisper(trans_audio_mel_features_1)
                if torch.is_nonzero(torch.count_nonzero(trans_audio_mel_features_2)):
                    x_2: torch.Tensor = self.whisper(trans_audio_mel_features_2)
            x_1 = self.upsampling_layer(x_1.transpose(1, 2)).transpose(1, 2)
            if torch.is_nonzero(torch.count_nonzero(trans_audio_mel_features_2)):
                x_2 = self.upsampling_layer(x_2.transpose(1, 2)).transpose(1, 2)
                x = torch.cat((x_1, x_2), dim=1)
            else:
                x = x_1
        else:
            pad_widths = [(0, 0)] * trans_audio_mel_features.ndim
            pad_widths[-1] = (0, 2 * N_FRAMES - trans_audio_mel_features.shape[-1])
            padded_features = nn.functional.pad(trans_audio_mel_features, [pad for sizes in pad_widths[::-1] for pad in sizes])
            if self.just_encoder:
                x: torch.Tensor = self.whisper.encoder(padded_features)
            else:
                x: torch.Tensor = self.whisper(padded_features)
            x = self.upsampling_layer(x.transpose(1, 2)).transpose(1, 2)
            
        x = x[:, :audio_mel_features.size()[1], :]
        logits = self.final_layer(x)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


def train_step(*, model: Model, extern_data, **_kwargs):
    global scripted_model
    audio_features = extern_data["data"].raw_tensor
    phonemes = extern_data["classes"].raw_tensor
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor
    log_probs, logits = model(
        audio_mel_features=audio_features,
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)
    targets_masked = targets_masked.long()

    loss = nn.functional.cross_entropy(logits, targets_masked)

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export(*, model: Model, model_filename: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 3500, 80, device="cpu")
    scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data,))
    onnx_export(
        scripted_model,
        (dummy_data,),
        f=model_filename,
        verbose=True,
        input_names=["data"],
        output_names=["classes"],
        opset_version=14,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "classes": {0: "batch", 1: "time"},
        },
    )