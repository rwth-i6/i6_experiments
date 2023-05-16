import time
import torch
from torch import nn
from torch.onnx import export


class Conv1DBlock(torch.nn.Module):

    def __init__(self, in_size, out_size, filter_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_size, out_size, filter_size, padding=filter_size//2)
        self.bn = nn.BatchNorm1d(num_features=out_size)
        self.dropout = dropout

    def forward(self, x):
        """

        :param x: should be [B, C, T]
        :return:
        """
        x = self.conv(x)
        x = nn.functional.relu(x)
        # TODO: does not consider masking!
        x = self.bn(x)
        x = nn.functional.dropout1d(x, p=self.dropout)
        return x


class Model(torch.nn.Module):


    def __init__(self,
                 conv_hidden_size: int,
                 lstm_size: int,
                 speaker_embedding_size: int,
                 dropout: float,
                 target_size: int,
                 epoch: int,
                 step: int,
        ):
        super().__init__()
        self.audio_embedding = nn.Linear(80, conv_hidden_size)
        self.speaker_embedding = nn.Embedding(251, speaker_embedding_size)
        self.convs = nn.Sequential(
            Conv1DBlock(conv_hidden_size + speaker_embedding_size, conv_hidden_size, filter_size=5, dropout=dropout),
            Conv1DBlock(conv_hidden_size, conv_hidden_size, filter_size=5, dropout=dropout),
            Conv1DBlock(conv_hidden_size, conv_hidden_size, filter_size=5, dropout=dropout),
            Conv1DBlock(conv_hidden_size, conv_hidden_size, filter_size=5, dropout=dropout),
            Conv1DBlock(conv_hidden_size, conv_hidden_size, filter_size=5, dropout=dropout),
        )
        self.blstm = nn.LSTM(
            input_size=conv_hidden_size,
            hidden_size=lstm_size,
            bidirectional=True,
            batch_first=True)
        self.final_linear = nn.Linear(2*lstm_size, target_size)
        self.lstm_size = lstm_size
        self.target_size = target_size
        self.dropout = dropout

    def forward(
            self,
            audio_features: torch.Tensor,
            speaker_labels: torch.Tensor,
            audio_features_len: torch.Tensor,
    ):
        speaker_embeddings: torch.Tensor = self.speaker_embedding(torch.squeeze(speaker_labels, dim=1))
        # manually broadcast speaker embeddings to each time step
        speaker_embeddings = torch.repeat_interleave(torch.unsqueeze(speaker_embeddings, 1), audio_features.size()[1], dim=1)  # [B, T, F]
        audio_embedding = self.audio_embedding(audio_features)  # [B, T, F]

        conv_in = torch.concat([speaker_embeddings, audio_embedding], dim=2)  # [B, T, F]
        conv_in = torch.swapaxes(conv_in, 1, 2)  # [B, C, T]
        conv_out = self.convs(conv_in)
        blstm_in = torch.permute(conv_out, dims=(0, 2, 1))  # [B, C, T] -> [B, T, C]

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(blstm_in, audio_features_len.to("cpu"), batch_first=True)
        blstm_packed_out, _  = self.blstm(blstm_packed_in)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(blstm_packed_out, padding_value=0.0, batch_first=True)  # [B, T, C]
        blstm_out = nn.functional.dropout1d(blstm_out, p=self.dropout)
        logits = self.final_linear(blstm_out)  # [B, T, #PHONES]
        log_probs = torch.log_softmax(logits, dim=2)  # [B, T, #PHONES]

        return log_probs

def train_step(*, model: Model, data, run_ctx, **kwargs):
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)

    logprobs = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
        speaker_labels=speaker_labels,
    )

    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))
    ctc_loss = nn.functional.ctc_loss(transposed_logprobs, phonemes, input_lengths=audio_features_len, target_lengths=phonemes_len,
                                      blank=model.target_size-1, reduction="sum")
    num_frames = torch.sum(phonemes_len)
    run_ctx.mark_as_loss(name="ctc_sum", loss=ctc_loss, inv_norm_factor=None)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, scale=0.0, inv_norm_factor=num_frames)
