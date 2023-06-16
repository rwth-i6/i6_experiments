import time
import torch
from torch import nn
from typing import Dict
from torch.onnx import export


from returnn.tensor import Tensor, Dim, batch_dim
from returnn import frontend as rf

feature_time_dim = Dim(None, name="feature_time")
phoneme_time_dim = Dim(None, name="phoneme_time")
speaker_time_dim = Dim(None, name="speaker_time")
in_dim = Dim(80, name="in")
out_dim = Dim(44, name="out")
spk_dim = Dim(1, name="spk")

def get_extern_data(**kwargs):

    extern_data = {
        "audio_features": {"dims": (batch_dim, feature_time_dim, in_dim), "dtype": "float32"},
        "phonemes": {"dims": (batch_dim, phoneme_time_dim), "sparse_dim": out_dim, "dtype": "int32"},
        "speaker_labels": {"dims": (batch_dim, spk_dim), "dtype": "int32"},
    }
    return extern_data




class Conv1DBlock(rf.Module):

    def __init__(self, in_dim, out_dim, filter_size, dropout):
        super().__init__()
        self.out_dim = out_dim
        self.conv = rf.Conv1d(in_dim=in_dim, out_dim=out_dim, filter_size=filter_size, padding="same")
        # self.bn = rf.BatchNorm(in_dim=out_dim, epsilon=1e-5, use_mask=True)
        self.dropout = dropout

    def __call__(self, x: Tensor, spatial_dim: Dim):
        x, out_spatial_dim = self.conv(x, in_spatial_dim=spatial_dim)
        x = rf.relu(x)
        # x = self.bn(x)
        # x = rf.dropout(x, drop_prob=0.1, axis=x.dims)  # does this broadcast also over batch?
        return x, out_spatial_dim

class Model(rf.Module):

    def __init__(self, conv_hidden_size: int, lstm_size: int, target_size: int, **kwargs):
        super().__init__()
        self.conv_hidden_size = rf.Dim(name="conv_hidden", dimension=conv_hidden_size)
        self.audio_embedding = rf.Linear(in_dim, out_dim=self.conv_hidden_size)

        out_dims = [
            rf.Dim(name="conv_dim_%s" % str(x), dimension=conv_hidden_size)
            for x in range(5)
        ]

        sequential_list = []
        temp_in_dim = self.conv_hidden_size
        for x in range(5):
            sequential_list.append(
                Conv1DBlock(
                    in_dim=temp_in_dim,
                    out_dim=out_dims[x],
                    filter_size=5,
                    dropout=0.1,
                )
            )
            temp_in_dim = out_dims[x]

        self.convs = rf.Sequential(sequential_list)
        #self.blstm = rf.LSTM(
        #    input_size=conv_hidden_size,
        #    hidden_size=lstm_size,
        #    bidirectional=True,
        #    batch_first=False)
        self.out_dim = rf.Dim(name="target_out", dimension=target_size)
        self.final_linear = rf.Linear(out_dims[-1], self.out_dim)
        self.lstm_size = lstm_size

    def __call__(
            self,
            audio_features: Tensor,
            audio_features_time_dim: Dim,
    ):

        for dim in audio_features.dim_tags:
            print(dim.get_dim_value())
        audio_embedding = self.audio_embedding(audio_features)
        conv_out, _ = self.convs((audio_embedding, audio_features_time_dim))
        softmax_in = self.final_linear(conv_out)

        log_probs = rf.log_softmax(softmax_in, axis=self.out_dim, use_mask=True)

        return log_probs

def train_step(*, model: Model, extern_data: Dict[str, Tensor], **_kwargs):
    features = extern_data["audio_features"]
    speaker_labels = extern_data["speaker_labels"]
    logprobs = model(features, feature_time_dim)

    phonemes = extern_data["phonemes"]


    raw_phonemes = phonemes.raw_tensor
    raw_logprobs = logprobs.raw_tensor

    logprobs_len = feature_time_dim.dyn_size
    phonemes_len = phoneme_time_dim.dyn_size

    print(logprobs_len)
    print(phonemes_len)
    print(logprobs_len.size())
    print(phonemes_len.size())
    print(raw_logprobs.size())


    loss = torch.nn.functional.ctc_loss(raw_logprobs.transpose(0, 1), raw_phonemes, input_lengths=logprobs_len, target_lengths=phonemes_len, blank=43)

    rf.get_run_ctx().mark_as_loss(loss, name="ce")
