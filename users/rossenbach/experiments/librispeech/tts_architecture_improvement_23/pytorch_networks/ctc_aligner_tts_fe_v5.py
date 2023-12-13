"""
Trying to make the aligner more TF-like

Also deterministic CTC
"""

from dataclasses import dataclass
import torch
import numpy
from torch import nn
import multiprocessing
from librosa import filters
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union


def _lengths_to_op_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to operation compatible key mask

    :param lengths: [B]
    :return: B x T, where 1 means within sequence and 0 means outside sequence
    """
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    return padding_mask


@dataclass
class DbMelFeatureExtractionConfig():
    """

    :param sample_rate: audio sample rate in Hz
    :param win_size: window size in seconds
    :param hop_size: window shift in seconds
    :param f_min: minimum mel filter frequency in Hz
    :param f_max: maximum mel fitler frequency in Hz
    :param min_amp: minimum amplitude for safe log
    :param num_filters: number of mel windows
    :param center: centered STFT with automatic padding
    :param norm: tuple optional of mean & std_dev for feature normalization
    """
    sample_rate: int
    win_size: float
    hop_size: float
    f_min: int
    f_max: int
    min_amp: float
    num_filters: int
    center: bool
    norm: Optional[Tuple[float, float]] = None

    @classmethod
    def from_dict(cls, d):
        return DbMelFeatureExtractionConfig(**d)


class DbMelFeatureExtraction(nn.Module):

    def __init__(
            self,
            config: DbMelFeatureExtractionConfig
    ):
        super().__init__()
        self.register_buffer("n_fft", torch.tensor(int(config.win_size * config.sample_rate)))
        self.register_buffer("hop_length", torch.tensor(int(config.hop_size * config.sample_rate)))
        self.register_buffer("min_amp", torch.tensor(config.min_amp))
        self.center = config.center
        if config.norm is not None:
            self.apply_norm = True
            self.register_buffer("norm_mean", torch.tensor(config.norm[0]))
            self.register_buffer("norm_std_dev", torch.tensor(config.norm[1]))
        else:
            self.apply_norm = False

        self.register_buffer("mel_basis", torch.tensor(filters.mel(
            sr=config.sample_rate,
            n_fft=int(config.sample_rate * config.win_size),
            n_mels=config.num_filters,
            fmin=config.f_min,
            fmax=config.f_max)))
        self.register_buffer("window", torch.hann_window(int(config.win_size * config.sample_rate)))

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length in samples: [B]
        :return features as [B,T,F] and length in frames [B]
        """

        S = torch.abs(torch.stft(
            raw_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            pad_mode="constant",
            return_complex=True,
        )) ** 2
        if len(S.size()) == 2:
            # For some reason torch.stft "eats" batch sizes of 1, so we need to add it again if needed
            S = torch.unsqueeze(S, 0)
        melspec = torch.einsum("...ft,mf->...mt", S, self.mel_basis)
        melspec = 20 * torch.log10(torch.max(self.min_amp, melspec))
        feature_data = torch.transpose(melspec, 1, 2)

        if self.apply_norm:
            feature_data = (feature_data - self.norm_mean) / self.norm_std_dev

        if self.center:
            length = (length // self.hop_length) + 1
        else:
            length = ((length - self.n_fft) // self.hop_length) + 1

        return feature_data, length.int()



class Conv1DBlock(torch.nn.Module):
    """
    A 1D-Convolution with ReLU, batch-norm and non-broadcasted dropout
    Will pad to the same output length
    """

    def __init__(self, in_size, out_size, filter_size, dropout):
        """
        :param in_size: input feature size
        :param out_size: output feature size
        :param filter_size: filter size
        :param dropout: dropout probability
        """
        super().__init__()
        assert filter_size % 2 == 1, "Only odd filter sizes allowed"
        self.conv = nn.Conv1d(in_size, out_size, filter_size, padding=filter_size // 2)
        self.bn = nn.BatchNorm1d(num_features=out_size)
        self.dropout = dropout

    def forward(self, x):
        """
        :param x: [B, F_in, T]
        :return: [B, F_out, T]
        """
        x = self.conv(x)
        x = nn.functional.relu(x)
        # TODO: does not consider masking!
        x = self.bn(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

@dataclass
class TTSPredictorConfig:
    hidden_dim: int
    dropout: float
    speaker_embedding_size: int

    @classmethod
    def from_dict(cls, d):
        return TTSPredictorConfig(**d)


class TTSPredictor(torch.nn.Module):
    """
    Predictor for the reconstruction Loss
    """
    def __init__(self, config: TTSPredictorConfig, input_size: int, target_size: int):
        """

        :param input_size:
        :param target_size:
        """
        super().__init__()
        self.input_size = input_size

        self.speaker_embedding = nn.Embedding(251, config.speaker_embedding_size)
        self.lstm_stack = nn.LSTM(
            input_size=input_size + config.speaker_embedding_size,
            hidden_size=config.hidden_dim,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout
        )
        self.output_dropout = nn.Dropout(config.dropout, inplace=True)
        self.final_projection = nn.Linear(config.hidden_dim * 2, target_size)
        
    def forward(self, token_predictions, token_len, speaker_labels):
        """

        :param token_predictions: [B, T, target_size]
        :param token_len: [B]
        :param speaker_labels:  [B, 1]
        :return:
        """
        speaker_embeddings: torch.Tensor = self.speaker_embedding(torch.squeeze(speaker_labels, dim=1))
        # manually broadcast speaker embeddings to each time step
        speaker_embeddings = torch.repeat_interleave(
            torch.unsqueeze(speaker_embeddings, 1), token_predictions.size()[1], dim=1
        )  # [B, T, #SPK_EMB_SIZE]
        blstm_in = torch.concat([token_predictions, speaker_embeddings], dim=2)  # [B, T, F]
        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(blstm_in, token_len.to("cpu"), batch_first=True)
        blstm_packed_out, _ = self.lstm_stack(blstm_packed_in)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            blstm_packed_out, padding_value=0.0, batch_first=True
        )  # [B, T, F]
        blstm_out = self.output_dropout(blstm_out)

        return self.final_projection(blstm_out)


@dataclass
class Config:
    conv_hidden_size: int
    lstm_size: int
    speaker_embedding_size: int
    dropout: float
    final_dropout: float
    target_size: int
    tts_loss_from_epoch: int
    feature_extraction_config: DbMelFeatureExtractionConfig
    tts_predictor_config: TTSPredictorConfig

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = DbMelFeatureExtractionConfig.from_dict(d["feature_extraction_config"])
        d["tts_predictor_config"] = TTSPredictorConfig.from_dict(d["tts_predictor_config"])
        return Config(**d)


class Model(torch.nn.Module):
    """
    Default TTS aligner with 5 convolution blocks of size 5 followed by a BLSTM
    """

    def __init__(
        self,
        config: Union[Config, Dict[str, Any]],
        **kwargs,
    ):
        if isinstance(config, dict):
            config = Config.from_dict(config)
        super().__init__()
        self.lstm_size = config.lstm_size
        self.dropout = config.dropout
        self.final_dropout = config.final_dropout
        self.label_target_size = config.target_size
        self.tts_loss_from_epoch = config.tts_loss_from_epoch

        self.feature_extracton = DbMelFeatureExtraction(config=config.feature_extraction_config)
        self.prediction_size = config.feature_extraction_config.num_filters

        self.audio_embedding = nn.Linear(config.feature_extraction_config.num_filters, config.conv_hidden_size)
        if config.speaker_embedding_size > 0:
            self.speaker_embedding = nn.Embedding(251, config.speaker_embedding_size)
        else:
            self.speaker_embedding = None

        self.convs = nn.Sequential(
            Conv1DBlock(config.conv_hidden_size + config.speaker_embedding_size, config.conv_hidden_size,
                        filter_size=5, dropout=config.dropout),
            Conv1DBlock(config.conv_hidden_size, config.conv_hidden_size, filter_size=5, dropout=config.dropout),
            Conv1DBlock(config.conv_hidden_size, config.conv_hidden_size, filter_size=5, dropout=config.dropout),
            Conv1DBlock(config.conv_hidden_size, config.conv_hidden_size, filter_size=5, dropout=config.dropout),
            Conv1DBlock(config.conv_hidden_size, config.conv_hidden_size, filter_size=5, dropout=config.dropout),
        )
        self.blstm = nn.LSTM(input_size=config.conv_hidden_size, hidden_size=config.lstm_size, bidirectional=True, batch_first=True)
        self.final_linear = nn.Linear(2 * config.lstm_size, config.target_size)
        self.tts_predictor = TTSPredictor(
            config.tts_predictor_config,
            input_size=self.label_target_size,
            target_size=config.feature_extraction_config.num_filters,
        )


        self.final_dropout_layer = nn.Dropout1d(p=self.final_dropout, inplace=True) if self.final_dropout > 0 else None

        # initialize weights
        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(module: torch.nn.Module):
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            print("apply weight init for %s" % str(module))
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LSTM):
            print("apply weight init (LSTM) for %s" % str(module))
            for p in module.parameters():
                if p.dim() > 1: nn.init.xavier_uniform_(p)
                else: nn.init.zeros_(p)
        elif isinstance(module, torch.nn.Embedding):
            print("apply weight init (Embedding) for %s" % str(module))
            nn.init.normal_(module.weight, std=0.3)

    def forward(
        self,
        audio_features: torch.Tensor,
        speaker_labels: torch.Tensor,
        audio_features_len: torch.Tensor,
    ):
        """
        :param audio_features: [B, T, F]
        :param speaker_labels: [B, 1]
        :param audio_features_len: length of T as [B]
        :return: logprobs as [B, T, #PHONES]
        """

        squeezed_features = torch.squeeze(audio_features)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extracton(squeezed_features, audio_features_len)

        audio_embedding = self.audio_embedding(audio_features)  # [B, T, F]

        if self.speaker_embedding is not None:
            speaker_embeddings: torch.Tensor = self.speaker_embedding(torch.squeeze(speaker_labels, dim=1))
            # manually broadcast speaker embeddings to each time step
            speaker_embeddings = torch.repeat_interleave(
                torch.unsqueeze(speaker_embeddings, 1), audio_features.size()[1], dim=1
            )  # [B, T, #SPK_EMB_SIZE]

            conv_in = torch.concat([speaker_embeddings, audio_embedding], dim=2)  # [B, T, F]
        else:
            conv_in = audio_embedding

        conv_in = torch.swapaxes(conv_in, 1, 2)  # [B, F, T]
        conv_out = self.convs(conv_in)
        blstm_in = torch.permute(conv_out, dims=(0, 2, 1))  # [B, F, T] -> [B, T, F]

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(blstm_in, audio_features_len.to("cpu"), batch_first=True)
        blstm_packed_out, _ = self.blstm(blstm_packed_in)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            blstm_packed_out, padding_value=0.0, batch_first=True
        )  # [B, T, F]
        if self.final_dropout_layer is not None:
            blstm_out = self.final_dropout_layer(blstm_out.transpose(1, 2)).transpose(1, 2)
        logits = self.final_linear(blstm_out)  # [B, T, #PHONES]
        log_probs = torch.log_softmax(logits, dim=2)  # [B, T, #PHONES]

        real_probs = torch.exp(log_probs)

        predicted_features = self.tts_predictor(real_probs, audio_features_len, speaker_labels)

        return log_probs, audio_features_len, audio_features, predicted_features
        # return audio_features, audio_features_len


def train_step(*, model: Model, data, run_ctx, **kwargs):

    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)

    logprobs, audio_features_len, target_features, predicted_features = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
        speaker_labels=speaker_labels,
    )

    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    with torch.backends.cudnn.flags(deterministic=True):
        ctc_loss = nn.functional.ctc_loss(
            transposed_logprobs,
            phonemes,
            input_lengths=audio_features_len,
            target_lengths=phonemes_len,
            blank=model.label_target_size - 1,
            reduction="sum",
        )
    num_phonemes = torch.sum(phonemes_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes, scale=0.0)
    run_ctx.mark_as_loss(name="ctc_sum", loss=ctc_loss)

    if run_ctx.epoch >= model.tts_loss_from_epoch:
        mask = _lengths_to_op_mask(audio_features_len).unsqueeze(2)  # [B, T, 1]
        predictions = predicted_features * mask
        target_features = target_features * mask
        num_frames = torch.sum(audio_features_len)

        l1loss = nn.functional.l1_loss(predictions, target_features, reduction="sum") / model.prediction_size
        mse_loss = nn.functional.mse_loss(predictions, target_features, reduction="sum") / model.prediction_size
        run_ctx.mark_as_loss(name="rec_l1", loss=l1loss, inv_norm_factor=num_frames, scale=0.0)
        run_ctx.mark_as_loss(name="rec_l1_sum", loss=l1loss, scale=0.5)
        run_ctx.mark_as_loss(name="rec_mse", loss=mse_loss, inv_norm_factor=num_frames, scale=0.0)



############# FORWARD STUFF ################
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


# Duration extraction helpers are taken from
# https://github.com/as-ideas/DeepForcedAligner/blob/main/dfa/duration_extraction.py
# with commit id d1f565604bba25d4c56e3e12b289ab335980e069
# MIT license
def to_node_index(i, j, cols):
    return cols * i + j


def from_node_index(node_index, cols):
    return node_index // cols, node_index % cols


def to_adj_matrix(mat):
    """
    :param mat: [T x N] matrix where for each time frame we have the N scores of our target phoneme labels
    :return: A sparse CTC-style adjacent lattice matrix where the connection weight is the score of the "target" node
        of each connection.
    """
    rows = mat.shape[0]
    cols = mat.shape[1]

    row_ind = []
    col_ind = []
    data = []

    for i in range(rows):
        for j in range(cols):

            node = to_node_index(i, j, cols)

            if j < cols - 1:
                right_node = to_node_index(i, j + 1, cols)
                weight_right = mat[i, j + 1]
                row_ind.append(node)
                col_ind.append(right_node)
                data.append(weight_right)

            if i < rows - 1 and j < cols:
                bottom_node = to_node_index(i + 1, j, cols)
                weight_bottom = mat[i + 1, j]
                row_ind.append(node)
                col_ind.append(bottom_node)
                data.append(weight_bottom)

            if i < rows - 1 and j < cols - 1:
                bottom_right_node = to_node_index(i + 1, j + 1, cols)
                weight_bottom_right = mat[i + 1, j + 1]
                row_ind.append(node)
                col_ind.append(bottom_right_node)
                data.append(weight_bottom_right)

    adj_mat = coo_matrix((data, (row_ind, col_ind)), shape=(rows * cols, rows * cols))
    return adj_mat.tocsr()


@dataclass()
class AlignSequence:
    """
    :param logprobs: [T x F] log probabilities
    :param phonemes: [N] indexed tokens with indices in the range [0, F-1]
    """
    logprobs: np.ndarray
    phonemes: np.ndarray


def extract_durations_with_dijkstra(sequence: AlignSequence) -> np.array:
    """
    Extracts durations from the attention matrix by finding the shortest monotonic path from
    top left to bottom right.

    :return durations: [N] durations which sum to T
    """

    neg_log_weights = -sequence.logprobs[:, sequence.phonemes]
    adj_matrix = to_adj_matrix(neg_log_weights)
    dist_matrix, predecessors = dijkstra(csgraph=adj_matrix, directed=True, indices=0, return_predecessors=True)
    path = []
    pr_index = predecessors[-1]
    while pr_index != 0:
        path.append(pr_index)
        pr_index = predecessors[pr_index]
    path.reverse()

    # append first and last node
    path = [0] + path + [dist_matrix.size - 1]
    cols = neg_log_weights.shape[1]
    mel_text = {}
    durations = np.zeros(sequence.phonemes.shape[0], dtype=np.int32)

    # collect indices (mel, text) along the path
    for node_index in path:
        i, j = from_node_index(node_index, cols)
        mel_text[i] = j

    for j in mel_text.values():
        durations[j] += 1

    return durations


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    from returnn.datasets.hdf import SimpleHDFWriter
    run_ctx.hdf_writer = SimpleHDFWriter("output.hdf", dim=None, ndim=1)
    run_ctx.recognition_file = open("recog.txt", "wt")
    run_ctx.pool = multiprocessing.Pool(8)


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.hdf_writer.close()


def forward_step(*, model: Model, data, run_ctx, **kwargs):
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)

    tags = [data["seq_tag"][i] for i in list(indices.cpu().numpy())]

    logprobs, audio_features_len, target_features, predicted_features = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
        speaker_labels=speaker_labels,
    )

    numpy_logprobs = logprobs.detach().cpu().numpy()
    numpy_phonemes = phonemes.detach().cpu().numpy()

    align_sequences = []

    for single_logprobs, single_phonemes, feat_len, phon_len in zip(
        numpy_logprobs, numpy_phonemes, audio_features_len, phonemes_len
    ):
        align_sequences.append(AlignSequence(single_logprobs[:feat_len], single_phonemes[:phon_len]))

    durations = run_ctx.pool.map(extract_durations_with_dijkstra, align_sequences)
    for tag, duration, feat_len, phon_len in zip(tags, durations, audio_features_len, phonemes_len):
        total_sum = numpy.sum(duration)
        assert total_sum == feat_len
        assert len(duration) == phon_len
        run_ctx.hdf_writer.insert_batch(numpy.asarray([duration]), [len(duration)], [tag])
