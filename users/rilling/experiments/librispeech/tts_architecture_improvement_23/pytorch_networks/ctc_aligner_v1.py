from dataclasses import dataclass
import torch
import numpy
from torch import nn
import multiprocessing

from returnn.datasets.hdf import SimpleHDFWriter


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


class Model(torch.nn.Module):
    """
    Default TTS aligner with 5 convolution blocks of size 5 followed by a BLSTM
    """

    def __init__(
        self,
        conv_hidden_size: int,
        lstm_size: int,
        speaker_embedding_size: int,
        dropout: float,
        target_size: int,
        **kwargs,
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
        self.blstm = nn.LSTM(input_size=conv_hidden_size, hidden_size=lstm_size, bidirectional=True, batch_first=True)
        self.final_linear = nn.Linear(2 * lstm_size, target_size)
        self.lstm_size = lstm_size
        self.target_size = target_size
        self.dropout = dropout

        # initialize weights
        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(module: torch.nn.Module):
        if isinstance(module, torch.nn.Conv1d):
            nn.init.xavier_normal_(module.weight)

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
        speaker_embeddings: torch.Tensor = self.speaker_embedding(torch.squeeze(speaker_labels, dim=1))
        # manually broadcast speaker embeddings to each time step
        speaker_embeddings = torch.repeat_interleave(
            torch.unsqueeze(speaker_embeddings, 1), audio_features.size()[1], dim=1
        )  # [B, T, #SPK_EMB_SIZE]
        audio_embedding = self.audio_embedding(audio_features)  # [B, T, F]

        conv_in = torch.concat([speaker_embeddings, audio_embedding], dim=2)  # [B, T, F]
        conv_in = torch.swapaxes(conv_in, 1, 2)  # [B, F, T]
        conv_out = self.convs(conv_in)
        blstm_in = torch.permute(conv_out, dims=(0, 2, 1))  # [B, F, T] -> [B, T, F]

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(blstm_in, audio_features_len.to("cpu"), batch_first=True)
        blstm_packed_out, _ = self.blstm(blstm_packed_in)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            blstm_packed_out, padding_value=0.0, batch_first=True
        )  # [B, T, F]
        blstm_out = nn.functional.dropout(blstm_out, p=self.dropout, training=self.training)
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

    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        phonemes,
        input_lengths=audio_features_len,
        target_lengths=phonemes_len,
        blank=model.target_size - 1,
        reduction="sum",
    )
    num_frames = torch.sum(phonemes_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_frames)


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
    run_ctx.hdf_writer = SimpleHDFWriter("durations.hdf", dim=None, ndim=1)
    run_ctx.pool = multiprocessing.Pool(8)


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.hdf_writer.close()


def forward_step(*, model: Model, data, run_ctx, **kwargs):
    tags = data["seq_tag"]
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
