from dataclasses import dataclass
import torch
from torch import nn
import multiprocessing
import math
import os
import soundfile

from IPython import embed

from returnn.datasets.hdf import SimpleHDFWriter

from .shared import modules
from .shared import commons
from .shared import attentions


class TextEncoder(nn.Module):
    """
    Text Encoder model
    """

    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        window_size=None,
        block_length=None,
        mean_only=False,
        prenet=False,
        gin_channels=0,
    ):
        """Text Encoder Model based on Multi-Head Self-Attention combined with FF-CCNs

        Args:
            n_vocab (int): Size of vocabulary for embeddings
            out_channels (int): Number of output channels
            hidden_channels (int): Number of hidden channels
            filter_channels (int): Number of filter channels
            filter_channels_dp (int): Number of filter channels for duration predictor
            n_heads (int): Number of heads in encoder's Multi-Head Attention
            n_layers (int): Number of layers consisting of Multi-Head Attention and CNNs in encoder
            kernel_size (int): Kernel Size for CNNs in encoder layers
            p_dropout (float): Dropout probability for both encoder and duration predictor
            window_size (int, optional): Window size  in Multi-Head Self-Attention for encoder. Defaults to None.
            block_length (_type_, optional): Block length for optional block masking in Multi-Head Attention for encoder. Defaults to None.
            mean_only (bool, optional): Boolean to only project text encodings to mean values instead of mean and std. Defaults to False.
            prenet (bool, optional): Boolean to add ConvReluNorm prenet before encoder . Defaults to False.
            gin_channels (int, optional): Number of channels for speaker condition. Defaults to 0.
        """
        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.prenet = prenet
        self.gin_channels = gin_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if prenet:
            self.pre = modules.ConvReluNorm(
                hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5
            )
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
        )

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        # self.proj_w = DurationPredictor(hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)

        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        # print(f"Dimension of input in Text Encoder before DP: {x_dp.shape}")

        # logw = self.proj_w(x_dp, x_mask)
        return x_m, x_logs, x_mask


class Model(nn.Module):
    """
    Flow-based TTS model based on GlowTTS Structure
    Following the definition from https://arxiv.org/abs/2005.11129
    and code from https://github.com/jaywalnut310/glow-tts
    """

    def __init__(
        self,
        n_vocab: int,
        hidden_channels: int,
        filter_channels: int,
        filter_channels_dp: int,
        out_channels: int,
        kernel_size: int = 3,
        n_heads: int = 2,
        n_layers_enc: int = 6,
        p_dropout: float = 0.0,
        n_blocks_dec: int = 12,
        kernel_size_dec: int = 5,
        dilation_rate: int = 5,
        n_block_layers: int = 4,
        p_dropout_dec: float = 0.0,
        n_speakers: int = 0,
        gin_channels: int = 0,
        n_split: int = 4,
        n_sqz: int = 1,
        sigmoid_scale: bool = False,
        window_size: int = None,
        block_length: int = None,
        mean_only: bool = False,
        hidden_channels_enc: int = None,
        hidden_channels_dec: int = None,
        prenet: bool = False,
        p_speaker_drop=0,
        **kwargs,
    ):
        """_summary_

        Args:
            n_vocab (int): vocabulary size
            hidden_channels (int): Number of hidden channels in encoder
            filter_channels (int): Number of filter channels in encoder
            filter_channels_dp (int): Number of filter channels in decoder
            out_channels (int): Number of channels in the output
            kernel_size (int, optional): Size of kernels in the encoder. Defaults to 3.
            n_heads (int, optional): Number of heads in the Multi-Head Attention of the encoder. Defaults to 2.
            n_layers_enc (int, optional): Number of layers in the encoder. Defaults to 6.
            p_dropout (_type_, optional): Dropout probability in the encoder. Defaults to 0..
            n_blocks_dec (int, optional): Number of coupling blocks in the decoder. Defaults to 12.
            kernel_size_dec (int, optional): Kernel size in the decoder. Defaults to 5.
            dilation_rate (int, optional): Dilation rate for CNNs of coupling blocks in decoder. Defaults to 5.
            n_block_layers (int, optional): Number of layers in the CNN of the coupling blocks in decoder. Defaults to 4.
            p_dropout_dec (_type_, optional): Dropout probability in the decoder. Defaults to 0..
            n_speakers (int, optional): Number of speakers. Defaults to 0.
            gin_channels (int, optional): Number of speaker embedding channels. Defaults to 0.
            n_split (int, optional): Number of splits for the 1x1 convolution for flows in the decoder. Defaults to 4.
            n_sqz (int, optional): Squeeze. Defaults to 1.
            sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
            window_size (int, optional): Window size  in Multi-Head Self-Attention for encoder. Defaults to None.
            block_length (_type_, optional): Block length for optional block masking in Multi-Head Attention for encoder. Defaults to None.
            mean_only (bool, optional): Boolean to only project text encodings to mean values instead of mean and std. Defaults to False.
            hidden_channels_enc (int, optional): Number of hidden channels in encoder. Defaults to hidden_channels.
            hidden_channels_dec (_type_, optional): Number of hidden channels in decodder. Defaults to hidden_channels.
            prenet (bool, optional): Boolean to add ConvReluNorm prenet before encoder . Defaults to False.
        """
        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.n_layers_enc = n_layers_enc
        self.p_dropout = p_dropout
        self.n_blocks_dec = n_blocks_dec
        self.kernel_size_dec = kernel_size_dec
        self.dilation_rate = dilation_rate
        self.n_block_layers = n_block_layers
        self.p_dropout_dec = p_dropout_dec
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.prenet = prenet
        self.p_speaker_drop = p_speaker_drop

        self.encoder = TextEncoder(
            n_vocab,
            out_channels,
            hidden_channels_enc or hidden_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_layers_enc,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            mean_only=mean_only,
            prenet=prenet,
            gin_channels=gin_channels,
        )

        self.linear = nn.Linear(2 * out_channels, n_vocab + 1)

    def forward(
        self, x, x_lengths, raw_audio=None, raw_audio_lengths=None, g=None, gen=False, noise_scale=1.0, length_scale=1.0
    ):
        if not gen:
            with torch.no_grad():
                x_m, x_logs, x_mask = self.encoder(x, x_lengths, g=g)  # mean, std logs, duration logs, mask

        # breakpoint()
        encoder_out = torch.cat((x_m, x_logs), dim=1).transpose(1, 2)  # [B, T, F]

        linear_out = self.linear(encoder_out)

        log_probs = torch.log_softmax(linear_out, dim=2)

        return log_probs, torch.sum(x_mask, dim=1)

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    phon_labels = data["phon_labels"]  # [B, N] (sparse)
    phon_labels_len = data["phon_labels:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        x=phon_labels,
        x_lengths=phon_labels_len,
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        phon_labels,
        input_lengths=phon_labels_len,
        target_lengths=phon_labels_len,
        blank=model.n_vocab,
        reduction="sum",
        zero_infinity=True,
    )
    num_phonemes = torch.sum(phon_labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    from torchaudio.models.decoder import ctc_decoder

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")
    import subprocess

    if kwargs["arpa_lm"] is not None:
        lm = subprocess.check_output(["cf", kwargs["arpa_lm"]]).decode().strip()
    else:
        lm = None
    from returnn.datasets.util.vocabulary import Vocabulary

    vocab = Vocabulary.create_vocab(vocab_file=kwargs["returnn_vocab"], unknown_label=None)
    labels = vocab.labels
    print(f"labels from vocab:{labels}")

    print(f"Using ctc_decoder for TTS data...")

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=kwargs["lexicon"],
        lm=lm,
        lm_weight=kwargs["lm_weight"],
        tokens=labels + ["[SILENCE]", "[UNK]"],
        blank_token="[blank]",
        sil_token="[space]",  # [space] is our actual silence
        unk_word="[UNKNOWN]",
        nbest=1,
        beam_size=kwargs["beam_size"],
        beam_threshold=kwargs["beam_threshold"],
        sil_score=kwargs.get("sil_score", 0.0),
        word_score=kwargs.get("word_score", 0.0),
    )
    run_ctx.labels = labels
    run_ctx.blank_log_penalty = kwargs.get("blank_log_penalty", None)

    if kwargs.get("prior_file", None):
        run_ctx.prior = np.loadtxt(kwargs["prior_file"], dtype="float32")
        run_ctx.prior_scale = kwargs["prior_scale"]
    else:
        run_ctx.prior = None


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]
    phonemes = data["phon_labels"]
    phonemes_len = data["phon_labels:size1"]

    logprobs, audio_features_len = model(x=phonemes, x_lengths=phonemes_len)

    tags = data["seq_tag"]

    logprobs_cpu = logprobs.cpu()
    from IPython import embed
    embed()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())

    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
